"""FastAPI Router for Video Import & Processing Pipeline Endpoints."""
import os
import uuid
import shutil
import logging
import importlib
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from typing import Dict, Any

_models = importlib.import_module(".api-models", package="src")
ProcessUrlRequest = _models.ProcessUrlRequest
PipelineApiResponse = _models.PipelineApiResponse

_formatter = importlib.import_module(".pipeline-formatter", package="src")
format_pipeline_result = _formatter.format_pipeline_result

_pipeline_runner = importlib.import_module(".pipeline-runner", package="src")
run_pipeline = _pipeline_runner.run_pipeline

_inspector = importlib.import_module(".media-inspector", package="src")
InvalidMediaError = _inspector.InvalidMediaError
InvalidLanguageError = _inspector.InvalidLanguageError
ProhibitedContentError = _inspector.ProhibitedContentError

logger = logging.getLogger("PipelineRouter")
router = APIRouter(prefix="/api/v1/pipeline", tags=["Video Pipeline"])

TEMP_UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "temp_uploads"))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4a", ".mp3", ".wav", ".mkv", ".webm"}


@router.get("/health")
def pipeline_health() -> Dict[str, Any]:
    """Health status of the video import pipeline service."""
    return {
        "status": "UP",
        "service": "rakushu-video-pipeline",
        "supported_inputs": ["youtube_url", "direct_url", "multipart_upload"]
    }


@router.post("/process-url", response_model=PipelineApiResponse)
def process_video_url_endpoint(req: ProcessUrlRequest) -> PipelineApiResponse:
    """Imports video from a YouTube or direct media URL, transcribes, and enriches linguistic data."""
    logger.info(f"Received pipeline URL request: {req.url}")
    try:
        pipeline_result = run_pipeline(req.url)
        if pipeline_result.segment.text.startswith("[REJECTED]"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=pipeline_result.segment.text
            )
        return format_pipeline_result(pipeline_result)
    except (InvalidMediaError, InvalidLanguageError, ProhibitedContentError) as err:
        logger.warning(f"Media validation error on URL: {err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as exc:
        logger.error(f"Unexpected error running pipeline on URL: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal pipeline processing failure: {str(exc)}"
        )


@router.post("/upload-video", response_model=PipelineApiResponse)
def upload_video_endpoint(file: UploadFile = File(...)) -> PipelineApiResponse:
    """Accepts uploaded video/audio file, processes through ASR -> NLP -> Knowledge -> LLM."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}"
        )

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    temp_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)

    try:
        logger.info(f"Saving uploaded file to temporary path: {temp_path}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline_result = run_pipeline(temp_path)
        if pipeline_result.segment.text.startswith("[REJECTED]"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=pipeline_result.segment.text
            )
        return format_pipeline_result(pipeline_result)
    except (InvalidMediaError, InvalidLanguageError, ProhibitedContentError) as err:
        logger.warning(f"Media validation error on uploaded file: {err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error processing uploaded video: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal pipeline upload processing failure: {str(exc)}"
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temporary uploaded file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")
