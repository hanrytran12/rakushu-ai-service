"""FastAPI Router for Video Import & Processing Pipeline Streaming Endpoints."""
import os
import uuid
import shutil
import logging
import importlib
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

_models = importlib.import_module(".api-models", package="src")
ProcessUrlRequest = _models.ProcessUrlRequest

_pipeline_streamer = importlib.import_module(".pipeline-streamer", package="src")
stream_pipeline = _pipeline_streamer.stream_pipeline

logger = logging.getLogger("PipelineRouter")
router = APIRouter(prefix="/api/v1/pipeline", tags=["Video Pipeline"])

TEMP_UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "temp_uploads"))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4a", ".mp3", ".wav", ".mkv", ".webm"}


def _cleanup_file(path: str) -> None:
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _save_upload(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file format '{ext}'.")
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{upload_file.filename or 'upload'}")
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(upload_file.file, buf)
    return temp_path


def _stream_response(generator, background: Optional[BackgroundTask] = None) -> StreamingResponse:
    return StreamingResponse(
        generator, media_type="text/event-stream", background=background,
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@router.get("/health")
def pipeline_health() -> Dict[str, Any]:
    """Health status of the streaming video import pipeline service."""
    return {
        "status": "UP",
        "service": "rakushu-video-pipeline",
        "supported_inputs": ["youtube_url", "direct_url", "multipart_upload"],
        "streaming_supported": True
    }


@router.post("/stream-url")
@router.post("/process-url", include_in_schema=False)
@router.post("/process-url/stream", include_in_schema=False)
def stream_video_url_endpoint(req: ProcessUrlRequest) -> StreamingResponse:
    """Streams real-time SSE events (ASR -> Global Translation -> Sentence-by-sentence) from YouTube or media URL."""
    logger.info(f">>> [Stream Request] Received URL: '{req.url}'")
    return _stream_response(stream_pipeline(req.url))


@router.post("/stream-upload")
@router.post("/upload-video", include_in_schema=False)
@router.post("/upload-stream", include_in_schema=False)
@router.post("/upload-video/stream", include_in_schema=False)
@router.post("/stream-video", include_in_schema=False)
def stream_video_upload_endpoint(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
) -> StreamingResponse:
    """Uploads video file and streams real-time SSE events progressive results to client."""
    upload_file = file or video
    if not upload_file:
        raise HTTPException(status_code=400, detail="Missing file. Send 'file' or 'video' form field.")

    logger.info(f">>> [Stream Request] Uploaded file: '{upload_file.filename}'")
    temp_path = _save_upload(upload_file)
    return _stream_response(stream_pipeline(temp_path), background=BackgroundTask(_cleanup_file, temp_path))
