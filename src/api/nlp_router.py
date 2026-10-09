"""Read-only and standalone POST APIs for NLP token and Bunsetsu output."""
from typing import Any, Dict, List

from fastapi import APIRouter, Body, HTTPException, status

from src.api.api_models import TranscriptionRequest
from src.api.nlp_output_formatter import (
    build_bunsetsu_response,
    build_tokens_response,
)
from src.models import SubtitleSegment
from src.services.bunsetsu_graph_service import BunsetsuGraphService
from src.services.nlp_service import NlpService
from src.services.pipeline_output_store import pipeline_output_store

router = APIRouter(prefix="/api/v1/nlp", tags=["NLP Output"])
_bunsetsu_graph_service = BunsetsuGraphService()
_nlp_service = NlpService()


def _get_completed_output(video_id: str) -> Dict[str, Any]:
    output = pipeline_output_store.get(video_id)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No completed pipeline output found for video_id '{video_id}'.",
        )
    return output


def _validate_raw_tokens(tokens: List[Dict[str, Any]]) -> None:
    """Reject token payloads that cannot safely form a Bunsetsu graph."""
    seen_ids = set()
    for index, token in enumerate(tokens):
        token_id = token.get("id")
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"tokens[{index}].id must be a positive integer.",
            )
        if token_id in seen_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"tokens[{index}].id duplicates another token id.",
            )
        seen_ids.add(token_id)
        if not isinstance(token.get("orth"), str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"tokens[{index}].orth must be a string.",
            )
        head_id = token.get("head_absolute")
        if not isinstance(head_id, int) or isinstance(head_id, bool):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"tokens[{index}].head_absolute must be an integer.",
            )
        label = token.get("bunsetu_bi_label")
        if not isinstance(label, str) or label not in {"B", "I"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"tokens[{index}].bunsetu_bi_label must be 'B' or 'I'.",
            )
        if not isinstance(token.get("dep"), str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"tokens[{index}].dep must be a string.",
            )
        if not isinstance(token.get("bunsetu_position_type"), str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"tokens[{index}].bunsetu_position_type must be a string.",
            )


    token_ids = {token["id"] for token in tokens}
    for index, token in enumerate(tokens):
        head_id = token["head_absolute"]
        if head_id != 0 and head_id not in token_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"tokens[{index}].head_absolute references an unknown token id.",
            )


@router.get("/videos/{video_id}/tokens")
def get_pipeline_tokens(video_id: str) -> Dict[str, Any]:
    """Return GiNZA token analysis already produced by a completed pipeline."""
    output = _get_completed_output(video_id)
    return build_tokens_response(video_id, output["sentences"])


@router.post("/tokens")
def post_tokens(request: TranscriptionRequest) -> Dict[str, Any]:
    """Analyze supplied Japanese transcription without running the video pipeline."""
    transcription = request.transcription.strip()
    if not transcription:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="transcription must not be blank.",
        )

    sentences = []
    for index, sentence_text in enumerate(
        _nlp_service.split_sentences(transcription), start=1
    ):
        analysis = _nlp_service.analyze_json(SubtitleSegment(text=sentence_text))
        sentences.append({
            "sentence_number": index,
            "text": sentence_text,
            "nlp": analysis,
        })

    output_format = (
        sentences[0]["nlp"].get("format", "ginza-json")
        if sentences else "ginza-json"
    )
    return build_tokens_response(None, sentences, output_format=output_format)


@router.get("/videos/{video_id}/bunsetsu")
def get_pipeline_bunsetsu(video_id: str) -> Dict[str, Any]:
    """Build Bunsetsu and dependency relations from stored GiNZA token JSON."""
    output = _get_completed_output(video_id)
    return build_bunsetsu_response(
        video_id, output["sentences"], _bunsetsu_graph_service
    )


@router.post("/bunsetsu")
def post_bunsetsu(
    tokens: List[Dict[str, Any]] = Body(
        ..., description="Raw array of normalized GiNZA token objects"
    ),
) -> Dict[str, Any]:
    """Build a Bunsetsu graph directly from normalized token objects."""
    _validate_raw_tokens(tokens)
    if not tokens:
        return build_bunsetsu_response(None, [], _bunsetsu_graph_service)

    sentence_text = "".join(token["orth"] for token in tokens)
    sentence = {
        "sentence_number": 1,
        "text": sentence_text,
        "nlp": {"tokens": tokens},
    }
    return build_bunsetsu_response(None, [sentence], _bunsetsu_graph_service)
