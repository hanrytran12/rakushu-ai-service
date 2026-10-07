"""Read-only APIs for NLP outputs produced by a completed video pipeline."""
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status

from src.services.bunsetsu_graph_service import BunsetsuGraphService
from src.services.pipeline_output_store import pipeline_output_store

router = APIRouter(prefix="/api/v1/nlp", tags=["NLP Output"])
_bunsetsu_graph_service = BunsetsuGraphService()


def _get_completed_output(video_id: str) -> Dict[str, Any]:
    output = pipeline_output_store.get(video_id)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No completed pipeline output found for video_id '{video_id}'.",
        )
    return output


def _flatten_ginza_tokens(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten the exact token objects emitted by GiNZA's JSON formatter."""
    tokens: List[Dict[str, Any]] = []
    for sentence in sentences:
        ginza_json = sentence.get("nlp", {}).get("ginza_json", {})
        for paragraph in ginza_json.get("paragraphs", []):
            for ginza_sentence in paragraph.get("sentences", []):
                tokens.extend(ginza_sentence.get("tokens", []))
    return tokens


@router.get("/videos/{video_id}/tokens")
def get_pipeline_tokens(video_id: str) -> Dict[str, Any]:
    """Return the GiNZA token analysis already produced by the completed pipeline."""
    output = _get_completed_output(video_id)
    ginza_documents = [
        sentence["nlp"]["ginza_full"]
        for sentence in output["sentences"]
    ]
    return {
        "success": True,
        "video_id": video_id,
        "format": "ginza-json",
        "ginza": ginza_documents,
        "tokens": _flatten_ginza_tokens(output["sentences"]),
    }


@router.get("/videos/{video_id}/bunsetsu")
def get_pipeline_bunsetsu(video_id: str) -> Dict[str, Any]:
    """Build Bunsetsu and dependency relations from stored GiNZA token JSON."""
    output = _get_completed_output(video_id)
    sentences = []

    all_bunsetsu: List[Dict[str, Any]] = []
    all_relations: List[Dict[str, Any]] = []
    all_roots: List[int] = []
    bunsetsu_offset = 0

    for sentence in output["sentences"]:
        graph = _bunsetsu_graph_service.build(
            sentence.get("nlp", {}).get("tokens", [])
        )
        sentence_bunsetsu = graph["bunsetsu"]
        sentence_relations = graph["relations"]
        sentence_roots = graph["roots"]

        for bunsetsu in sentence_bunsetsu:
            local_id = bunsetsu["id"]
            global_id = bunsetsu_offset + local_id
            bunsetsu["id"] = global_id
            bunsetsu["sentence_number"] = sentence["sentence_number"]
            bunsetsu["sentence_text"] = sentence["text"]
            if bunsetsu["dep"]["to"]:
                bunsetsu["dep"]["to"] += bunsetsu_offset
            bunsetsu["children"] = [
                child + bunsetsu_offset for child in bunsetsu["children"]
            ]
            for case_frame in bunsetsu["case_frame"]:
                case_frame["bunsetsu"] += bunsetsu_offset

        for relation in sentence_relations:
            relation["sentence_number"] = sentence["sentence_number"]
            relation["from_bunsetsu"] += bunsetsu_offset
            relation["to_bunsetsu"] += bunsetsu_offset

        global_sentence_roots = [
            root + bunsetsu_offset for root in sentence_roots
        ]
        all_roots.extend(global_sentence_roots)
        all_bunsetsu.extend(sentence_bunsetsu)
        all_relations.extend(sentence_relations)
        sentences.append({
            "sentence_number": sentence["sentence_number"],
            "text": sentence["text"],
            "bunsetsu": sentence_bunsetsu,
            "relations": sentence_relations,
            "roots": global_sentence_roots,
        })
        bunsetsu_offset += len(sentence_bunsetsu)

    return {
        "success": True,
        "video_id": video_id,
        "bunsetsu": all_bunsetsu,
        "relations": all_relations,
        "roots": all_roots,
        "sentences": sentences,
    }
