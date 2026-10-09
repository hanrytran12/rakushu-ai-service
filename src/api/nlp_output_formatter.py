"""Shared response builders for GET and standalone POST NLP endpoints."""
from typing import Any, Dict, List, Optional

from src.services.bunsetsu_graph_service import BunsetsuGraphService


def flatten_ginza_tokens(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten the native GiNZA formatter token objects across sentence records."""
    tokens: List[Dict[str, Any]] = []
    for sentence in sentences:
        nlp = sentence.get("nlp", {})
        ginza_json = nlp.get("ginza_json", {})
        for paragraph in ginza_json.get("paragraphs", []):
            for ginza_sentence in paragraph.get("sentences", []):
                tokens.extend(ginza_sentence.get("tokens", []))
        if not ginza_json and nlp.get("format") == "rakushu-ginza-json-fallback":
            tokens.extend(nlp.get("tokens", []))
    return tokens


def build_tokens_response(
    video_id: Optional[str],
    sentences: List[Dict[str, Any]],
    output_format: str = "ginza-json",
) -> Dict[str, Any]:
    """Build the shared response contract used by GET Tokens and POST Tokens."""
    ginza_documents = []
    for sentence in sentences:
        nlp = sentence.get("nlp", {})
        ginza_documents.append(
            nlp.get("ginza_full", {
                "json": nlp.get("ginza_json", {}),
                "tokens": nlp.get("tokens", []),
            })
        )
    return {
        "success": True,
        "video_id": video_id,
        "format": output_format,
        "ginza": ginza_documents,
        "tokens": flatten_ginza_tokens(sentences),
    }


def build_bunsetsu_response(
    video_id: Optional[str],
    sentences: List[Dict[str, Any]],
    graph_service: BunsetsuGraphService,
) -> Dict[str, Any]:
    """Build the shared Bunsetsu graph contract from normalized sentence tokens."""
    response_sentences = []
    all_bunsetsu: List[Dict[str, Any]] = []
    all_relations: List[Dict[str, Any]] = []
    all_roots: List[int] = []
    bunsetsu_offset = 0

    for sentence in sentences:
        graph = graph_service.build(sentence.get("nlp", {}).get("tokens", []))
        sentence_bunsetsu = graph["bunsetsu"]
        sentence_relations = graph["relations"]
        sentence_roots = graph["roots"]

        for bunsetsu in sentence_bunsetsu:
            bunsetsu["id"] += bunsetsu_offset
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

        global_roots = [root + bunsetsu_offset for root in sentence_roots]
        all_roots.extend(global_roots)
        all_bunsetsu.extend(sentence_bunsetsu)
        all_relations.extend(sentence_relations)
        response_sentences.append({
            "sentence_number": sentence["sentence_number"],
            "text": sentence["text"],
            "bunsetsu": sentence_bunsetsu,
            "relations": sentence_relations,
            "roots": global_roots,
        })
        bunsetsu_offset += len(sentence_bunsetsu)

    return {
        "success": True,
        "video_id": video_id,
        "bunsetsu": all_bunsetsu,
        "relations": all_relations,
        "roots": all_roots,
        "sentences": response_sentences,
    }
