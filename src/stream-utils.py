import sys
import json
import concurrent.futures
from typing import Dict, Any, Callable, Generator, List

if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def format_sse(event_type: str, data: Dict[str, Any]) -> str:
    """Formats an SSE message block with event name and JSON payload."""
    payload = {"event": event_type, "data": data}
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def execute_with_heartbeat(
    fn: Callable[..., Any],
    *args: Any,
    heartbeat_interval: float = 3.0,
    heartbeat_comment: str = ": keep-alive\n\n",
    **kwargs: Any
) -> Generator[str, None, Any]:
    """Runs a blocking callable in a thread while yielding periodic SSE heartbeat comments."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        while True:
            try:
                return future.result(timeout=heartbeat_interval)
            except concurrent.futures.TimeoutError:
                yield heartbeat_comment


def build_sentence_payload(
    sent_seg: Any, idx: int, total: int, s_start: float, s_end: float,
    sent_text: str, trans: str, phrases: List[Any], tokens: List[Any],
    hier_units: List[Any], oovs: List[Any]
) -> Dict[str, Any]:
    """Constructs a normalized dictionary payload for a processed sentence."""
    return {
        "sentence_id": sent_seg.segment_id,
        "sequence_number": idx,
        "total_sentences": total,
        "start_time": s_start,
        "end_time": s_end,
        "text": sent_text,
        "translation": trans,
        "bunsetsu_phrases": [
            {"phrase_id": b.phrase_id, "text": b.text, "translation": b.translation,
             "start_time": b.start_time, "end_time": b.end_time, "token_surfaces": b.token_surfaces}
            for b in phrases
        ],
        "tokens": [
            {"surface": t.surface, "reading": t.reading, "lemma": t.lemma,
             "pos": t.pos, "context_meaning": t.context_meaning}
            for t in tokens
        ],
        "knowledge_units": [u.to_dict() for u in hier_units],
        "oov_candidates": [
            {"term": o.term, "suggested_pos": o.suggested_pos,
             "suggested_meaning": o.suggested_meaning, "confidence_score": o.confidence_score}
            for o in oovs
        ]
    }


def log_sentence_breakdown(
    logger: Any, idx: int, total: int, s_start: float, s_end: float,
    t_sent: float, sent_text: str, s_trans: str, tokens: List[Any],
    phrases: List[Any], oovs: Any = None
) -> None:
    """Logs detailed sentence translation, token-meaning associations, bunsetsu, and OOVs."""
    logger.info(f"--- [Sentence #{idx}/{total}] ({s_start}s-{s_end}s) in {t_sent}s:")
    logger.info(f"    JP: \"{sent_text}\"")
    logger.info(f"    VI: \"{s_trans}\"")

    # Tokens with assigned contextual meanings
    logger.info("    [Tách Tokens & Gắn Nghĩa]:")
    for t in tokens:
        if getattr(t, "pos", "") not in ("PUNCTUATION",):
            meaning = getattr(t, "context_meaning", "")
            meaning_str = f": {meaning}" if meaning else ""
            logger.info(f"      • {t.surface} ({t.pos}){meaning_str}")

    # Grouped Bunsetsu phrases with chunk translations
    logger.info("    [Gắn Lại Bunsetsu]:")
    for b_idx, b in enumerate(phrases, start=1):
        trans = getattr(b, "translation", "")
        trans_str = f" -> {trans}" if trans else ""
        logger.info(f"      {b_idx}. [{b.text}]{trans_str}")

    # OOV Candidates (Từ mới ngoài từ điển)
    if oovs:
        logger.info(f"    [Từ Mới Ngoài Từ Điển (OOV)]: {len(oovs)} từ phát hiện")
        for o in oovs:
            pos_str = f" ({o.suggested_pos})" if getattr(o, "suggested_pos", None) else ""
            score = int(o.confidence_score * 100) if getattr(o, "confidence_score", None) else 90
            logger.info(f"      ★ {o.term}{pos_str} -> {o.suggested_meaning} [Độ tin cậy: {score}%]")
    else:
        logger.info("    [Từ Mới Ngoài Từ Điển (OOV)]: Không có (100% từ đã có sẵn trong từ điển)")


def push_oovs_to_backend(oovs: List[Any], logger: Any = None) -> int:
    """Sends discovered novel OOV candidates to Backend API for Curator review."""
    if not oovs:
        return 0
    import os
    import json
    import urllib.request

    backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:5000")
    ingest_url = f"{backend_url.rstrip('/')}/api/curator/oov/ingest"

    unique_oovs = {}
    for o in oovs:
        term = getattr(o, "term", "")
        if term and term not in unique_oovs:
            unique_oovs[term] = {
                "term": term,
                "confidenceScore": float(getattr(o, "confidence_score", 0.85) or 0.85),
                "tentativeReading": getattr(o, "tentative_reading", None) or term,
                "tentativePos": getattr(o, "suggested_pos", None) or "NOUN",
                "suggestedMeaning": getattr(o, "suggested_meaning", None) or "",
                "contextSnippet": getattr(o, "context_snippet", None) or ""
            }

    payload = list(unique_oovs.values())
    if not payload:
        return 0

    raw_urls = [os.environ.get("BACKEND_API_URL"), "http://localhost:5221", "http://localhost:5000"]
    urls = [u.rstrip("/") for u in raw_urls if u]
    data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for base_url in dict.fromkeys(urls):
        ingest_url = f"{base_url}/api/curator/oov/ingest"
        try:
            req = urllib.request.Request(
                ingest_url,
                data=data_bytes,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if logger:
                    logger.info(f"[OOV Ingest] Successfully pushed {len(payload)} OOV candidate(s) to Backend API at {base_url} ({resp.status}).")
                return len(payload)
        except Exception:
            continue

    if logger:
        logger.warning("[OOV Ingest Warning] Could not connect to Backend API. Please make sure Rakushu.Api is running on http://localhost:5221 or http://localhost:5000.")
    return 0

