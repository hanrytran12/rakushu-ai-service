"""In-memory store for completed video pipeline NLP output."""
from threading import Lock
from typing import Any, Dict, List, Optional


class PipelineOutputStore:
    """Keeps completed NLP output keyed by video_id for read-only APIs."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._outputs: Dict[str, Dict[str, Any]] = {}

    def save(self, video_id: str, nlp_sentences: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._outputs[video_id] = {
                "video_id": video_id,
                "sentences": list(nlp_sentences),
            }

    def get(self, video_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            output = self._outputs.get(video_id)
            if output is None:
                return None
            return {
                "video_id": output["video_id"],
                "sentences": list(output["sentences"]),
            }


pipeline_output_store = PipelineOutputStore()
