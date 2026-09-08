"""Data models aligned with the Rakushu Capstone Logical ERD."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import uuid


@dataclass
class SubtitleSegment:
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    video_id: str = "video-sample-01"
    start_time: float = 0.0
    end_time: float = 10.0
    text: str = ""
    sequence_number: int = 1

    def to_dict(self):
        return asdict(self)


@dataclass
class TokenModel:
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment_id: str = ""
    surface: str = ""
    pos: str = ""
    lemma: str = ""
    reading: str = ""
    romaji: str = ""
    start_position: int = 0
    end_position: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class DictionaryEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    term: str = ""
    reading: str = ""
    pos: str = ""
    meaning: str = ""
    jlpt_level: str = "N5"
    commonality: int = 1

    def to_dict(self):
        return asdict(self)


@dataclass
class OovCandidate:
    oov_candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_id: str = ""
    term: str = ""
    suggested_meaning: str = ""
    suggested_pos: str = ""
    context_snippet: str = ""
    confidence_score: float = 0.0
    status: str = "PENDING_CURATOR_REVIEW"

    def to_dict(self):
        return asdict(self)


@dataclass
class BunsetsuPhrase:
    phrase_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment_id: str = ""
    text: str = ""
    translation: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    phrase_order: int = 0
    token_surfaces: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class PipelineResult:
    segment: SubtitleSegment
    tokens: List[TokenModel]
    matched_knowledge: List[DictionaryEntry]
    oov_candidates: List[OovCandidate]
    bunsetsu_phrases: List[BunsetsuPhrase]

    def to_dict(self):
        return {
            "segment": self.segment.to_dict(),
            "tokens": [t.to_dict() for t in self.tokens],
            "matched_knowledge": [k.to_dict() for k in self.matched_knowledge],
            "oov_candidates": [o.to_dict() for o in self.oov_candidates],
            "bunsetsu_phrases": [p.to_dict() for p in self.bunsetsu_phrases],
        }
