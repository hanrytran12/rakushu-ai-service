"""Data models and API schemas for Linguistic Curator and OOV Lifecycle."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import uuid
from pydantic import BaseModel, Field


class OovStatus:
    PENDING_CURATOR_REVIEW = "PENDING_CURATOR_REVIEW"
    DISCARDED = "DISCARDED"
    ADAPTED = "ADAPTED"
    REJECTED = "REJECTED"
    RESOLVED_MANUAL = "RESOLVED_MANUAL"


class CuratorDecision:
    ADAPT = "ADAPT"
    REJECT = "REJECT"
    MANUAL_ADD = "MANUAL_ADD"


@dataclass
class CuratorReview:
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    oov_candidate_id: str = ""
    curator_id: str = "curator-01"
    decision: str = CuratorDecision.ADAPT
    edited_term: str = ""
    edited_reading: str = ""
    edited_pos: str = ""
    edited_meaning: str = ""
    comment: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CuratorReviewRequest(BaseModel):
    """Payload when a curator adapts or rejects an AI OOV candidate."""
    oov_candidate_id: str = Field(..., description="ID of OOV candidate to review")
    curator_id: str = Field(default="curator-01", description="Curator user ID")
    decision: str = Field(..., description="Decision: 'ADAPT' or 'REJECT'")
    edited_term: Optional[str] = Field(None, description="Modified or confirmed term")
    edited_reading: Optional[str] = Field(None, description="Modified or confirmed reading")
    edited_pos: Optional[str] = Field(None, description="Modified or confirmed POS tag")
    edited_meaning: Optional[str] = Field(None, description="Modified or confirmed definition")
    comment: Optional[str] = Field("", description="Optional review note")


class CuratorManualAddRequest(BaseModel):
    """Payload when curator rejects AI identification but chooses to add token manually."""
    oov_candidate_id: str = Field(..., description="ID of OOV candidate being rejected")
    curator_id: str = Field(default="curator-01", description="Curator user ID")
    term: str = Field(..., description="Manually input Japanese term", min_length=1)
    reading: str = Field(..., description="Manually input phonetic reading", min_length=1)
    pos: str = Field(default="NOUN", description="Manually input POS tag")
    meaning: str = Field(..., description="Manually input definition", min_length=1)
    comment: Optional[str] = Field("", description="Review comment or justification")


class OovCandidateResponseItem(BaseModel):
    oov_candidate_id: str
    token_id: str
    term: str
    tentative_reading: Optional[str] = ""
    tentative_pos: Optional[str] = ""
    suggested_meaning: Optional[str] = ""
    context_snippet: Optional[str] = ""
    confidence_score: float = 0.0
    status: str
    detected_at: Optional[str] = ""


class OovCandidateListResponse(BaseModel):
    success: bool = True
    total: int = 0
    items: List[OovCandidateResponseItem] = []


class CuratorActionResponse(BaseModel):
    success: bool = True
    message: str = ""
    oov_candidate_id: str = ""
    status: str = ""
    merged_term: Optional[str] = None
    review: Optional[Dict[str, Any]] = None


class DictionaryLookupItem(BaseModel):
    term: str
    reading: str
    pos: str
    meaning: str
    definition_tags: Optional[str] = ""
    source: Optional[str] = "dictionary"


class DictionaryLookupResponse(BaseModel):
    found: bool = False
    entry: Optional[DictionaryLookupItem] = None
    message: Optional[str] = None
