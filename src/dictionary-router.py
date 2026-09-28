"""FastAPI Router for Dictionary Lookup and Internal Knowledge Synchronization."""
import importlib
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, status

_models = importlib.import_module(".schema-models", package="src")
DictionaryEntry = _models.DictionaryEntry

_knowledge_mod = importlib.import_module(".knowledge-service", package="src")
KnowledgeService = _knowledge_mod.KnowledgeService

router = APIRouter(prefix="/api/v1", tags=["Dictionary & Knowledge Base"])
_knowledge_svc = KnowledgeService()


class SyncDictionaryEntryRequest(BaseModel):
    term: str = Field(..., min_length=1)
    reading: str = Field(..., min_length=1)
    pos: str = Field(default="NOUN")
    meaning: str = Field(..., min_length=1)
    definition_tags: Optional[str] = "curator-verified"
    original_term: Optional[str] = None
    status: Optional[str] = "ADAPTED"


class SyncOovStatusRequest(BaseModel):
    term: str = Field(..., min_length=1)
    status: str = Field(..., min_length=1)
    candidate_id: Optional[str] = None


class DictionaryLookupItem(BaseModel):
    term: str
    reading: str
    pos: str
    meaning: str
    definition_tags: Optional[str] = ""


class DictionaryLookupResponse(BaseModel):
    found: bool = False
    entry: Optional[DictionaryLookupItem] = None
    message: Optional[str] = None


@router.get("/dictionary/lookup", response_model=DictionaryLookupResponse)
def dictionary_lookup(term: str = Query(..., description="Japanese word to lookup")):
    """Instant offline dictionary lookup for Learner query."""
    entry = _knowledge_svc.lookup(term)
    if not entry:
        return DictionaryLookupResponse(found=False, entry=None, message=f"Term '{term}' not found in dictionary")
    return DictionaryLookupResponse(
        found=True,
        entry=DictionaryLookupItem(
            term=entry.term, reading=entry.reading, pos=entry.pos,
            meaning=entry.meaning, definition_tags=entry.definition_tags or ""
        )
    )


@router.post("/internal/dictionary/sync")
def sync_dictionary_entry(req: SyncDictionaryEntryRequest):
    """Internal API called by BE Service to synchronize approved/manual words into SQLite dictionary."""
    entry = DictionaryEntry(
        term=req.term, reading=req.reading, pos=req.pos,
        meaning=req.meaning, definition_tags=req.definition_tags or "curator-verified",
        rules="", score=100, sequence=1, term_tags="OOV_ENRICHED"
    )
    _knowledge_svc.register_enriched_oov(
        entry,
        new_status=req.status or "ADAPTED",
        original_term=req.original_term
    )
    return {"success": True, "message": f"Term '{req.term}' synchronized into SQLite dictionary."}


@router.post("/internal/oov/sync-status")
def sync_oov_status(req: SyncOovStatusRequest):
    """Internal API called by BE Service to synchronize candidate status (e.g. REJECTED, ADAPTED)."""
    updated = _knowledge_svc.update_oov_status(
        term=req.term,
        new_status=req.status,
        candidate_id=req.candidate_id
    )
    return {"success": True, "updated": updated, "message": f"Status for '{req.term}' updated to '{req.status}'."}
