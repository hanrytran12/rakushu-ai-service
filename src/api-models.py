"""Pydantic API request and response models for Rakushu Video Pipeline."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ProcessUrlRequest(BaseModel):
    """Payload for importing video via URL."""
    url: str = Field(..., description="YouTube URL or direct media URL", min_length=5)
    max_duration_seconds: Optional[int] = Field(default=600, description="Max allowed video duration in seconds")


class TokenItem(BaseModel):
    surface: str
    reading: Optional[str] = ""
    lemma: Optional[str] = ""
    pos: Optional[str] = ""
    xpos: Optional[str] = ""
    deprel: Optional[str] = ""
    context_meaning: Optional[str] = ""


class BunsetsuItem(BaseModel):
    phrase_id: str
    text: str
    translation: str
    start_time: float
    end_time: float
    token_surfaces: List[str] = []


class KnowledgeUnitItem(BaseModel):
    unit_type: str  # GRAMMAR | PHRASE | COMPOUND_WORD | WORD
    surface: str
    reading: Optional[str] = ""
    meaning: Optional[str] = ""
    start_token_idx: int = 0
    end_token_idx: int = 0
    grammar_info: Optional[Dict[str, Any]] = None
    phrase_info: Optional[Dict[str, Any]] = None
    compound_info: Optional[Dict[str, Any]] = None
    word_info: Optional[Dict[str, Any]] = None


class OovItem(BaseModel):
    term: str
    suggested_pos: Optional[str] = ""
    suggested_meaning: Optional[str] = ""
    confidence_score: float = 0.0


class SentenceItem(BaseModel):
    sentence_id: str
    sequence_number: int
    start_time: float
    end_time: float
    text: str
    translation: str
    tokens: List[TokenItem] = []
    bunsetsu_phrases: List[BunsetsuItem] = []
    knowledge_units: List[KnowledgeUnitItem] = []
    oov_candidates: List[OovItem] = []


class PipelineSummary(BaseModel):
    total_sentences: int = 0
    total_tokens: int = 0
    total_grammar_matched: int = 0
    total_phrases_matched: int = 0
    total_compound_words_matched: int = 0
    total_oov_discovered: int = 0


class PipelineData(BaseModel):
    video_id: str
    video_title: str
    source_url: str
    duration: float
    global_transcript: str
    global_translation: str
    sentences: List[SentenceItem] = []
    summary: PipelineSummary


class PipelineApiResponse(BaseModel):
    success: bool = True
    data: Optional[PipelineData] = None
    message: Optional[str] = None
