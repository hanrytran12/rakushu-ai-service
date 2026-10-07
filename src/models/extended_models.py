"""Data models for Grammar, Phrase, and CompoundWord Knowledge."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class GrammarEntry:
    id: Optional[int] = None
    pattern: str = ""
    reading: str = ""
    jlpt_level: str = ""  # N5, N4, N3, N2, N1
    formation: str = ""
    meaning: str = ""
    nuance: str = ""
    examples: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhraseEntry:
    id: Optional[int] = None
    term: str = ""
    reading: str = ""
    phrase_type: str = ""  # IDIOM, YOJIJUKUGO, PROVERB, EXPRESSION
    meaning: str = ""
    tags: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompoundWordEntry:
    id: Optional[int] = None
    term: str = ""
    reading: str = ""
    compound_type: str = ""  # VERB, NOUN
    components: List[str] = field(default_factory=list)
    components_reading: List[str] = field(default_factory=list)
    transitivity: str = ""
    meaning: str = ""
    structure: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchedKnowledgeUnit:
    """Unified container for hierarchical knowledge match (Grammar, Phrase, CompoundWord, or Word)."""
    unit_type: str = "WORD"  # GRAMMAR | PHRASE | COMPOUND_WORD | WORD
    surface: str = ""
    reading: str = ""
    meaning: str = ""
    start_token_idx: int = 0
    end_token_idx: int = 0
    grammar_info: Optional[GrammarEntry] = None
    phrase_info: Optional[PhraseEntry] = None
    compound_info: Optional[CompoundWordEntry] = None
    word_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_type": self.unit_type,
            "surface": self.surface,
            "reading": self.reading,
            "meaning": self.meaning,
            "start_token_idx": self.start_token_idx,
            "end_token_idx": self.end_token_idx,
            "grammar_info": self.grammar_info.to_dict() if self.grammar_info else None,
            "phrase_info": self.phrase_info.to_dict() if self.phrase_info else None,
            "compound_info": self.compound_info.to_dict() if self.compound_info else None,
            "word_info": self.word_info,
        }
