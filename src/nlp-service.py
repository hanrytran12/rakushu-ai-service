"""NLP Service: Japanese morphological tokenizer producing structured tokens."""
import re
from typing import List
from src import TokenModel, SubtitleSegment

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    _HAS_JANOME = True
except ImportError:
    _HAS_JANOME = False


class NlpService:
    """Performs morphological analysis, lemmatization, POS tagging, and reading generation."""

    POS_MAP = {
        "名詞": "NOUN",
        "動詞": "VERB",
        "形容詞": "ADJECTIVE",
        "助詞": "PARTICLE",
        "助動詞": "AUX_VERB",
        "記号": "PUNCTUATION",
        "感動詞": "INTERJECTION",
        "副詞": "ADVERB",
        "接続詞": "CONJUNCTION",
        "連体詞": "DETERMINER",
    }

    def __init__(self):
        self.tokenizer = JanomeTokenizer() if _HAS_JANOME else None

    def tokenize(self, segment: SubtitleSegment) -> List[TokenModel]:
        """Tokenizes Japanese segment text into rich TokenModel instances."""
        if self.tokenizer:
            return self._tokenize_janome(segment)
        return self._tokenize_fallback(segment)

    def _tokenize_janome(self, segment: SubtitleSegment) -> List[TokenModel]:
        raw_tokens: List[TokenModel] = []
        pos = 0
        text = segment.text

        for item in self.tokenizer.tokenize(text):
            surface = item.surface
            main_pos = item.part_of_speech.split(",")[0]
            mapped_pos = self.POS_MAP.get(main_pos, "OTHER")
            lemma = item.base_form if item.base_form != "*" else surface
            reading = item.reading if item.reading != "*" else surface

            start = text.find(surface, pos)
            if start == -1:
                start = pos
            end = start + len(surface)
            pos = end

            raw_tokens.append(TokenModel(
                segment_id=segment.segment_id,
                surface=surface,
                pos=mapped_pos,
                lemma=lemma,
                reading=reading,
                romaji="",
                start_position=start,
                end_position=end
            ))

        # Merge compound nouns (e.g. 若者 + 言葉 -> 若者言葉)
        merged: List[TokenModel] = []
        for tok in raw_tokens:
            if merged and merged[-1].pos == "NOUN" and tok.pos == "NOUN":
                prev = merged[-1]
                prev.surface += tok.surface
                prev.lemma += tok.lemma
                prev.reading += tok.reading
                prev.end_position = tok.end_position
            else:
                merged.append(tok)
        return merged

    def _tokenize_fallback(self, segment: SubtitleSegment) -> List[TokenModel]:
        text = segment.text
        tokens: List[TokenModel] = []
        pos = 0
        for ch in text:
            tokens.append(TokenModel(
                segment_id=segment.segment_id,
                surface=ch,
                pos="PUNCTUATION" if ch in "、。" else "NOUN",
                lemma=ch,
                reading=ch,
                romaji="",
                start_position=pos,
                end_position=pos + 1
            ))
            pos += 1
        return tokens
