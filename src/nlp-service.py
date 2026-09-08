"""NLP Service: Japanese morphological tokenizer and sentence segmenter using GiNZA."""
from typing import List, Optional
from src import TokenModel, SubtitleSegment

try:
    import spacy
    import ginza
    _GINZA_NLP = spacy.load("ja_ginza")
    _HAS_GINZA = True
except Exception:
    _GINZA_NLP = None
    _HAS_GINZA = False

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    _JANOME_TOKENIZER = JanomeTokenizer()
    _HAS_JANOME = True
except ImportError:
    _JANOME_TOKENIZER = None
    _HAS_JANOME = False


class NlpService:
    """Japanese morphological analysis, POS tagging, and sentence splitting via GiNZA."""

    GINZA_POS_MAP = {
        "NOUN": "NOUN", "PROPN": "NOUN", "VERB": "VERB", "ADJ": "ADJECTIVE",
        "ADP": "PARTICLE", "SCONJ": "PARTICLE", "AUX": "AUX_VERB",
        "PUNCT": "PUNCTUATION", "INTJ": "INTERJECTION", "ADV": "ADVERB",
        "CCONJ": "CONJUNCTION", "DET": "DETERMINER",
    }

    JANOME_POS_MAP = {
        "名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJECTIVE",
        "助詞": "PARTICLE", "助動詞": "AUX_VERB", "記号": "PUNCTUATION",
        "感動詞": "INTERJECTION", "副詞": "ADVERB", "接続詞": "CONJUNCTION",
    }

    def split_sentences(self, text: str) -> List[str]:
        """Splits Japanese text into discrete sentences via GiNZA or punctuation delimiters."""
        if _HAS_GINZA and _GINZA_NLP:
            doc = _GINZA_NLP(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            refined = []
            for s in sents:
                sub_doc = _GINZA_NLP(s)
                split_idx = None
                for t in sub_doc:
                    if t.text == "、" and t.i > 0:
                        prev_text = "".join(tok.text for tok in sub_doc[:t.i])
                        greetings = ["こんにちは", "こんばんは", "おはよう", "ありがとう", "すみません"]
                        if any(g in prev_text for g in greetings):
                            split_idx = t.idx + len(t.text)
                            break
                if split_idx and split_idx < len(s):
                    refined.append(s[:split_idx - 1] + "。")
                    refined.append(s[split_idx:].strip())
                else:
                    refined.append(s)
            if refined:
                return refined

        # Fallback: split on Japanese & standard full-stop marks
        import re
        parts = re.split(r"(?<=[。！？!?\n])", text)
        return [p.strip() for p in parts if p.strip()]

    def tokenize(self, segment: SubtitleSegment) -> List[TokenModel]:
        """Tokenizes segment text into rich TokenModel instances."""
        if _HAS_GINZA and _GINZA_NLP:
            return self._tokenize_ginza(segment)
        if _HAS_JANOME and _JANOME_TOKENIZER:
            return self._tokenize_janome(segment)
        return self._tokenize_fallback(segment)

    def _tokenize_ginza(self, segment: SubtitleSegment) -> List[TokenModel]:
        doc = _GINZA_NLP(segment.text)
        raw_tokens: List[TokenModel] = []
        text = segment.text

        for t in doc:
            pos = "PARTICLE" if t.dep_ == "fixed" else self.GINZA_POS_MAP.get(t.pos_, "OTHER")
            reading = ginza.reading_form(t, True)
            lemma = t.lemma_ if t.lemma_ != "*" else t.text

            raw_tokens.append(TokenModel(
                segment_id=segment.segment_id,
                surface=t.text,
                pos=pos,
                lemma=lemma,
                reading=reading,
                romaji="",
                start_position=t.idx,
                end_position=t.idx + len(t.text)
            ))

        # Merge compound nouns (e.g. 若者 + 言葉 -> 若者言葉, ポッド + キャスト -> ポッドキャスト)
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

    def _tokenize_janome(self, segment: SubtitleSegment) -> List[TokenModel]:
        raw_tokens: List[TokenModel] = []
        pos = 0
        text = segment.text
        for item in _JANOME_TOKENIZER.tokenize(text):
            surface = item.surface
            main_pos = item.part_of_speech.split(",")[0]
            mapped_pos = self.JANOME_POS_MAP.get(main_pos, "OTHER")
            start = text.find(surface, pos)
            start = pos if start == -1 else start
            end = start + len(surface)
            pos = end

            raw_tokens.append(TokenModel(
                segment_id=segment.segment_id,
                surface=surface,
                pos=mapped_pos,
                lemma=item.base_form if item.base_form != "*" else surface,
                reading=item.reading if item.reading != "*" else surface,
                start_position=start,
                end_position=end
            ))

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
        return [
            TokenModel(
                segment_id=segment.segment_id,
                surface=ch,
                pos="PUNCTUATION" if ch in "、。！？!?" else "NOUN",
                lemma=ch,
                reading=ch,
                start_position=i,
                end_position=i + 1
            )
            for i, ch in enumerate(segment.text)
        ]
