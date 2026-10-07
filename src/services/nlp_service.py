"""NLP Service: Japanese morphological tokenizer and sentence segmenter using GiNZA."""
import json
from typing import Any, Dict, List, Optional
from src.models import TokenModel, SubtitleSegment

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

    def __init__(self):
        self._ginza_doc_cache: Dict[str, Any] = {}

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

    def analyze_json(self, segment: SubtitleSegment) -> Dict[str, Any]:
        """Return GiNZA JSON-style token output plus Bunsetsu metadata."""
        if not (_HAS_GINZA and _GINZA_NLP):
            tokens = self.tokenize(segment)
            return {
                "format": "rakushu-ginza-json-fallback",
                "text": segment.text,
                "tokens": [
                    {
                        "id": i,
                        "orth": token.surface,
                        "tag": token.pos,
                        "pos": token.pos,
                        "lemma": token.lemma,
                        "norm": token.lemma,
                        "head": 0,
                        "head_absolute": 0,
                        "dep": "ROOT" if i == 1 else "dep",
                        "ner": "O",
                        "start": token.start_position,
                        "end": token.end_position,
                        "reading": token.reading,
                        "inf": None,
                        "bunsetu_bi_label": "B" if i == 1 else "I",
                        "bunsetu_position_type": "ROOT" if i == 1 else "CONT",
                        "clause_head": 1,
                        "whitespace": "",
                    }
                    for i, token in enumerate(tokens, start=1)
                ],
            }

        doc = self._get_ginza_doc(segment.text)

        # GiNZA's own JSON formatter is the source of truth for the API.
        # This is the same formatter used by `ginza -f json`.
        from ginza.analyzer import format_json
        ginza_json = json.loads(format_json(next(doc.sents)))
        tokens: List[Dict[str, Any]] = []
        for token in doc:
            head_offset = token.head.i - token.i
            reading = ginza.reading_form(token, True)
            inf = ginza.inflection(token)
            tokens.append({
                "id": token.i + 1,
                "orth": token.orth_,
                "tag": token.tag_,
                "pos": token.pos_,
                "lemma": token.lemma_,
                "norm": token.norm_,
                "head": head_offset,
                "head_absolute": token.head.i + 1,
                "dep": token.dep_,
                "ner": token.ent_iob_ if not token.ent_type_ else f"{token.ent_iob_}-{token.ent_type_}",
                "start": token.idx,
                "end": token.idx + len(token.text),
                "whitespace": token.whitespace_,
                "reading": reading or "",
                "inf": inf or "",
                "bunsetu_bi_label": ginza.bunsetu_bi_label(token),
                "bunsetu_position_type": ginza.bunsetu_position_type(token),
                "clause_head": ginza.clause_head_i(token) + 1,
                "ne": ginza.ent_label_ontonotes(token) or "O",
                "ene": ginza.ent_label_ene(token) or "O",
                "morph": token.morph.to_dict(),
            })

        return {
            "format": "ginza-json",
            "text": segment.text,
            "ginza_json": ginza_json,
            # `ginza_full` keeps the native formatter output and adds every
            # GiNZA metadata field used by its CoNLL-U/accessor APIs.
            "ginza_full": {
                "json": ginza_json,
                "tokens": tokens,
            },
            # Internal normalized token data is retained for Bunsetsu/pipeline
            # consumers.
            "tokens": tokens,
        }

    def _get_ginza_doc(self, text: str):
        if text not in self._ginza_doc_cache:
            self._ginza_doc_cache[text] = _GINZA_NLP(text)
        return self._ginza_doc_cache[text]

    def tokenize(self, segment: SubtitleSegment) -> List[TokenModel]:
        """Tokenizes segment text into rich TokenModel instances."""
        if _HAS_GINZA and _GINZA_NLP:
            return self._tokenize_ginza(segment)
        if _HAS_JANOME and _JANOME_TOKENIZER:
            return self._tokenize_janome(segment)
        return self._tokenize_fallback(segment)

    def _tokenize_ginza(self, segment: SubtitleSegment) -> List[TokenModel]:
        doc = self._get_ginza_doc(segment.text)
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
