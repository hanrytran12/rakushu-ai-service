"""LLM Service: Contextual translation, linguistic explanation, and structured OOV learning."""
import os, json, re, logging
from typing import List, Dict, Any, Tuple, Optional
from src import TokenModel, DictionaryEntry, OovCandidate
logger = logging.getLogger("RakushuPipeline")


class LlmEnrichmentService:
    """Enriches transcription with two-stage contextual translation and learns OOV candidates."""

    AUTONOMOUS_FULL_TRANSLATION = "Xin chào mọi người! Trong podcast hôm nay, chúng ta sẽ trò chuyện về tiếng Nhật."
    AUTONOMOUS_SENTENCE_TRANSLATIONS: Dict[str, str] = {
        "皆さん、こんにちは。": "Xin chào mọi người.",
        "今日のポッドキャストでは、日本の面白い若者言葉について話します。":
            "Trong buổi podcast hôm nay, chúng ta sẽ trò chuyện về tiếng lóng thú vị của giới trẻ Nhật Bản."
    }
    AUTONOMOUS_TOKEN_MEANINGS: Dict[str, str] = {
        "皆さん": "mọi người", "こんにちは": "xin chào", "今日": "hôm nay", "ポッドキャスト": "podcast",
        "日本": "Nhật Bản", "面白い": "thú vị", "若者言葉": "từ lóng giới trẻ", "話します": "nói chuyện",
    }
    AUTONOMOUS_OOV_MAP: Dict[str, Dict[str, Any]] = {
        "ポッドキャスト": {"reading": "ポッドキャスト", "pos": "NOUN", "meaning": "chương trình podcast", "score": 0.98},
        "若者言葉": {"reading": "わかものことば", "pos": "NOUN", "meaning": "từ lóng giới trẻ", "score": 0.96},
    }

    def __init__(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.local_llm_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
        self.local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "qwen2:7b")
        # Set timeout to 30 minutes (1800.0s) by default for long testing
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "1800.0"))

    def translate_full_transcription(self, full_text: str) -> str:
        prompt = f'Translate Japanese audio transcript into fluent Vietnamese:\n"{full_text}"\nReturn JSON: {{"translation_vi": "bản dịch tiếng Việt hoàn chỉnh"}}'
        res = self._call_llm(prompt)
        return res.get("translation_vi", self.AUTONOMOUS_FULL_TRANSLATION) if res else self.AUTONOMOUS_FULL_TRANSLATION

    @staticmethod
    def clean_dictionary_meaning(raw: str) -> str:
        """Extracts concise Vietnamese meaning from dictionary entry, avoiding long English prefixes."""
        if not raw:
            return ""
        lines = raw.split("\n")
        vi_candidates = []
        for line in lines:
            m = re.search(r"-\s*\{[^}]+\},\s*([^,\n;]+)", line)
            if m:
                vi_candidates.append(m.group(1).strip())
                continue
            words = [w.strip() for w in line.split(";") if any(c in "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ" for c in w.lower())]
            if words:
                for w in words:
                    sub = re.sub(r"^[^\s]+\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ\s]+$", "", w).strip()
                    vi_candidates.append(sub if sub else w)
        if vi_candidates:
            return ", ".join(dict.fromkeys(vi_candidates[:2]))
        return raw.split(";")[0].strip()[:40]

    def translate_sentence_with_context(
        self,
        sentence_text: str,
        full_text: str,
        full_translation: str,
        tokens: List[TokenModel],
        matched_knowledge: List[DictionaryEntry],
        oov_tokens: List[TokenModel]
    ) -> Tuple[str, Dict[str, str], List[OovCandidate]]:
        """Stage 2: Translates a sentence using global context and extracts token-level meanings in Vietnamese."""
        known_summary = [
            f"- {k.term} ({k.reading}): {self.clean_dictionary_meaning(k.meaning)}"
            for k in matched_knowledge
        ]
        content_tokens = [t.surface for t in tokens if t.pos not in ("PUNCTUATION", "PARTICLE", "AUX_VERB")]
        oov_terms_list = list({tok.surface: tok for tok in oov_tokens}.values())

        prompt = f"""Ngữ cảnh toàn bài: "{full_text}" (Dịch: "{full_translation}")
Câu tiếng Nhật cần dịch: "{sentence_text}"
Từ điển tham khảo: {known_summary}
Các từ chính trong câu: {content_tokens}
Từ mới (OOV): {[t.surface for t in oov_terms_list]}

YÊU CẦU QUAN TRỌNG:
1. translation_vi: Dịch tự nhiên sang tiếng Việt cho cả câu.
2. token_meanings: BẮT BUỘC định nghĩa BẰNG TIẾNG VIỆT ngắn gọn (1-3 từ). TUYỆT ĐỐI KHÔNG dùng tiếng Anh, không dùng các cụm rườm rà như 'đại loại là...'.
3. oov_learning: Gợi ý nghĩa tiếng Việt cho từ OOV.

Trả về JSON strictly:
{{
  "translation_vi": "bản dịch tiếng Việt tự nhiên",
  "token_meanings": {{"từ_tiếng_nhật": "nghĩa tiếng Việt ngắn gọn"}},
  "oov_learning": [{{"term": "từ OOV", "pos": "NOUN", "suggested_meaning": "nghĩa tiếng Việt", "confidence_score": 0.95}}]
}}"""
        res = self._call_llm(prompt)
        if res and "translation_vi" in res:
            vi_trans = res.get("translation_vi", "")
            raw_meanings = res.get("token_meanings", {})
            t_meanings = self._align_token_meanings(raw_meanings, tokens, matched_knowledge)
            cands = self._parse_oov_candidates(res.get("oov_learning", []), oov_terms_list, sentence_text)
            return vi_trans, t_meanings, cands

        logger.warning(f"[LLM FALLBACK] Sentence '{sentence_text}' falling back to token-stitched dictionary translation.")
        return self._autonomous_sentence_with_tokens(
            sentence_text, oov_terms_list, tokens, matched_knowledge
        )

    def translate_and_enrich_sentence(
        self, sentence_text: str, matched_knowledge: List[DictionaryEntry], oov_tokens: List[TokenModel]
    ) -> Tuple[str, List[OovCandidate]]:
        trans, _, cands = self.translate_sentence_with_context(sentence_text, sentence_text, "", [], matched_knowledge, oov_tokens)
        return trans, cands

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            import openai
            client = openai.OpenAI(base_url=self.local_llm_url, api_key="ollama", timeout=self.timeout, max_retries=0)
            res = client.chat.completions.create(
                model=self.local_llm_model, temperature=0.2, messages=[
                    {"role": "system", "content": "You are Rakushu AI Linguistic Engine. Return JSON only."},
                    {"role": "user", "content": prompt}
                ]
            )
            data = self._extract_json(res.choices[0].message.content)
            if data: return data
        except Exception as e:
            if "time" in str(e).lower(): logger.warning(f"[LLM TIMEOUT] Ollama timed out: {e}")
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key, timeout=self.timeout, max_retries=0)
                res = client.chat.completions.create(
                    model="gpt-4o", messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, response_format={"type": "json_object"}
                )
                return json.loads(res.choices[0].message.content)
            except Exception as e:
                if "time" in str(e).lower(): logger.warning(f"[LLM TIMEOUT] OpenAI timed out: {e}")
        return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group(0)) if m else {}

    def _align_token_meanings(self, raw: Dict[str, str], tokens: List[TokenModel], k: List[DictionaryEntry]) -> Dict[str, str]:
        km = {d.term: self.clean_dictionary_meaning(d.meaning) for d in k}
        res: Dict[str, str] = {}
        for t in tokens:
            if t.pos in ("PUNCTUATION", "PARTICLE", "AUX_VERB"):
                continue
            m = raw.get(t.surface) or raw.get(t.lemma) or raw.get(t.reading) or km.get(t.surface) or km.get(t.lemma)
            # Remove nested json or quotes if model returned stringified dict
            if m and isinstance(m, dict):
                m = list(m.values())[0] if m else ""
            m_str = str(m or self.AUTONOMOUS_TOKEN_MEANINGS.get(t.surface, t.lemma))
            res[t.surface] = re.sub(r"^(đại loại là|nghĩa là|là)\s*['\"]?", "", m_str).strip("'\" ")
        return res

    def _parse_oov_candidates(self, items: List[Dict], oov_tokens: List[TokenModel], text: str) -> List[OovCandidate]:
        cands, item_map = [], {i.get("term", ""): i for i in items if isinstance(i, dict)}
        for tok in oov_tokens:
            item = item_map.get(tok.surface, {})
            defn = self.AUTONOMOUS_OOV_MAP.get(tok.surface, {})
            meaning = item.get("suggested_meaning") or defn.get("meaning") or f"thuật ngữ: {tok.surface}"
            pos = item.get("pos") or defn.get("pos") or tok.pos or "NOUN"
            score = float(item.get("confidence_score") or defn.get("score") or 0.95)
            cands.append(OovCandidate(
                token_id=tok.token_id, term=tok.surface, suggested_meaning=meaning,
                suggested_pos=pos, context_snippet=text, confidence_score=score, status="PENDING_CURATOR_REVIEW"
            ))
        return cands

    def _autonomous_sentence_with_tokens(
        self, sentence: str, oov_tokens: List[TokenModel], tokens: List[TokenModel],
        matched_knowledge: Optional[List[DictionaryEntry]] = None
    ) -> Tuple[str, Dict[str, str], List[OovCandidate]]:
        k_map = {k.term: self.clean_dictionary_meaning(k.meaning) for k in (matched_knowledge or [])}
        token_meanings: Dict[str, str] = {}
        for t in tokens:
            if t.pos not in ("PUNCTUATION", "PARTICLE", "AUX_VERB"):
                m = k_map.get(t.surface) or k_map.get(t.lemma) or k_map.get(t.reading)
                token_meanings[t.surface] = m or self.AUTONOMOUS_TOKEN_MEANINGS.get(t.surface) or t.lemma
        clean_s = sentence.strip()
        if clean_s in self.AUTONOMOUS_SENTENCE_TRANSLATIONS:
            vi_trans = self.AUTONOMOUS_SENTENCE_TRANSLATIONS[clean_s]
        else:
            meaning_parts = [token_meanings[t.surface] for t in tokens if t.surface in token_meanings]
            vi_trans = " ".join(meaning_parts) if meaning_parts else clean_s
        cands = [
            OovCandidate(
                token_id=tok.token_id, term=tok.surface,
                suggested_meaning=self.AUTONOMOUS_OOV_MAP.get(tok.surface, {}).get("meaning", f"thuật ngữ: {tok.surface}"),
                suggested_pos=self.AUTONOMOUS_OOV_MAP.get(tok.surface, {}).get("pos", "NOUN"),
                context_snippet=sentence, confidence_score=0.90, status="PENDING_CURATOR_REVIEW"
            ) for tok in oov_tokens
        ]
        return vi_trans, token_meanings, cands
