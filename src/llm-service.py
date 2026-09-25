"""LLM Service: Contextual translation, linguistic explanation, and structured OOV learning."""
import os
import json
import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from src import TokenModel, DictionaryEntry, OovCandidate

logger = logging.getLogger("RakushuPipeline")


class LlmEnrichmentService:
    """Enriches transcription with contextual translation, retries, and OOV candidate extraction."""

    def __init__(self):
        self.local_llm_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434")
        self.local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "qwen3:4b")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "3600.0"))
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", "3"))

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Invokes local LLM with 3 retries upon failure. Returns parsed JSON or None."""
        base = self.local_llm_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        sys_content = system_prompt or "You are Rakushu AI Linguistic Engine. Return JSON only."
        payload = {
            "model": self.local_llm_model,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt}
            ],
            "think": False, "stream": False, "format": "json", "options": {"temperature": 0.2}
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                import requests
                res = requests.post(f"{base}/api/chat", json=payload, timeout=self.timeout)
                if res.status_code == 200:
                    data = self._extract_json(res.json().get("message", {}).get("content", ""))
                    if data:
                        return data
                logger.warning(f"[LLM Retry {attempt}/{self.max_retries}] Status {res.status_code}: {res.text[:80]}")
            except Exception as exc:
                logger.warning(f"[LLM Retry {attempt}/{self.max_retries}] Call failed: {exc}")

            if attempt < self.max_retries:
                time.sleep(1.0)

        logger.error(f"[LLM FAILED] Failed all {self.max_retries} attempts to call LLM.")
        return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON object from response string."""
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group(0)) if m else {}

    def translate_full_transcription(self, full_text: str) -> str:
        """Stage 1: Translates complete transcription to establish global discourse context."""
        sentences = re.split(r"(?<=[。！？\n])", full_text)
        chunks, cur, translations = [], "", []
        for s in sentences:
            if not s:
                continue
            if len(cur) + len(s) > 500 and cur:
                chunks.append(cur.strip())
                cur = s
            else:
                cur += s
        if cur.strip():
            chunks.append(cur.strip())

        sys_p = "You are a professional Japanese-Vietnamese translator. Always output a valid JSON object with key 'translation_vi'."
        for c in chunks:
            prompt = f'Dịch đoạn văn tiếng Nhật sau sang tiếng Việt tự nhiên, đầy đủ:\n"{c}"\nTrả về đúng định dạng JSON: {{"translation_vi": "bản dịch tiếng Việt"}}'
            res = self._call_llm(prompt, system_prompt=sys_p)
            if res and isinstance(res, dict) and "translation_vi" in res:
                translations.append(str(res["translation_vi"]).strip())
            else:
                logger.warning(f"[LLM Fallback] Failed chunk global translation after {self.max_retries} retries.")
        return " ".join(translations) if translations else ""

    def translate_sentence_with_context(
        self,
        sentence_text: str,
        full_text: str,
        full_translation: str,
        tokens: List[TokenModel],
        matched_knowledge: List[DictionaryEntry],
        oov_tokens: List[TokenModel]
    ) -> Tuple[str, Dict[str, str], List[OovCandidate]]:
        """Stage 2: Translates a sentence using global context and extracts token-level meanings."""
        known_summary = [f"- {k.term} ({k.reading}): {k.meaning}" for k in matched_knowledge]
        content_tokens = [t.surface for t in tokens if t.pos not in ("PUNCTUATION", "PARTICLE", "AUX_VERB")]
        oov_terms_list = list({tok.surface: tok for tok in oov_tokens}.values())
        ctx_text = (full_text[:300] + "...") if len(full_text) > 300 else full_text
        ctx_trans = (full_translation[:300] + "...") if len(full_translation) > 300 else full_translation

        prompt = f"""Ngữ cảnh: "{ctx_text}" (Nghĩa: "{ctx_trans}")
Câu cần dịch: "{sentence_text}"
Từ điển: {known_summary}
Từ chính: {content_tokens}
Từ OOV: {[t.surface for t in oov_terms_list]}

Yêu cầu dịch câu tiếng Nhật trên sang tiếng Việt và trả về JSON:
- "translation_vi": bản dịch tiếng Việt của câu
- "token_meanings": {{"từ_tiếng_nhật": "nghĩa ngắn gọn"}}
- "oov_learning": [{{"term": "từ OOV", "pos": "NOUN", "suggested_meaning": "định nghĩa", "confidence_score": 0.95}}]"""

        res = self._call_llm(prompt)
        if res and "translation_vi" in res:
            vi_trans = str(res.get("translation_vi", "")).strip()
            if vi_trans and vi_trans.lower() not in ("bản dịch tiếng việt tự nhiên", "bản dịch tiếng việt"):
                raw_meanings = res.get("token_meanings", {})
                t_meanings = self._align_token_meanings(raw_meanings, tokens, matched_knowledge)
                cands = self._parse_oov_candidates(res.get("oov_learning", []), oov_terms_list, sentence_text)
                return vi_trans, t_meanings, cands

        # Tier 4 Fallback directly when all 3 retries fail
        logger.warning(f"[LLM Tier 4 Fallback] Sentence '{sentence_text}' could not be translated after {self.max_retries} retries.")
        km = {d.term: d.meaning.split("(")[0].strip() for d in matched_knowledge}
        default_meanings = {
            t.surface: km.get(t.surface, km.get(t.lemma, ""))
            for t in tokens if t.pos not in ("PUNCTUATION", "PARTICLE", "AUX_VERB")
        }
        return "", default_meanings, []

    def _align_token_meanings(self, raw: Dict[str, str], tokens: List[TokenModel], k: List[DictionaryEntry]) -> Dict[str, str]:
        """Aligns extracted token meanings with dictionary knowledge."""
        km = {d.term: d.meaning.split("(")[0].strip() for d in k}
        res: Dict[str, str] = {}
        for t in tokens:
            if t.pos in ("PUNCTUATION", "PARTICLE", "AUX_VERB"):
                continue
            m = raw.get(t.surface) or raw.get(t.lemma) or raw.get(t.reading) or km.get(t.surface) or km.get(t.lemma)
            res[t.surface] = m or t.lemma
        return res

    def _parse_oov_candidates(self, items: List[Dict], oov_tokens: List[TokenModel], text: str) -> List[OovCandidate]:
        """Parses learned OOV candidates from LLM output."""
        cands = []
        item_map = {i.get("term", ""): i for i in items if isinstance(i, dict)}
        for tok in oov_tokens:
            item = item_map.get(tok.surface, {})
            meaning = item.get("suggested_meaning") or f"thuật ngữ: {tok.surface}"
            pos = item.get("pos") or tok.pos or "NOUN"
            score = float(item.get("confidence_score") or 0.85)
            cands.append(OovCandidate(
                token_id=tok.token_id, term=tok.surface, suggested_meaning=meaning,
                suggested_pos=pos, context_snippet=text, confidence_score=score,
                status="PENDING_CURATOR_REVIEW"
            ))
        return cands
