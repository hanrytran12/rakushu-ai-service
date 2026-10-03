"""LLM Service: Contextual translation, linguistic explanation, and structured OOV learning."""
import os
import json
import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from src.models import TokenModel, DictionaryEntry, OovCandidate

logger = logging.getLogger("RakushuPipeline")


class LlmEnrichmentService:
    """Enriches transcription with contextual translation, retries, and OOV candidate extraction."""

    def __init__(self):
        self.local_llm_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434")
        self.local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "qwen3:4b")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "3600.0"))
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", "3"))
        self.last_sentence_translation_failed = False

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_attempts: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Invokes local LLM with bounded retries and returns parsed JSON or None."""
        base = self.local_llm_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        sys_content = system_prompt or "You are Rakushu AI Linguistic Engine. Return JSON only."
        base_options = {"temperature": 0.3, "repeat_penalty": 1.15, "top_p": 0.9}
        retry_options = [
            base_options,
            {"temperature": 0.2, "repeat_penalty": 1.20, "top_p": 0.85, "num_predict": 512},
            {"temperature": 0.1, "repeat_penalty": 1.25, "top_p": 0.80, "num_predict": 256},
        ]
        last_status = 0

        attempts = max(1, max_attempts if max_attempts is not None else self.max_retries)
        for attempt in range(1, attempts + 1):
            payload = {
                "model": self.local_llm_model,
                "messages": [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt}
                ],
                "think": False, "stream": False, "format": "json",
                "options": retry_options[min(attempt - 1, len(retry_options) - 1)]
            }
            try:
                import requests
                res = requests.post(f"{base}/api/chat", json=payload, timeout=self.timeout)
                last_status = res.status_code
                if res.status_code == 200:
                    data = self._extract_json(res.json().get("message", {}).get("content", ""))
                    if data:
                        return data
                detail = res.text[:120].replace("\n", " ")
                logger.warning(
                    f"[LLM Retry {attempt}/{attempts}] Status {res.status_code}: {detail}"
                )
            except Exception as exc:
                logger.warning(f"[LLM Retry {attempt}/{attempts}] Call failed: {exc}")

            if attempt < attempts:
                time.sleep(1.0)

        logger.error(
            f"[LLM FAILED] Failed all {attempts} attempts "
            f"(last_status={last_status})."
        )
        return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON object from response string."""
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group(0)) if m else {}

    def build_translation_chunks(
        self,
        sentences: List[str],
        max_chars: int = 900,
        max_sentences: int = 12,
    ) -> List[Tuple[int, int]]:
        """Plans bounded chunks using both character and sentence-count limits.

        Sentence count is the primary safety bound because the chunk response contains
        structured translation/token/OOV data for every sentence.
        """
        chunks: List[Tuple[int, int]] = []
        start = 0
        chars = 0
        for idx, sentence in enumerate(sentences):
            size = len(sentence)
            count = idx - start

            if idx > start and (chars + size > max_chars or count >= max_sentences):
                chunks.append((start, idx))
                start, chars = idx, 0

            chars += size

        if start < len(sentences):
            chunks.append((start, len(sentences)))
        return chunks

    def _translate_chunk_batch(
        self,
        items: List[Dict[str, Any]],
        max_attempts: int = 2,
    ) -> Dict[int, Dict[str, Any]]:
        """Runs one bounded batch request and validates sentence IDs."""
        if not items:
            return {}

        prompt = f"""Bạn là engine dịch Nhật-Việt cho Rakushu.
Dịch và làm giàu TẤT CẢ các câu dưới đây trong một lần gọi.
Giữ nguyên số câu và tuyệt đối không gộp, bỏ hoặc đổi số câu.
Ngữ cảnh trong cùng chunk được dùng để hiểu đại từ, chủ thể và liên kết diễn ngôn.

INPUT:
{json.dumps(items, ensure_ascii=False)}

Trả về JSON đúng cấu trúc:
{{
  "sentences": [
    {{
      "sentence_number": 1,
      "translation_vi": "...",
      "token_meanings": {{"từ": "nghĩa ngắn gọn"}},
      "oov_learning": [{{"term": "...", "pos": "NOUN", "suggested_meaning": "...", "confidence_score": 0.95}}]
    }}
  ]
}}
Chỉ trả JSON. Không thêm markdown hay giải thích.
"""
        res = self._call_llm(prompt, max_attempts=max_attempts)
        if not res or not isinstance(res.get("sentences"), list):
            return {}

        expected = {int(item["sentence_number"]) for item in items}
        parsed: Dict[int, Dict[str, Any]] = {}
        for item in res["sentences"]:
            if not isinstance(item, dict):
                continue
            try:
                number = int(item.get("sentence_number"))
            except (TypeError, ValueError):
                continue
            translation = str(item.get("translation_vi", "")).strip()
            if number in expected and number not in parsed and translation:
                parsed[number] = item
        return parsed

    def _translate_chunk_adaptive(
        self,
        items: List[Dict[str, Any]],
        depth: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        """Recovers failed chunk translations by retrying missing items, then splitting."""
        if not items:
            return {}

        parsed = self._translate_chunk_batch(items, max_attempts=2)
        expected = {int(item["sentence_number"]) for item in items}
        missing = sorted(expected - set(parsed))

        if not missing:
            return parsed

        logger.warning(
            f"[Chunk Validation] Batch returned {len(parsed)}/{len(items)}; "
            f"missing={missing}"
        )

        # Retry only the missing subset once, instead of retrying the original large batch.
        missing_items = [
            item for item in items
            if int(item["sentence_number"]) in missing
        ]
        if missing_items and len(missing_items) < len(items):
            logger.info(
                f"[Chunk Recovery] Retrying only {len(missing_items)} missing sentences "
                f"from batch size {len(items)}."
            )
            recovered = self._translate_chunk_batch(missing_items, max_attempts=2)
            parsed.update(recovered)
            missing = sorted(expected - set(parsed))
            if not missing:
                return parsed

        # If the batch remains incomplete or completely failed, split it.
        # This prevents repeated expensive retries of the same oversized prompt.
        if len(items) > 1:
            midpoint = len(items) // 2
            left_items = items[:midpoint]
            right_items = items[midpoint:]
            logger.warning(
                f"[Chunk Adaptive Split] depth={depth} splitting "
                f"{len(items)} sentences into {len(left_items)} + {len(right_items)}."
            )
            left_results = self._translate_chunk_adaptive(left_items, depth + 1)
            right_results = self._translate_chunk_adaptive(right_items, depth + 1)
            parsed.update(left_results)
            parsed.update(right_results)

        final_missing = sorted(expected - set(parsed))
        if final_missing:
            logger.warning(
                f"[Chunk Recovery] Still missing after adaptive split: {final_missing}"
            )
        return parsed

    def translate_chunk(
        self,
        sentences: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Translates a bounded chunk with adaptive recovery and splitting."""
        if not sentences:
            return {}

        items = [
            {
                "sentence_number": int(item["sentence_number"]),
                "text": item["text"],
            }
            for item in sentences
        ]
        return self._translate_chunk_adaptive(items)

    def build_sentence_context(
        self,
        sentences: List[str],
        target_index: int,
        before: int = 3,
        after: int = 2,
    ) -> str:
        """Builds bounded local context around one target sentence."""
        start = max(0, target_index - before)
        end = min(len(sentences), target_index + after + 1)
        lines = []
        for idx in range(start, end):
            label = "TARGET" if idx == target_index else "CONTEXT"
            lines.append(f"[{label} {idx + 1}] {sentences[idx]}")
        return "\n".join(lines)

    def translate_sentence_with_context(
        self,
        sentence_text: str,
        tokens: List[TokenModel] = None,
        matched_knowledge: List[DictionaryEntry] = None,
        oov_tokens: List[TokenModel] = None,
        context_text: str = "",
        expanded_context_text: str = "",
    ) -> Tuple[str, Dict[str, str], List[OovCandidate]]:
        """Stage 2: Translates one sentence using bounded local context."""
        self.last_sentence_translation_failed = False
        tokens = tokens or []
        matched_knowledge = matched_knowledge or []
        oov_tokens = oov_tokens or []
        known_summary = [f"- {k.term} ({k.reading}): {k.meaning}" for k in matched_knowledge]
        content_tokens = [t.surface for t in tokens if t.pos not in ("PUNCTUATION", "PARTICLE", "AUX_VERB")]
        oov_terms_list = list({tok.surface: tok for tok in oov_tokens}.values())

        prompt = f"""Ngữ cảnh tham khảo (không cần dịch lại):
{context_text or "[Không có ngữ cảnh bổ sung]"}

 Câu cần dịch: "{sentence_text}"
 Từ điển: {known_summary}
 Từ chính: {content_tokens}
 Từ OOV: {[t.surface for t in oov_terms_list]}

Yêu cầu dịch CHỈ câu cần dịch sang tiếng Việt và trả về JSON:
- "translation_vi": bản dịch tiếng Việt của câu
- "token_meanings": {{"từ_tiếng_nhật": "nghĩa ngắn gọn"}}
- "oov_learning": [{{"term": "từ OOV", "pos": "NOUN", "suggested_meaning": "định nghĩa", "confidence_score": 0.95}}]
- "context_sufficient": true nếu ngữ cảnh hiện tại đủ để xác định các đại từ/tham chiếu; false nếu cần thêm ngữ cảnh
- "context_issue": mô tả ngắn điều còn thiếu, nếu có
"""

        logger.info(f"[Sentence Fallback] Target: {sentence_text}")
        logger.info(f"[Sentence Fallback] Initial context: 3 before + target + 2 after")
        logger.info(f"[Sentence Fallback] Initial context text:\n{context_text or '[NO CONTEXT]'}")
        res = self._call_llm(prompt)
        if res and res.get("context_sufficient") is False and expanded_context_text:
            logger.info(f"[Sentence Fallback] Context insufficient for target: {sentence_text}")
            logger.info(f"[Sentence Fallback] Expanding context: 10 before + target + 3 after")
            logger.info(f"[Sentence Fallback] Expanded context text:\n{expanded_context_text}")
            prompt = prompt.replace(context_text or "[Không có ngữ cảnh bổ sung]", expanded_context_text)
            prompt = prompt.replace("Ngữ cảnh hiện tại chưa đủ.", "Đây là lần thử mở rộng ngữ cảnh; ưu tiên ngữ cảnh gần target.")
            res = self._call_llm(prompt)
        if res and "translation_vi" in res:
            vi_trans = str(res.get("translation_vi", "")).strip()
            if vi_trans and vi_trans.lower() not in ("bản dịch tiếng việt tự nhiên", "bản dịch tiếng việt"):
                raw_meanings = res.get("token_meanings", {})
                t_meanings = self._align_token_meanings(raw_meanings, tokens, matched_knowledge)
                cands = self._parse_oov_candidates(res.get("oov_learning", []), oov_terms_list, sentence_text)
                logger.info(f"[Sentence Fallback] SUCCESS target='{sentence_text}' | translation='{vi_trans}'")
                return vi_trans, t_meanings, cands

        self.last_sentence_translation_failed = True
        logger.warning(f"[Sentence Fallback] FINAL FAILURE target='{sentence_text}' | returning empty translation")
        logger.warning(f"[Sentence Fallback] FINAL FAILURE target='{sentence_text}' | returning empty translation")
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
            reading = item.get("reading") or tok.reading or tok.surface
            score = float(item.get("confidence_score") or item.get("score") or 0.85)
            cands.append(OovCandidate(
                token_id=tok.token_id, term=tok.surface, tentative_reading=reading,
                suggested_meaning=meaning, suggested_pos=pos, context_snippet=text,
                confidence_score=score, status="PENDING_CURATOR_REVIEW"
            ))
        return cands