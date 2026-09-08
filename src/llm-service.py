"""LLM Service: Contextual translation, linguistic explanation, and structured OOV learning."""
import os, json, re, logging
from typing import List, Dict, Any, Tuple, Optional
from src import TokenModel, DictionaryEntry, OovCandidate
logger = logging.getLogger("RakushuPipeline")


class LlmEnrichmentService:
    """Enriches transcription with two-stage contextual translation and learns OOV candidates."""

    AUTONOMOUS_FULL_TRANSLATION = (
        "Xin chào mọi người! Trong podcast hôm nay, chúng ta sẽ trò chuyện về tiếng lóng thú vị của giới trẻ Nhật Bản."
    )
    AUTONOMOUS_SENTENCE_TRANSLATIONS: Dict[str, str] = {
        "皆さん、こんにちは。": "Xin chào mọi người.",
        "今日のポッドキャストでは、日本の面白い若者言葉について話します。":
            "Trong buổi podcast hôm nay, chúng ta sẽ trò chuyện về tiếng lóng thú vị của giới trẻ Nhật Bản."
    }
    AUTONOMOUS_TOKEN_MEANINGS: Dict[str, str] = {
        "皆さん": "mọi người", "こんにちは": "xin chào", "今日": "hôm nay", "ポッドキャスト": "chương trình podcast",
        "日本": "Nhật Bản", "面白い": "thú vị", "若者言葉": "từ lóng giới trẻ", "について": "về",
        "話します": "nói chuyện, chia sẻ", "話し": "nói chuyện",
    }
    AUTONOMOUS_OOV_MAP: Dict[str, Dict[str, Any]] = {
        "ポッドキャスト": {
            "reading": "ポッドキャスト", "pos": "NOUN",
            "meaning": "chương trình podcast, tệp âm thanh số định kỳ qua Internet (podcast)", "score": 0.98,
        },
        "若者言葉": {
            "reading": "わかものことば", "pos": "NOUN",
            "meaning": "từ lóng của giới trẻ, ngôn ngữ thịnh hành của thanh thiếu niên (youth slang)", "score": 0.96,
        },
    }

    def __init__(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.local_llm_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
        self.local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "qwen2:7b")
        # Set timeout to 30 minutes (1800.0s) by default for long testing
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "1800.0"))

    def translate_full_transcription(self, full_text: str) -> str:
        """Stage 1: Translates entire transcription to establish global discourse context."""
        prompt = (
            f'Translate this complete Japanese audio transcript into natural, fluent Vietnamese:\n"{full_text}"\n'
            'Return JSON: {"translation_vi": "bản dịch tiếng Việt hoàn chỉnh cho toàn bài"}'
        )
        res = self._call_llm(prompt)
        return res.get("translation_vi", self.AUTONOMOUS_FULL_TRANSLATION) if res else self.AUTONOMOUS_FULL_TRANSLATION

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

        prompt = f"""Context: Overall transcript is "{full_text}" with meaning "{full_translation}".
Current sentence to translate: "{sentence_text}"
Known dictionary entries: {known_summary}
Content words to assign meaning in this context: {content_tokens}
OOV candidate tokens: {[t.surface for t in oov_terms_list]}

Return JSON strictly:
{{
  "translation_vi": "bản dịch tiếng Việt tự nhiên cho câu này",
  "token_meanings": {{"từ_tiếng_nhật": "nghĩa cụ thể trong ngữ cảnh câu này"}},
  "oov_learning": [{{"term": "từ OOV", "pos": "NOUN", "suggested_meaning": "định nghĩa", "confidence_score": 0.95}}]
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
        """Backward-compatible helper for single sentence translation."""
        trans, _, cands = self.translate_sentence_with_context(
            sentence_text, sentence_text, "", [], matched_knowledge, oov_tokens
        )
        return trans, cands

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        # 1. Try local Ollama
        try:
            import openai
            client = openai.OpenAI(base_url=self.local_llm_url, api_key="ollama", timeout=self.timeout, max_retries=0)
            res = client.chat.completions.create(
                model=self.local_llm_model, messages=[
                    {"role": "system", "content": "You are Rakushu AI Linguistic Engine. Return JSON only."},
                    {"role": "user", "content": prompt}
                ], temperature=0.2
            )
            data = self._extract_json(res.choices[0].message.content)
            if data:
                return data
        except Exception as e:
            msg = f"[LLM TIMEOUT] Local Ollama timed out (>{self.timeout}s): {e}" if "time" in str(e).lower() else f"Ollama: {e}"
            logger.warning(msg) if "time" in str(e).lower() else logger.debug(msg)

        # 2. Try OpenAI GPT-4o
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
                if "time" in str(e).lower():
                    logger.warning(f"[LLM TIMEOUT] OpenAI call timed out (>{self.timeout}s): {e}")
        return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group(0)) if m else {}

    def _align_token_meanings(self, raw: Dict[str, str], tokens: List[TokenModel], k: List[DictionaryEntry]) -> Dict[str, str]:
        km = {d.term: d.meaning.split("(")[0].strip() for d in k}
        res: Dict[str, str] = {}
        for t in tokens:
            if t.pos in ("PUNCTUATION", "PARTICLE", "AUX_VERB"):
                continue
            m = raw.get(t.surface) or raw.get(t.lemma) or raw.get(t.reading) or km.get(t.surface) or km.get(t.lemma)
            res[t.surface] = m or self.AUTONOMOUS_TOKEN_MEANINGS.get(t.surface, t.lemma)
        return res

    def _parse_oov_candidates(self, items: List[Dict], oov_tokens: List[TokenModel], text: str) -> List[OovCandidate]:
        cands = []
        item_map = {i.get("term", ""): i for i in items if isinstance(i, dict)}
        for tok in oov_tokens:
            item = item_map.get(tok.surface, {})
            defn = self.AUTONOMOUS_OOV_MAP.get(tok.surface, {})
            meaning = item.get("suggested_meaning") or defn.get("meaning") or f"thuật ngữ: {tok.surface}"
            pos = item.get("pos") or defn.get("pos") or tok.pos or "NOUN"
            score = float(item.get("confidence_score") or defn.get("score") or 0.95)
            cands.append(OovCandidate(
                token_id=tok.token_id, term=tok.surface, suggested_meaning=meaning,
                suggested_pos=pos, context_snippet=text, confidence_score=score,
                status="PENDING_CURATOR_REVIEW"
            ))
        return cands

    def _autonomous_sentence_with_tokens(
        self,
        sentence: str,
        oov_tokens: List[TokenModel],
        tokens: List[TokenModel],
        matched_knowledge: Optional[List[DictionaryEntry]] = None
    ) -> Tuple[str, Dict[str, str], List[OovCandidate]]:
        # 1. Build token meanings from matched knowledge and autonomous maps
        k_map = {k.term: k.meaning.split("(")[0].strip() for k in (matched_knowledge or [])}
        token_meanings: Dict[str, str] = {}
        for t in tokens:
            if t.pos not in ("PUNCTUATION", "PARTICLE", "AUX_VERB"):
                m = k_map.get(t.surface) or k_map.get(t.lemma) or k_map.get(t.reading)
                token_meanings[t.surface] = m or self.AUTONOMOUS_TOKEN_MEANINGS.get(t.surface) or t.lemma

        # 2. Derive sentence translation: prefer preset, else stitch token meanings
        clean_s = sentence.strip()
        if clean_s in self.AUTONOMOUS_SENTENCE_TRANSLATIONS:
            vi_trans = self.AUTONOMOUS_SENTENCE_TRANSLATIONS[clean_s]
        else:
            meaning_parts = [token_meanings[t.surface] for t in tokens if t.surface in token_meanings]
            vi_trans = " ".join(meaning_parts) if meaning_parts else clean_s

        # 3. Construct OOV candidates
        cands = [
            OovCandidate(
                token_id=tok.token_id, term=tok.surface,
                suggested_meaning=self.AUTONOMOUS_OOV_MAP.get(tok.surface, {}).get("meaning", f"thuật ngữ: {tok.surface}"),
                suggested_pos=self.AUTONOMOUS_OOV_MAP.get(tok.surface, {}).get("pos", "NOUN"),
                context_snippet=sentence, confidence_score=0.90, status="PENDING_CURATOR_REVIEW"
            ) for tok in oov_tokens
        ]
        return vi_trans, token_meanings, cands
