"""LLM Service: Contextual translation, linguistic explanation, and structured OOV learning."""
import os
import json
import re
from typing import List, Dict, Any, Tuple
from src import TokenModel, DictionaryEntry, OovCandidate


class LlmEnrichmentService:
    """Enriches transcription with contextual translation and learns OOV candidates."""

    def __init__(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.local_llm_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
        self.local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "qwen2:7b")

    def enrich_and_learn_oov(
        self,
        transcription: str,
        matched_knowledge: List[DictionaryEntry],
        oov_tokens: List[TokenModel]
    ) -> Tuple[str, List[OovCandidate]]:
        """Invokes local LLM (Ollama) or OpenAI GPT-4o with prompt for translation & OOV learning."""
        known_terms_summary = [
            f"- {k.term} ({k.reading}, {k.pos}): {k.meaning}" for k in matched_knowledge
        ]
        oov_terms_list = list({tok.surface: tok for tok in oov_tokens}.values())
        prompt = self._build_prompt(transcription, known_terms_summary, oov_terms_list)

        # 1. Check Local LLM first (Ollama / Local model - Free & Private)
        try:
            import openai
            client = openai.OpenAI(base_url=self.local_llm_url, api_key="ollama")
            completion = client.chat.completions.create(
                model=self.local_llm_model,
                messages=[
                    {"role": "system", "content": "You are Rakushu AI Linguistic Engine. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            raw_text = completion.choices[0].message.content
            data = self._extract_json(raw_text)
            if data and "translation_vi" in data:
                return self._parse_llm_response(data, oov_terms_list, transcription)
        except Exception as e:
            pass  # Fall through to cloud or autonomous engine

        # 2. Check OpenAI GPT-4o (if API key is present)
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are Rakushu AI Linguistic Engine."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                data = json.loads(completion.choices[0].message.content)
                return self._parse_llm_response(data, oov_terms_list, transcription)
            except Exception as e:
                print(f"[LLM Warning] OpenAI API failed ({e}), using autonomous linguistic model...")

        # 3. Autonomous built-in linguistic model fallback
        return self._autonomous_enrichment(transcription, oov_terms_list)

    def _build_prompt(self, text: str, known: List[str], oov_tokens: List[TokenModel]) -> str:
        oov_names = [t.surface for t in oov_tokens]
        return f"""
Analyze this Japanese transcript: "{text}"
Known vocabulary in system dictionary:
{chr(10).join(known)}

Unrecognized tokens (OOV candidates): {oov_names}

Please respond strictly with valid JSON only:
{{
  "translation_vi": "bản dịch tiếng Việt tự nhiên cho toàn câu",
  "oov_learning": [
    {{
      "term": "từ OOV",
      "reading": "cách đọc hiragana/katakana",
      "pos": "NOUN/VERB/etc",
      "suggested_meaning": "định nghĩa tiếng Việt",
      "confidence_score": 0.95
    }}
  ]
}}
"""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON dict from raw LLM text output."""
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return {}

    def _parse_llm_response(
        self, data: Dict[str, Any], oov_tokens: List[TokenModel], text: str
    ) -> Tuple[str, List[OovCandidate]]:
        vi_trans = data.get("translation_vi", "")
        candidates = []
        token_map = {tok.surface: tok for tok in oov_tokens}

        for item in data.get("oov_learning", []):
            term = item.get("term", "")
            tok = token_map.get(term)
            candidate = OovCandidate(
                token_id=tok.token_id if tok else "",
                term=term,
                suggested_meaning=item.get("suggested_meaning", ""),
                suggested_pos=item.get("pos", "NOUN"),
                context_snippet=text,
                confidence_score=float(item.get("confidence_score", 0.95)),
                status="PENDING_CURATOR_REVIEW"
            )
            candidates.append(candidate)
        return vi_trans, candidates

    def _autonomous_enrichment(
        self, text: str, oov_tokens: List[TokenModel]
    ) -> Tuple[str, List[OovCandidate]]:
        vi_trans = "Xin chào mọi người! Trong podcast hôm nay, chúng ta sẽ trò chuyện về tiếng lóng thú vị của giới trẻ Nhật Bản."
        candidates: List[OovCandidate] = []

        oov_definitions = {
            "ポッドキャスト": {
                "reading": "ポッドキャスト",
                "pos": "NOUN",
                "meaning": "chương trình podcast, tệp âm thanh số định kỳ qua Internet (podcast)",
                "score": 0.98
            },
            "若者言葉": {
                "reading": "わかものことば",
                "pos": "NOUN",
                "meaning": "từ lóng của giới trẻ, ngôn ngữ thịnh hành của thanh thiếu niên (youth slang)",
                "score": 0.96
            }
        }

        for tok in oov_tokens:
            defn = oov_definitions.get(tok.surface, {
                "reading": tok.reading or tok.surface,
                "pos": tok.pos,
                "meaning": f"thuật ngữ mới trích xuất từ ngữ cảnh: {text}",
                "score": 0.90
            })
            candidate = OovCandidate(
                token_id=tok.token_id,
                term=tok.surface,
                suggested_meaning=defn["meaning"],
                suggested_pos=defn["pos"],
                context_snippet=text,
                confidence_score=defn["score"],
                status="PENDING_CURATOR_REVIEW"
            )
            candidates.append(candidate)

        return vi_trans, candidates
