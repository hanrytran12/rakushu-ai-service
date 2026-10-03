"""Comprehensive test suite for the Rakushu AI processing pipeline."""
import unittest
import os
import sys
import importlib

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import SubtitleSegment, SentenceSubtitle, OovCandidate
from src.services import NlpService, KnowledgeService, LlmEnrichmentService, BunsetsuService
runner_mod = importlib.import_module("src.pipeline.pipeline_runner")
run_pipeline = runner_mod.run_pipeline


class TestRakushuPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_text = "皆さん、こんにちは。今日のポッドキャストでは、日本の面白い若者言葉について話します。"
        cls.segment = SubtitleSegment(
            start_time=0.0,
            end_time=10.0,
            text=cls.sample_text
        )

    def test_01_ginza_sentence_splitting(self):
        """Validates GiNZA-based sentence boundary splitting."""
        nlp_svc = NlpService()
        sents = nlp_svc.split_sentences(self.sample_text)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0], "皆さん、こんにちは。")
        self.assertEqual(sents[1], "今日のポッドキャストでは、日本の面白い若者言葉について話します。")

    def test_02_nlp_tokenization(self):
        """Validates tokenization, POS tagging, lemmatization, and readings via GiNZA."""
        nlp_svc = NlpService()
        tokens = nlp_svc.tokenize(self.segment)

        self.assertGreater(len(tokens), 8)
        surfaces = [t.surface for t in tokens]
        self.assertIn("皆さん", surfaces)
        self.assertIn("今日", surfaces)
        self.assertIn("ポッドキャスト", surfaces)
        self.assertIn("若者言葉", surfaces)

        # Validate token attributes
        minasan = next(t for t in tokens if t.surface == "皆さん")
        self.assertEqual(minasan.pos, "NOUN")
        self.assertIn(minasan.reading.upper(), ["ミナサン", "みなさん"])

    def test_03_knowledge_and_oov_detection(self):
        """Validates matching against knowledge base and flagging OOV candidates."""
        nlp_svc = NlpService()
        tokens = nlp_svc.tokenize(self.segment)

        knowledge_svc = KnowledgeService()
        matched, oov_tokens = knowledge_svc.match_tokens(tokens)

        matched_terms = [k.term for k in matched]
        oov_surfaces = [t.surface for t in oov_tokens]

        # Verify known words matched
        self.assertIn("皆さん", matched_terms)
        self.assertIn("今日", matched_terms)
        self.assertIn("日本", matched_terms)
        self.assertIn("面白い", matched_terms)

        # Verify omitted words were flagged as OOV
        self.assertIn("ポッドキャスト", oov_surfaces)
        self.assertIn("若者言葉", oov_surfaces)

    def test_04_sentence_level_translation(self):
        """Validates sentence translation with local context and token meanings."""
        llm_svc = LlmEnrichmentService()
        nlp_svc = NlpService()
        knowledge_svc = KnowledgeService()

        # Stage 2: Sentence 1 with context
        s1 = SubtitleSegment(text="皆さん、こんにちは。")
        toks1 = nlp_svc.tokenize(s1)
        mk1, oov1 = knowledge_svc.match_tokens(toks1)
        trans1, t_meanings1, cands1 = llm_svc.translate_sentence_with_context(
            sentence_text=s1.text,
            context_text=self.sample_text,
            tokens=toks1,
            matched_knowledge=mk1,
            oov_tokens=oov1,
            token_candidates=knowledge_svc.build_token_candidates(toks1),
        )
        self.assertIn("chào", trans1.lower())
        self.assertIn("皆さん", t_meanings1)

        # Stage 2: Sentence 2 with context
        s2 = SubtitleSegment(text="今日のポッドキャストでは、日本の面白い若者言葉について話します。")
        toks2 = nlp_svc.tokenize(s2)
        mk2, oov2 = knowledge_svc.match_tokens(toks2)
        trans2, t_meanings2, cands2 = llm_svc.translate_sentence_with_context(
            sentence_text=s2.text,
            context_text=self.sample_text,
            tokens=toks2,
            matched_knowledge=mk2,
            oov_tokens=oov2,
            token_candidates=knowledge_svc.build_token_candidates(toks2),
        )
        self.assertTrue(len(trans2) > 0)
        self.assertEqual(len(cands2), 2)
        self.assertNotIn("ポッドキャスト", t_meanings2)
        self.assertNotIn("若者言葉", t_meanings2)

    def test_05_local_context_builder(self):
        """Validates bounded local context and context-dependent expansion."""
        llm_svc = LlmEnrichmentService()
        sentences = ["一文目です。", "二文目です。", "三文目です。", "四文目です。", "五文目です。"]

        context = llm_svc.build_sentence_context(sentences, 3, before=3, after=2)
        self.assertIn("[CONTEXT 1] 一文目です。", context)
        self.assertIn("[CONTEXT 3] 三文目です。", context)
        self.assertIn("[TARGET 4] 四文目です。", context)
        self.assertIn("[CONTEXT 5] 五文目です。", context)


    def test_06_translation_chunk_planner_keeps_sentence_boundaries(self):
        """Plans bounded chunks without splitting individual sentences."""
        llm_svc = LlmEnrichmentService()
        sentences = ["A" * 700, "B" * 700, "C" * 700, "D" * 100]
        chunks = llm_svc.build_translation_chunks(sentences, max_chars=1400)
        self.assertEqual(chunks, [(0, 2), (2, 4)])

    def test_06b_translation_chunk_planner_limits_sentence_count(self):
        """Planner must cap sentence count even when character budget is not reached."""
        llm_svc = LlmEnrichmentService()
        sentences = [f"Câu {i}。" for i in range(25)]
        chunks = llm_svc.build_translation_chunks(sentences, max_chars=900)
        self.assertEqual(chunks, [(0, 12), (12, 24), (24, 25)])

    def test_07_adaptive_retry_changes_llm_options(self):
        """A failed inference retry must not resend the exact same decoding config."""
        from unittest.mock import patch
        class FakeResponse:
            def __init__(self, status_code, body):
                self.status_code = status_code
                self._body = body
                self.text = body if isinstance(body, str) else str(body)

            def json(self):
                return self._body

        calls = []
        responses = [
            FakeResponse(500, {"error": "prediction aborted, token repeat limit reached"}),
            FakeResponse(200, {"message": {"content": '{"translation_vi":"dịch"}'}}),
        ]

        def fake_post(url, json, timeout):
            calls.append(json["options"])
            return responses.pop(0)

        llm_svc = LlmEnrichmentService()
        with patch("requests.post", side_effect=fake_post):
            result = llm_svc._call_llm("dịch câu này")

        self.assertEqual(result["translation_vi"], "dịch")
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])
        self.assertGreater(calls[1]["repeat_penalty"], calls[0]["repeat_penalty"])

    def test_08_chunk_translation_validates_sentence_numbers(self):
        """Chunk translation returns valid items and leaves missing sentences for fallback."""
        llm_svc = LlmEnrichmentService()
        llm_svc._call_llm = lambda prompt, system_prompt=None, max_attempts=None: {
            "sentences": [
                {"sentence_number": 1, "translation_vi": "dịch 1", "token_meanings": {}, "oov_learning": []},
                {"sentence_number": 3, "translation_vi": "dịch 3", "token_meanings": {}, "oov_learning": []},
            ]
        }
        result = llm_svc.translate_chunk([
            {"sentence_number": 1, "text": "一。"},
            {"sentence_number": 2, "text": "二。"},
            {"sentence_number": 3, "text": "三。"},
        ])
        self.assertEqual(sorted(result), [1, 3])
        self.assertEqual(result[1]["translation_vi"], "dịch 1")

    def test_09_adaptive_chunk_splits_after_partial_batch(self):
        """A persistently partial batch is split instead of retrying the same large prompt."""
        llm_svc = LlmEnrichmentService()
        calls = []

        def fake_call(prompt, system_prompt=None, max_attempts=None):
            import json
            calls.append((prompt, max_attempts))
            payload = json.loads(prompt.split("INPUT:\n", 1)[1].split("\n\nTrả về JSON", 1)[0])
            items = payload
            if len(items) > 2:
                return {
                    "sentences": [
                        {"sentence_number": items[0]["sentence_number"], "translation_vi": "dịch"}
                    ]
                }
            return {
                "sentences": [
                    {"sentence_number": item["sentence_number"], "translation_vi": "dịch"}
                    for item in items
                ]
            }

        llm_svc._call_llm = fake_call
        result = llm_svc.translate_chunk([
            {"sentence_number": 1, "text": "一。"},
            {"sentence_number": 2, "text": "二。"},
            {"sentence_number": 3, "text": "三。"},
            {"sentence_number": 4, "text": "四。"},
        ])

        self.assertEqual(sorted(result), [1, 2, 3, 4])
        self.assertTrue(any(len(prompt) > 0 and max_attempts == 2 for prompt, max_attempts in calls))
        self.assertGreater(len(calls), 2)

    def test_10_context_fallback_uses_expanded_window(self):
        """Retries with expanded context only when the first context is insufficient."""
        llm_svc = LlmEnrichmentService()
        prompts = []
        responses = iter([
            {"context_sufficient": False, "context_issue": "referent unclear"},
            {"translation_vi": "dịch", "token_meanings": {}, "oov_learning": [], "context_sufficient": True},
        ])
        def fake_call(prompt, system_prompt=None, max_attempts=None):
            prompts.append(prompt)
            return next(responses)
        llm_svc._call_llm = fake_call
        llm_svc.translate_sentence_with_context(
            sentence_text="Câu 5.",
            context_text="Câu 2.\nCâu 3.\nCâu 4.\nCâu 5.",
            expanded_context_text="Câu 1.\nCâu 2.\nCâu 3.\nCâu 4.\nCâu 5.\nCâu 6.",
        )
        self.assertEqual(len(prompts), 2)
        self.assertIn("Câu 1.", prompts[1])
        self.assertIn("Câu 6.", prompts[1])
    def test_11_bunsetsu_grouping(self):
        """Validates Bunsetsu phrase segmentation rules (Jiritsugo + Fuzokugo)."""
        nlp_svc = NlpService()
        tokens = nlp_svc.tokenize(self.segment)

        bunsetsu_svc = BunsetsuService()
        phrases = bunsetsu_svc.group_bunsetsu(self.segment, tokens)

        self.assertEqual(len(phrases), 8)
        phrase_texts = [p.text for p in phrases]
        expected_chunks = [
            "皆さん、", "こんにちは。", "今日の", "ポッドキャストでは、",
            "日本の", "面白い", "若者言葉について", "話します。"
        ]
        self.assertEqual(phrase_texts, expected_chunks)

        for idx, phrase in enumerate(phrases, 1):
            self.assertEqual(phrase.phrase_order, idx)
            self.assertTrue(len(phrase.translation) > 0)
            self.assertLess(phrase.start_time, phrase.end_time)

    def test_12_sentence_translation_uses_local_context(self):
        """Ensures Stage 2 receives nearby context instead of the transcript prefix."""
        llm_svc = LlmEnrichmentService()
        captured = {}
        llm_svc._call_llm = lambda prompt, system_prompt=None: (captured.update(prompt=prompt) or {
            "translation_vi": "dịch",
            "token_meanings": {},
            "oov_learning": [],
        })
        sentences = [f"Câu {i} về chủ đề riêng." for i in range(1, 8)]
        target = sentences[6]
        context = llm_svc.build_sentence_context(sentences, 6, before=3, after=2)
        llm_svc.translate_sentence_with_context(sentence_text=target, context_text=context)
        self.assertIn("Câu 4 về chủ đề riêng.", captured["prompt"])
        self.assertIn("Câu 7 về chủ đề riêng.", captured["prompt"])
        self.assertNotIn("Câu 1 về chủ đề riêng.", captured["prompt"])
        self.assertIn("Câu cần dịch", captured["prompt"])
    def test_13_end_to_end_pipeline(self):
        """Runs full end-to-end pipeline and checks sentence-level result objects."""
        media_file = "samples/japanese_podcast_10s.mp4"
        if not os.path.exists(media_file):
            create_media = importlib.import_module("samples.create-sample-media")
            create_media.create_sample_media(media_file)

        result = run_pipeline(media_file)

        self.assertIsNotNone(result.segment)
        self.assertEqual(len(result.sentences), 2)
        self.assertEqual(len(result.bunsetsu_phrases), 8)
        self.assertGreaterEqual(len(result.oov_candidates), 2)

        # Validate TokenModel has context_meaning populated
        tokens_with_meanings = [t for t in result.tokens if t.context_meaning]
        self.assertGreater(len(tokens_with_meanings), 3)

        # Validate SentenceSubtitle structures
        s1 = result.sentences[0]
        self.assertEqual(s1.segment.text, "皆さん、こんにちは。")
        self.assertTrue(len(s1.translation) > 0)
        self.assertEqual(len(s1.bunsetsu_phrases), 2)

        s2 = result.sentences[1]
        self.assertIn(s2.segment.text, [
            "今日のポッドキャストでは、日本の面白い若者言葉について話します。",
            "今日のポッドキャストでは、日本の面白い若もの言葉について話します。"
        ])
        self.assertTrue(len(s2.translation) > 0)
        self.assertEqual(len(s2.bunsetsu_phrases), 6)

        # Validate exportable dictionary format
        result_dict = result.to_dict()
        self.assertNotIn("full_translation", result_dict)
        self.assertIn("sentences", result_dict)
        self.assertEqual(len(result_dict["sentences"]), 2)



    def test_14_dictionary_candidate_selection_resolves_server_meaning(self):
        """LLM selects a candidate ID; backend supplies the actual dictionary meaning."""
        llm_svc = LlmEnrichmentService()
        tokens = [SubtitleSegment(text="やすい")]
        token_model = NlpService().tokenize(tokens[0])[0]
        token_model.surface = "やすい"
        token_model.lemma = "やすい"
        token_model.pos = "ADJECTIVE"
        candidates = {
            "やすい": [
                {
                    "candidate_id": "t1-c1",
                    "term": "やすい",
                    "reading": "やすい",
                    "pos": "ADJECTIVE",
                    "meaning": "rẻ, giá thấp",
                },
                {
                    "candidate_id": "t1-c2",
                    "term": "やすい",
                    "reading": "やすい",
                    "pos": "ADJECTIVE",
                    "meaning": "dễ, dễ dàng",
                },
            ]
        }

        selected = llm_svc.resolve_token_selections(
            [{"token": "やすい", "candidate_id": "t1-c1"}],
            candidates,
            [token_model],
        )
        self.assertEqual(selected["やすい"], "rẻ, giá thấp")

        invalid = llm_svc.resolve_token_selections(
            [{"token": "やすい", "candidate_id": "fake"}],
            candidates,
            [token_model],
        )
        self.assertEqual(invalid, {})

    def test_15_single_dictionary_candidate_does_not_require_llm_selection(self):
        """A deterministic single sense can be resolved without an LLM-generated meaning."""
        llm_svc = LlmEnrichmentService()
        token = NlpService().tokenize(SubtitleSegment(text="日本"))[0]
        candidates = {
            "日本": [{
                "candidate_id": "t1-c1",
                "term": "日本",
                "reading": "にほん",
                "pos": "NOUN",
                "meaning": "Nhật Bản",
            }]
        }
        result = llm_svc.resolve_token_selections([], candidates, [token])
        self.assertEqual(result["日本"], "Nhật Bản")

if __name__ == "__main__":
    unittest.main()