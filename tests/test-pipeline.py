"""Comprehensive test suite for the Rakushu AI processing pipeline."""
import unittest
import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    SubtitleSegment,
    TokenModel,
    DictionaryEntry,
    OovCandidate,
    BunsetsuPhrase,
    AsrService,
    NlpService,
    KnowledgeService,
    LlmEnrichmentService,
    BunsetsuService,
)
import importlib
runner_mod = importlib.import_module("src.pipeline-runner")
run_pipeline = runner_mod.run_pipeline


class TestRakushuPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_text = "皆さん、こんにちは。今日のポッドキャストでは、日本の面白い若者言葉について話します。"
        cls.segment = SubtitleSegment(
            start_time=0.0,
            end_time=8.4,
            text=cls.sample_text
        )

    def test_01_nlp_tokenization(self):
        """Validates tokenization, POS tagging, lemmatization, and readings."""
        nlp_svc = NlpService()
        tokens = nlp_svc.tokenize(self.segment)

        self.assertGreater(len(tokens), 10)
        surfaces = [t.surface for t in tokens]
        self.assertIn("皆さん", surfaces)
        self.assertIn("今日", surfaces)
        self.assertIn("ポッドキャスト", surfaces)
        self.assertIn("若者言葉", surfaces)
        # Janome segments inflected verb into stem and auxiliary verb (話し + ます)
        self.assertTrue("話し" in surfaces or "話します" in surfaces)

        # Validate token attributes
        minasan = next(t for t in tokens if t.surface == "皆さん")
        self.assertEqual(minasan.pos, "NOUN")
        self.assertEqual(minasan.reading.upper(), "ミナサン")

        hanashi = next(t for t in tokens if t.surface in ["話し", "話します"])
        self.assertEqual(hanashi.pos, "VERB")
        self.assertEqual(hanashi.lemma, "話す")

    def test_02_knowledge_and_oov_detection(self):
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

    def test_03_llm_oov_learning_schema(self):
        """Validates that LLM output structures OOV terms according to DICTIONARY_ENTRY schema."""
        nlp_svc = NlpService()
        tokens = nlp_svc.tokenize(self.segment)
        knowledge_svc = KnowledgeService()
        matched, oov_tokens = knowledge_svc.match_tokens(tokens)

        llm_svc = LlmEnrichmentService()
        translation_vi, enriched_oovs = llm_svc.enrich_and_learn_oov(
            self.segment.text, matched, oov_tokens
        )

        self.assertTrue(len(translation_vi) > 0)
        self.assertEqual(len(enriched_oovs), 2)

        terms = [e.term for e in enriched_oovs]
        self.assertIn("ポッドキャスト", terms)
        self.assertIn("若者言葉", terms)

        for cand in enriched_oovs:
            self.assertIsInstance(cand, OovCandidate)
            self.assertTrue(cand.suggested_meaning)
            self.assertTrue(cand.suggested_pos)
            self.assertGreaterEqual(cand.confidence_score, 0.8)
            self.assertEqual(cand.status, "PENDING_CURATOR_REVIEW")

    def test_04_bunsetsu_grouping(self):
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

        # Verify all phrases have translations and order
        for idx, phrase in enumerate(phrases, 1):
            self.assertEqual(phrase.phrase_order, idx)
            self.assertTrue(len(phrase.translation) > 0)
            self.assertLess(phrase.start_time, phrase.end_time)

    def test_05_end_to_end_pipeline(self):
        """Runs full end-to-end pipeline and checks consolidated result object."""
        media_file = "samples/japanese_podcast_10s.mp4"
        if not os.path.exists(media_file):
            import importlib
            create_media = importlib.import_module("samples.create-sample-media")
            create_media.create_sample_media(media_file)

        result = run_pipeline(media_file)

        self.assertIsNotNone(result.segment)
        self.assertIn(len(result.bunsetsu_phrases), [8, 9])
        self.assertGreaterEqual(len(result.oov_candidates), 2)
        self.assertGreater(len(result.matched_knowledge), 4)

        # Validate exportable dictionary format
        result_dict = result.to_dict()
        self.assertIn("segment", result_dict)
        self.assertIn("bunsetsu_phrases", result_dict)
        self.assertIn("oov_candidates", result_dict)


if __name__ == "__main__":
    unittest.main()
