"""Comprehensive test suite for the Rakushu AI processing pipeline."""
import unittest
import os
import sys
import importlib

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    SubtitleSegment,
    SentenceSubtitle,
    OovCandidate,
    NlpService,
    KnowledgeService,
    LlmEnrichmentService,
    BunsetsuService,
)
runner_mod = importlib.import_module("src.pipeline-runner")
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
        """Validates two-stage translation: full context + sentence & token meanings."""
        llm_svc = LlmEnrichmentService()
        nlp_svc = NlpService()
        knowledge_svc = KnowledgeService()

        # Stage 1: Full transcription translation
        full_trans = llm_svc.translate_full_transcription(self.sample_text)
        self.assertTrue(len(full_trans) > 0)
        self.assertIn("chào", full_trans.lower())

        # Stage 2: Sentence 1 with context
        s1 = SubtitleSegment(text="皆さん、こんにちは。")
        toks1 = nlp_svc.tokenize(s1)
        mk1, oov1 = knowledge_svc.match_tokens(toks1)
        trans1, t_meanings1, cands1 = llm_svc.translate_sentence_with_context(
            sentence_text=s1.text,
            full_text=self.sample_text,
            full_translation=full_trans,
            tokens=toks1,
            matched_knowledge=mk1,
            oov_tokens=oov1
        )
        self.assertIn("chào", trans1.lower())
        self.assertIn("皆さん", t_meanings1)

        # Stage 2: Sentence 2 with context
        s2 = SubtitleSegment(text="今日のポッドキャストでは、日本の面白い若者言葉について話します。")
        toks2 = nlp_svc.tokenize(s2)
        mk2, oov2 = knowledge_svc.match_tokens(toks2)
        trans2, t_meanings2, cands2 = llm_svc.translate_sentence_with_context(
            sentence_text=s2.text,
            full_text=self.sample_text,
            full_translation=full_trans,
            tokens=toks2,
            matched_knowledge=mk2,
            oov_tokens=oov2
        )
        self.assertTrue(len(trans2) > 0)
        self.assertEqual(len(cands2), 2)
        self.assertIn("ポッドキャスト", t_meanings2)
        self.assertIn("若者言葉", t_meanings2)

    def test_05_bunsetsu_grouping(self):
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

    def test_06_end_to_end_pipeline(self):
        """Runs full end-to-end pipeline and checks sentence-level result objects."""
        media_file = "samples/japanese_podcast_10s.mp4"
        if not os.path.exists(media_file):
            create_media = importlib.import_module("samples.create-sample-media")
            create_media.create_sample_media(media_file)

        result = run_pipeline(media_file)

        self.assertIsNotNone(result.segment)
        self.assertTrue(len(result.full_translation) > 0)
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
        self.assertIn("full_translation", result_dict)
        self.assertIn("sentences", result_dict)
        self.assertEqual(len(result_dict["sentences"]), 2)


if __name__ == "__main__":
    unittest.main()

