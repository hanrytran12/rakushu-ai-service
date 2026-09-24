"""Unit tests for Hierarchical Knowledge Matching: Grammar -> Phrase -> CompoundWord -> Word."""
import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    KnowledgeService, GinzaReferenceService, TokenModel, MatchedKnowledgeUnit
)


class TestHierarchicalMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        staging_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "jmdict_1000_staging.db"))
        cls.ks = KnowledgeService(db_path=staging_db)
        cls.ginza_ref = GinzaReferenceService()

    def test_01_ginza_reference_service(self):
        """Tests linguistic explanations from GiNZA Reference DB."""
        xpos = self.ginza_ref.explain_xpos("助詞-格助詞")
        self.assertIsNotNone(xpos)
        self.assertIn("Trợ từ cách", xpos["vietnamese"])

        dep_compound = self.ginza_ref.explain_deprel("compound")
        self.assertIsNotNone(dep_compound)
        self.assertIn("Từ ghép phức hợp", dep_compound["vietnamese"])

        dep_fixed = self.ginza_ref.explain_deprel("fixed")
        self.assertIsNotNone(dep_fixed)
        self.assertIn("Cụm từ ngữ cố định", dep_fixed["vietnamese"])

    def test_02_grammar_priority_matching(self):
        """Tests that grammar patterns are matched first as a whole block without shredding."""
        # Simulated tokens for "食べるわけにはいかない"
        tokens = [
            TokenModel(surface="食べる", lemma="食べる", pos="VERB"),
            TokenModel(surface="わけ", lemma="わけ", pos="NOUN"),
            TokenModel(surface="に", lemma="に", pos="PARTICLE"),
            TokenModel(surface="は", lemma="は", pos="PARTICLE"),
            TokenModel(surface="いか", lemma="いく", pos="VERB"),
            TokenModel(surface="ない", lemma="ない", pos="AUX_VERB"),
        ]
        matched, oov = self.ks.match_hierarchical(tokens, "食べるわけにはいかない")
        grammar_matches = [m for m in matched if m.unit_type == "GRAMMAR"]
        self.assertTrue(len(grammar_matches) >= 1)
        g = grammar_matches[0]
        self.assertEqual(g.surface, "わけにはいかない")
        self.assertIn(g.grammar_info.jlpt_level, ["N3", "N2", "N1"])
        # Ensure individual particles 'に' and 'は' were NOT looked up as isolated words
        word_surfaces = [m.surface for m in matched if m.unit_type == "WORD"]
        self.assertNotIn("わけ", word_surfaces)
        self.assertNotIn("いか", word_surfaces)

    def test_03_phrase_priority_matching(self):
        """Tests that idioms and set phrases are matched as a whole block."""
        # Simulated tokens for "あっという間に終わった"
        tokens = [
            TokenModel(surface="あっ", lemma="あっ", pos="INTERJECTION"),
            TokenModel(surface="という", lemma="という", pos="PARTICLE"),
            TokenModel(surface="間", lemma="間", pos="NOUN"),
            TokenModel(surface="に", lemma="に", pos="PARTICLE"),
            TokenModel(surface="終わっ", lemma="終わる", pos="VERB"),
            TokenModel(surface="た", lemma="た", pos="AUX_VERB"),
        ]
        matched, oov = self.ks.match_hierarchical(tokens, "あっという間に終わった")
        phrase_matches = [m for m in matched if m.unit_type == "PHRASE"]
        self.assertTrue(len(phrase_matches) >= 1)
        p = phrase_matches[0]
        self.assertEqual(p.surface, "あっという間に")
        self.assertIn("chốc lát", p.meaning.lower())

    def test_04_compound_word_priority_matching(self):
        """Tests that compound verbs and nouns are matched and decomposed into components."""
        # Simulated tokens for "空を仰ぎ見る"
        tokens = [
            TokenModel(surface="空", lemma="空", pos="NOUN"),
            TokenModel(surface="を", lemma="を", pos="PARTICLE"),
            TokenModel(surface="仰ぎ", lemma="仰ぐ", pos="VERB"),
            TokenModel(surface="見る", lemma="見る", pos="VERB"),
        ]
        matched, oov = self.ks.match_hierarchical(tokens, "空を仰ぎ見る")
        compound_matches = [m for m in matched if m.unit_type == "COMPOUND_WORD"]
        self.assertTrue(len(compound_matches) >= 1)
        c = compound_matches[0]
        self.assertEqual(c.surface, "仰ぎ見る")
        self.assertEqual(c.compound_info.components, ["仰ぐ", "見る"])
        self.assertEqual(c.compound_info.transitivity, "他")

    def test_05_word_fallback(self):
        """Tests that remaining unconsumed tokens fall back to single word dictionary lookup."""
        tokens = [
            TokenModel(surface="今日", lemma="今日", pos="NOUN"),
            TokenModel(surface="は", lemma="は", pos="PARTICLE"),
            TokenModel(surface="日本", lemma="日本", pos="NOUN"),
        ]
        matched, oov = self.ks.match_hierarchical(tokens, "今日は日本")
        words = [m for m in matched if m.unit_type == "WORD"]
        word_surfaces = [w.surface for w in words]
        self.assertIn("今日", word_surfaces)
        self.assertIn("日本", word_surfaces)


if __name__ == "__main__":
    unittest.main()
