"""Unit test suite for Grammar, Phrase, and CompoundWord Knowledge in KnowledgeService."""
import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import KnowledgeService, GrammarEntry, PhraseEntry, CompoundWordEntry


class TestExtendedKnowledge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        staging_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "jmdict_1000_staging.db"))
        dict_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dictionary.db"))
        cls.real_db_path = staging_db if os.path.exists(staging_db) else dict_db
        if not os.path.exists(cls.real_db_path):
            raise unittest.SkipTest("Target dictionary database not found, skipping tests")
        cls.svc = KnowledgeService(db_path=cls.real_db_path)

    def test_01_grammar_lookup(self):
        """Tests looking up grammar patterns with JLPT level, formation, and meaning."""
        # Test N1 pattern
        g1 = self.svc.lookup_grammar("あっての")
        self.assertIsNotNone(g1)
        self.assertEqual(g1.jlpt_level, "N1")
        self.assertIn("あっての", g1.pattern)
        self.assertTrue(len(g1.formation) > 0)
        self.assertTrue(len(g1.meaning) > 0)

        # Test with prefix tilde
        g2 = self.svc.lookup_grammar("～あっての")
        self.assertIsNotNone(g2)
        self.assertEqual(g2.jlpt_level, "N1")

        # Test another pattern
        g3 = self.svc.lookup_grammar("以外の何ものでもない")
        self.assertIsNotNone(g3)
        self.assertEqual(g3.jlpt_level, "N1")

    def test_02_phrase_lookup(self):
        """Tests looking up idiomatic expressions, proverbs, and yojijukugo."""
        # Test expression
        p1 = self.svc.lookup_phrase("阿吽の呼吸")
        self.assertIsNotNone(p1)
        self.assertEqual(p1.reading, "あうんのこきゅう")
        self.assertEqual(p1.phrase_type, "EXPRESSION")

        # Test idiom
        p2 = self.svc.lookup_phrase("玉に瑕")
        self.assertIsNotNone(p2)
        self.assertEqual(p2.reading, "たまにきず")
        self.assertIn(p2.phrase_type, ["IDIOM", "EXPRESSION"])
        self.assertTrue(
            "fly in the ointment" in p2.meaning.lower()
            or "flaw" in p2.meaning.lower()
            or "thiếu sót" in p2.meaning.lower()
            or "điểm yếu" in p2.meaning.lower()
        )

    def test_03_compound_word_lookup(self):
        """Tests looking up compound verbs and compound nouns with decomposed components."""
        # Test Compound Verb from NINJAL VVLexicon
        c_verb = self.svc.lookup_compound_word("仰ぎ見る")
        self.assertIsNotNone(c_verb)
        self.assertEqual(c_verb.term, "仰ぎ見る")
        self.assertEqual(c_verb.reading, "あおぎみる")
        self.assertEqual(c_verb.compound_type, "VERB")
        self.assertEqual(c_verb.components, ["仰ぐ", "見る"])
        self.assertEqual(c_verb.transitivity, "他")
        self.assertEqual(c_verb.structure, "V1 + V2")

        # Test another Compound Verb
        c_verb2 = self.svc.lookup_compound_word("煽り立てる")
        self.assertIsNotNone(c_verb2)
        self.assertEqual(c_verb2.compound_type, "VERB")
        self.assertEqual(c_verb2.components, ["煽る", "立てる"])

        # Test Compound Noun
        # Look up any noun with components in compound_words table
        import sqlite3
        conn = sqlite3.connect(self.real_db_path)
        cur = conn.cursor()
        cur.execute("SELECT term FROM compound_words WHERE compound_type = 'NOUN' LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            noun_term = row[0]
            c_noun = self.svc.lookup_compound_word(noun_term)
            self.assertIsNotNone(c_noun)
            self.assertEqual(c_noun.compound_type, "NOUN")
            self.assertGreaterEqual(len(c_noun.components), 2)

    def test_04_edge_cases(self):
        """Tests invalid or missing terms."""
        self.assertIsNone(self.svc.lookup_grammar(""))
        self.assertIsNone(self.svc.lookup_grammar("nonexistent_grammar_pattern_xyz"))
        self.assertIsNone(self.svc.lookup_phrase(""))
        self.assertIsNone(self.svc.lookup_phrase("nonexistent_phrase_xyz"))
        self.assertIsNone(self.svc.lookup_compound_word(""))
        self.assertIsNone(self.svc.lookup_compound_word("nonexistent_compound_xyz"))


if __name__ == "__main__":
    unittest.main()
