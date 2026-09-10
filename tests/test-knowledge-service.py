"""Unit test suite for KnowledgeService SQLite dictionary database and OOV detection."""
import unittest
import os
import sys
import tempfile
import sqlite3

# Adjust search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import KnowledgeService, TokenModel, DictionaryEntry


class TestKnowledgeService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dictionary.db"))

    def test_01_real_database_lookup(self):
        """Tests that real words are successfully retrieved from SQLite with definition tags and rules."""
        if not os.path.exists(self.real_db_path):
            self.skipTest("data/dictionary.db not found, skipping real db lookup")

        svc = KnowledgeService(db_path=self.real_db_path, exclude_terms=set())
        
        # Test basic vocabulary
        entry = svc.lookup("今日")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.term, "今日")
        self.assertEqual(entry.reading, "きょう")
        self.assertIn("hôm nay", entry.meaning.lower())

        # Test verb lookup with definition_tags and rules
        entry_verb = svc.lookup("食べる")
        self.assertIsNotNone(entry_verb)
        self.assertEqual(entry_verb.pos, "VERB")
        self.assertIn("v1", entry_verb.definition_tags)
        self.assertEqual(entry_verb.rules, "v1")
        self.assertGreater(entry_verb.sequence, 0)

        # Test get_tag_info
        tag_info = svc.get_tag_info("v1")
        self.assertIsNotNone(tag_info)
        self.assertEqual(tag_info.name, "v1")
        self.assertIn("Ichidan", tag_info.description_en)
        self.assertIn("Động từ nhóm 2", tag_info.description_vi)

        # Test get_rule_info
        rule_info = svc.get_rule_info("v1")
        self.assertIsNotNone(rule_info)
        self.assertEqual(rule_info.rule_code, "v1")
        self.assertIn("Ichidan", rule_info.name_vi)
        self.assertIn("食べる", rule_info.examples)

    def test_02_match_tokens_and_oov_detection(self):
        """Tests categorizing tokens into known vocabulary vs OOV candidates."""
        svc = KnowledgeService(exclude_terms={"テスト未知語"})
        tokens = [
            TokenModel(surface="日本", lemma="日本", pos="NOUN"),
            TokenModel(surface="テスト未知語", lemma="テスト未知語", pos="NOUN"),
            TokenModel(surface="です", lemma="です", pos="AUX_VERB"),
            TokenModel(surface="。", lemma="。", pos="PUNCTUATION"),
        ]

        matched, oov_tokens = svc.match_tokens(tokens)
        matched_terms = [m.term for m in matched]
        oov_surfaces = [t.surface for t in oov_tokens]

        self.assertIn("日本", matched_terms)
        self.assertIn("テスト未知語", oov_surfaces)
        # Functional grammar tokens should be ignored from both
        self.assertNotIn("です", matched_terms)
        self.assertNotIn("です", oov_surfaces)

    def test_03_register_enriched_oov_persistence(self):
        """Tests that registering an OOV dynamically persists it and enables future lookup."""
        # Use isolated temp database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        try:
            conn = sqlite3.connect(tmp_db)
            conn.execute("""
                CREATE TABLE dictionary (
                    term TEXT NOT NULL, reading TEXT NOT NULL DEFAULT '', pos TEXT NOT NULL DEFAULT 'NOUN',
                    definition_tags TEXT DEFAULT '', rules TEXT DEFAULT '', score INTEGER DEFAULT 1,
                    meaning TEXT NOT NULL, sequence INTEGER DEFAULT 0, term_tags TEXT DEFAULT '',
                    PRIMARY KEY (term, reading)
                );
            """)
            conn.commit()
            conn.close()

            svc = KnowledgeService(db_path=tmp_db, exclude_terms={"パリピ"})
            self.assertIsNone(svc.lookup("パリピ"))

            # Register OOV
            new_entry = DictionaryEntry(
                term="パリピ", reading="ぱりぴ", pos="NOUN",
                meaning="dân quẩy, người thích tiệc tùng (party people)", score=1
            )
            svc.register_enriched_oov(new_entry)

            # Check lookup now works
            queried = svc.lookup("パリピ")
            self.assertIsNotNone(queried)
            self.assertEqual(queried.term, "パリピ")
            self.assertEqual(queried.reading, "ぱりぴ")
            self.assertIn("tiệc tùng", queried.meaning)

            # Verify persisted in SQLite
            conn = sqlite3.connect(tmp_db)
            cur = conn.cursor()
            cur.execute("SELECT term, reading, meaning FROM dictionary WHERE term = 'パリピ'")
            row = cur.fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "パリピ")
        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)

    def test_04_fallback_when_no_database(self):
        """Tests that KnowledgeService falls back to built-in vocabulary when DB file is missing."""
        svc = KnowledgeService(db_path="non_existent_file.db", exclude_terms=set())
        entry = svc.lookup("今日")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.reading, "きょう")
        self.assertIn("hôm nay", entry.meaning)


if __name__ == "__main__":
    unittest.main()
