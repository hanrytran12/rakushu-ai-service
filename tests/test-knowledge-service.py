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
        """Tests that real words are successfully retrieved from SQLite with Vietnamese meanings."""
        if not os.path.exists(self.real_db_path):
            self.skipTest("data/dictionary.db not found, skipping real db lookup")

        svc = KnowledgeService(db_path=self.real_db_path, exclude_terms=set())
        
        # Test basic vocabulary
        entry = svc.lookup("今日")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.term, "今日")
        self.assertEqual(entry.reading, "きょう")
        self.assertIn("hôm nay", entry.meaning.lower())
        self.assertEqual(entry.jlpt_level, "N5")

        # Test verb lookup
        entry_verb = svc.lookup("話す")
        self.assertIsNotNone(entry_verb)
        self.assertEqual(entry_verb.pos, "VERB")
        self.assertEqual(entry_verb.jlpt_level, "N3")

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
                    term TEXT PRIMARY KEY, reading TEXT, pos TEXT, meaning TEXT,
                    jlpt_level TEXT, commonality INTEGER
                );
            """)
            conn.commit()
            conn.close()

            svc = KnowledgeService(db_path=tmp_db, exclude_terms={"パリピ"})
            self.assertIsNone(svc.lookup("パリピ"))

            # Register OOV
            new_entry = DictionaryEntry(
                term="パリピ", reading="ぱりぴ", pos="NOUN",
                meaning="dân quẩy, người thích tiệc tùng (party people)", jlpt_level="Slang", commonality=1
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
        self.assertEqual(entry.jlpt_level, "N5")


if __name__ == "__main__":
    unittest.main()
