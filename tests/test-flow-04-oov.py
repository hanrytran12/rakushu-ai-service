"""Comprehensive Unit & Integration Test Suite for Flow 4: OOV Detection & Knowledge Enhancement."""
import os
import sys
import tempfile
import sqlite3
import unittest
import importlib
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    TokenModel, SubtitleSegment, OovCandidate, CuratorReview,
    OovStatus, CuratorDecision, DictionaryEntry,
    KnowledgeService, OovService, LlmOovValidator
)

_curator_models = importlib.import_module("src.curator-models")
CuratorReviewRequest = _curator_models.CuratorReviewRequest
CuratorManualAddRequest = _curator_models.CuratorManualAddRequest

api_server = importlib.import_module("src.api-server")
app = api_server.app


class TestFlow04OovLifecycle(unittest.TestCase):
    """Test suite covering Workflow 4 end-to-end: System, AI, Linguistic Curator, Learner."""

    def setUp(self):
        fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(self.temp_db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE dictionary (
                term TEXT, reading TEXT, pos TEXT, definition_tags TEXT,
                rules TEXT, score INTEGER, meaning TEXT, sequence INTEGER, term_tags TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE staging_entries (
                term TEXT, reading TEXT, pos TEXT, meaning_vi TEXT, gloss_en TEXT
            )
        """)
        conn.commit()
        conn.close()

        self.knowledge_svc = KnowledgeService(db_path=self.temp_db_path)
        self.oov_svc = OovService(db_path=self.temp_db_path, knowledge_service=self.knowledge_svc)
        self.validator = LlmOovValidator()
        self.client = TestClient(app)

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except Exception:
                pass

    def test_01_oov_morphological_prefilter(self):
        """Validates that non-Japanese characters, single particles, and punctuation are rejected."""
        self.assertTrue(self.validator.is_morphologically_plausible("推し"))
        self.assertTrue(self.validator.is_morphologically_plausible("草生える"))
        self.assertTrue(self.validator.is_morphologically_plausible("チルい"))

        # Invalid / Discard candidates
        self.assertFalse(self.validator.is_morphologically_plausible("12345"))
        self.assertFalse(self.validator.is_morphologically_plausible("...!?"))
        self.assertFalse(self.validator.is_morphologically_plausible("は"))
        self.assertFalse(self.validator.is_morphologically_plausible("を"))
        self.assertFalse(self.validator.is_morphologically_plausible(""))

    def test_02_ai_validation_and_enhancement(self):
        """Tests AI validation logic and tentative definition generation."""
        # Valid token
        valid_tok = TokenModel(surface="推し", pos="NOUN", reading="おし")
        is_valid, cand = self.validator.validate_and_enhance(valid_tok, context_text="私の推しは彼です。")
        self.assertTrue(is_valid)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.term, "推し")
        self.assertEqual(cand.tentative_reading, "おし")
        self.assertGreaterEqual(cand.confidence_score, 0.5)
        self.assertEqual(cand.status, OovStatus.PENDING_CURATOR_REVIEW)

        # Invalid noise token
        invalid_tok = TokenModel(surface="999", pos="NUM", reading="")
        is_valid_inv, cand_inv = self.validator.validate_and_enhance(invalid_tok, context_text="テスト")
        self.assertFalse(is_valid_inv)
        self.assertIsNone(cand_inv)

    def test_03_flow_4_adapt_branch(self):
        """Full lifecycle: Candidate persisted -> Curator reviews and ADAPTS -> Merged to dictionary -> Learner looks up."""
        # 1. System flags token as OOV candidate
        self.assertIsNone(self.knowledge_svc.lookup("ワンチャン"))
        cand = OovCandidate(
            token_id="tok-01",
            term="ワンチャン",
            tentative_reading="わんちゃん",
            suggested_pos="NOUN",
            suggested_meaning="có cơ hội, có thể",
            context_snippet="ワンチャン行けるかもしれない。",
            confidence_score=0.88,
            status=OovStatus.PENDING_CURATOR_REVIEW
        )
        saved = self.oov_svc.persist_candidate(cand)
        self.assertIsNotNone(saved.oov_candidate_id)

        # 2. Curator reviews suggestions and decides ADAPT with edits
        success, msg, review = self.oov_svc.process_review(
            candidate_id=saved.oov_candidate_id,
            curator_id="curator-linguist-01",
            decision=CuratorDecision.ADAPT,
            edited_term="ワンチャン",
            edited_reading="ワンチャン",
            edited_pos="SLANG",
            edited_meaning="Có cơ hội, có khả năng (tiếng lóng giới trẻ Nhật)",
            comment="Đã chuẩn hóa định nghĩa tiếng lóng"
        )
        self.assertTrue(success)
        self.assertEqual(review.decision, CuratorDecision.ADAPT)

        # Verify status changed to ADAPTED
        updated_cand = self.oov_svc.get_candidate(saved.oov_candidate_id)
        self.assertEqual(updated_cand.status, OovStatus.ADAPTED)

        # 3. Learner looks up for new learning items -> Immediately found in system knowledge!
        found = self.knowledge_svc.lookup("ワンチャン")
        self.assertIsNotNone(found)
        self.assertEqual(found.term, "ワンチャン")
        self.assertIn("tiếng lóng", found.meaning)

    def test_04_flow_4_reject_branch_without_manual_add(self):
        """Curator marks candidate as AI incorrect identification and rejects without manual add."""
        cand = OovCandidate(
            token_id="tok-02",
            term="へんてこり",
            tentative_reading="へんてこり",
            suggested_pos="NOUN",
            suggested_meaning="kỳ lạ",
            confidence_score=0.45
        )
        saved = self.oov_svc.persist_candidate(cand)

        success, msg, review = self.oov_svc.process_review(
            candidate_id=saved.oov_candidate_id,
            curator_id="curator-02",
            decision=CuratorDecision.REJECT,
            comment="Nhận diện sai âm thanh (hallucination)"
        )
        self.assertTrue(success)
        self.assertEqual(review.decision, CuratorDecision.REJECT)

        updated_cand = self.oov_svc.get_candidate(saved.oov_candidate_id)
        self.assertEqual(updated_cand.status, OovStatus.REJECTED)

        # Must NOT be merged into system knowledge
        self.assertIsNone(self.knowledge_svc.lookup("へんてこり"))

    def test_05_flow_4_reject_with_manual_add(self):
        """Curator rejects AI suggestion, but decides to add the token info manually."""
        cand = OovCandidate(
            token_id="tok-03",
            term="チルする",
            tentative_reading="ちるする",
            suggested_pos="NOUN",
            suggested_meaning="nghĩa không rõ"
        )
        saved = self.oov_svc.persist_candidate(cand)

        # Curator manually inputs definition, POS tag, and reading
        success, msg, review = self.oov_svc.process_manual_add(
            candidate_id=saved.oov_candidate_id,
            curator_id="curator-03",
            term="チルする",
            reading="ちるする",
            pos="VERB",
            meaning="Thư giãn, xả hơi (từ tiếng Anh chill)",
            comment="Bổ sung động từ tiếng lóng chính xác"
        )
        self.assertTrue(success)
        self.assertEqual(review.decision, CuratorDecision.MANUAL_ADD)

        updated_cand = self.oov_svc.get_candidate(saved.oov_candidate_id)
        self.assertEqual(updated_cand.status, OovStatus.RESOLVED_MANUAL)

        # Learner looks up -> Found!
        entry = self.knowledge_svc.lookup("チルする")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.pos, "VERB")
        self.assertIn("Thư giãn", entry.meaning)

    def test_06_internal_dictionary_api(self):
        """Tests the internal dictionary sync and lookup endpoints."""
        # 1. Dictionary lookup endpoint for existing term
        res_dict = self.client.get("/api/v1/dictionary/lookup?term=日本")
        self.assertEqual(res_dict.status_code, 200)
        d_data = res_dict.json()
        self.assertTrue(d_data["found"])
        self.assertEqual(d_data["entry"]["term"], "日本")

        # 2. Sync new word from BE Service
        sync_payload = {
            "term": "推し活",
            "reading": "おしかつ",
            "pos": "NOUN",
            "meaning": "Hoạt động ủng hộ thần tượng (oshi)",
            "definition_tags": "curator-verified"
        }
        sync_res = self.client.post("/api/v1/internal/dictionary/sync", json=sync_payload)
        self.assertEqual(sync_res.status_code, 200)
        self.assertTrue(sync_res.json()["success"])

        # 3. Lookup newly synced word
        res_synced = self.client.get("/api/v1/dictionary/lookup?term=推し活")
        self.assertEqual(res_synced.status_code, 200)
        self.assertTrue(res_synced.json()["found"])
        self.assertEqual(res_synced.json()["entry"]["term"], "推し活")

        # 4. Sync OOV candidate status directly (e.g. Reject or manual)
        status_payload = {"term": "推し活", "status": "REJECTED"}
        status_res = self.client.post("/api/v1/internal/oov/sync-status", json=status_payload)
        self.assertEqual(status_res.status_code, 200)
        self.assertTrue(status_res.json()["success"])


if __name__ == "__main__":
    unittest.main()
