"""Focused tests for post-pipeline NLP output APIs."""
import os
import sys
import unittest

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.api_server import app
from src.models import SubtitleSegment
from src.services.nlp_service import NlpService
from src.services.pipeline_output_store import pipeline_output_store


class TestNlpOutputApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        pipeline_output_store._outputs.clear()

    def test_nlp_service_returns_ginza_json_fields(self):
        output = NlpService().analyze_json(SubtitleSegment(text="日本語を勉強します。"))
        self.assertEqual(output["format"], "ginza-json")
        ginza_json = output["ginza_json"]
        self.assertIn("paragraphs", ginza_json)
        token = ginza_json["paragraphs"][0]["sentences"][0]["tokens"][0]
        # These are the fields emitted by GiNZA's own `format_json` / `ginza -f json`.
        for field in ("id", "orth", "tag", "pos", "lemma", "head", "dep", "ner"):
            self.assertIn(field, token)

    def test_tokens_returns_stored_ginza_json(self):
        pipeline_output_store.save("video-1", [{
            "sentence_number": 1,
            "text": "日本語を勉強します。",
            "nlp": {
                "format": "ginza-json",
                "text": "日本語を勉強します。",
                "ginza_json": {
                    "paragraphs": [{
                        "raw": "日本語を勉強します。",
                        "sentences": [{
                            "tokens": [{
                                "id": 1,
                                "orth": "日本語",
                                "tag": "名詞",
                                "pos": "NOUN",
                                "lemma": "日本語",
                                "head": 4,
                                "dep": "obj",
                                "ner": "O",
                            }]
                        }]
                    }]
                },
                "ginza_full": {
                    "json": {
                        "paragraphs": [{
                            "raw": "日本語を勉強します。",
                            "sentences": [{"tokens": [{"id": 1, "orth": "日本語"}]}]
                        }]
                    },
                    "tokens": [{"id": 1, "orth": "日本語", "reading": "ニホンゴ"}]
                },
                "tokens": [{
                    "id": 1,
                    "orth": "日本語",
                    "tag": "名詞",
                    "pos": "NOUN",
                    "lemma": "日本語",
                    "head_absolute": 4,
                    "dep": "obj",
                    "bunsetu_bi_label": "B",
                    "bunsetu_position_type": "SEM_HEAD",
                    "clause_head": 4,
                }],
            },
        }])

        response = self.client.get("/api/v1/nlp/videos/video-1/tokens")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["format"], "ginza-json")
        self.assertEqual(body["ginza"][0]["json"]["paragraphs"][0]["raw"], "日本語を勉強します。")
        self.assertEqual(body["ginza"][0]["tokens"][0]["reading"], "ニホンゴ")
        self.assertEqual(body["tokens"][0]["orth"], "日本語")
        self.assertNotIn("bunsetu_bi_label", body["tokens"][0])

    def test_bunsetsu_builds_dependency_relations_from_stored_tokens(self):
        tokens = [
            {
                "id": 1, "orth": "日本語", "lemma": "日本語", "pos": "NOUN",
                "tag": "名詞", "head_absolute": 4, "dep": "obj",
                "bunsetu_bi_label": "B", "bunsetu_position_type": "SEM_HEAD",
                "clause_head": 4,
            },
            {
                "id": 2, "orth": "を", "lemma": "を", "pos": "ADP",
                "tag": "助詞", "head_absolute": 1, "dep": "case",
                "bunsetu_bi_label": "I", "bunsetu_position_type": "SYN_HEAD",
                "clause_head": 4,
            },
            {
                "id": 3, "orth": "勉強", "lemma": "勉強", "pos": "NOUN",
                "tag": "名詞", "head_absolute": 4, "dep": "obl",
                "bunsetu_bi_label": "B", "bunsetu_position_type": "SEM_HEAD",
                "clause_head": 4,
            },
            {
                "id": 4, "orth": "します", "lemma": "する", "pos": "VERB",
                "tag": "動詞", "head_absolute": 0, "dep": "ROOT",
                "bunsetu_bi_label": "B", "bunsetu_position_type": "ROOT",
                "clause_head": 4,
            },
        ]
        pipeline_output_store.save("video-2", [{
            "sentence_number": 1,
            "text": "日本語を勉強します",
            "nlp": {"format": "ginza-json", "text": "日本語を勉強します", "tokens": tokens},
        }])

        response = self.client.get("/api/v1/nlp/videos/video-2/bunsetsu")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([b["text"] for b in body["bunsetsu"]], ["日本語を", "勉強", "します"])
        self.assertEqual(
            [(r["from_bunsetsu"], r["to_bunsetsu"]) for r in body["relations"]],
            [(1, 3), (2, 3)],
        )
        self.assertEqual(body["bunsetsu"][0]["dep"]["rel"], "obj")
        self.assertEqual(body["bunsetsu"][2]["dep"]["to"], 0)
        self.assertEqual(body["bunsetsu"][2]["children"], [1, 2])
        self.assertEqual(body["roots"], [3])
        self.assertEqual(body["sentences"][0]["roots"], [3])
        self.assertNotIn(
            3,
            [relation["to_bunsetsu"] for relation in body["relations"]
             if relation["from_bunsetsu"] == 3],
        )

    def test_multi_sentence_bunsetsu_references_use_global_ids(self):
        sentence_one = [
            {
                "id": 1, "orth": "私", "lemma": "私", "pos": "PRON",
                "head_absolute": 2, "dep": "nsubj",
                "bunsetu_bi_label": "B", "bunsetu_position_type": "SEM_HEAD",
                "clause_head": 2,
            },
            {
                "id": 2, "orth": "行く", "lemma": "行く", "pos": "VERB",
                "head_absolute": 2, "dep": "root",
                "bunsetu_bi_label": "B", "bunsetu_position_type": "ROOT",
                "clause_head": 2,
            },
        ]
        sentence_two = [
            {
                "id": 1, "orth": "学校", "lemma": "学校", "pos": "NOUN",
                "head_absolute": 2, "dep": "obl",
                "bunsetu_bi_label": "B", "bunsetu_position_type": "SEM_HEAD",
                "clause_head": 2,
            },
            {
                "id": 2, "orth": "着く", "lemma": "着く", "pos": "VERB",
                "head_absolute": 2, "dep": "root",
                "bunsetu_bi_label": "B", "bunsetu_position_type": "ROOT",
                "clause_head": 2,
            },
        ]
        pipeline_output_store.save("video-3", [
            {"sentence_number": 1, "text": "私行く", "nlp": {"tokens": sentence_one}},
            {"sentence_number": 2, "text": "学校着く", "nlp": {"tokens": sentence_two}},
        ])

        response = self.client.get("/api/v1/nlp/videos/video-3/bunsetsu")
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual([item["id"] for item in body["bunsetsu"]], [1, 2, 3, 4])
        self.assertEqual(body["roots"], [2, 4])
        self.assertEqual(body["bunsetsu"][0]["dep"]["to"], 2)
        self.assertEqual(body["bunsetsu"][2]["dep"]["to"], 4)
        self.assertEqual(body["bunsetsu"][1]["dep"]["to"], 0)
        self.assertEqual(body["bunsetsu"][3]["dep"]["to"], 0)
        self.assertEqual(body["bunsetsu"][1]["children"], [1])
        self.assertEqual(body["bunsetsu"][3]["children"], [3])
        self.assertEqual(body["sentences"][0]["roots"], [2])
        self.assertEqual(body["sentences"][1]["roots"], [4])

        ids = {item["id"] for item in body["bunsetsu"]}
        for bunsetsu in body["bunsetsu"]:
            dep_to = bunsetsu["dep"]["to"]
            if dep_to:
                self.assertIn(dep_to, ids)
            for child in bunsetsu["children"]:
                self.assertIn(child, ids)
            for case_frame in bunsetsu["case_frame"]:
                self.assertIn(case_frame["bunsetsu"], ids)

        for relation in body["relations"]:
            self.assertIn(relation["from_bunsetsu"], ids)
            self.assertIn(relation["to_bunsetsu"], ids)
            self.assertNotEqual(
                relation["from_bunsetsu"], relation["to_bunsetsu"]
            )

    def test_unknown_video_returns_not_found(self):
        response = self.client.get("/api/v1/nlp/videos/missing/tokens")
        self.assertEqual(response.status_code, 404)

    def test_post_tokens_analyzes_transcription_with_get_contract(self):
        response = self.client.post(
            "/api/v1/nlp/tokens",
            json={"transcription": "日本語を勉強します。"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIsNone(body["video_id"])
        self.assertIn(body["format"], {"ginza-json", "rakushu-ginza-json-fallback"})
        self.assertIsInstance(body["ginza"], list)
        self.assertTrue(body["tokens"])
        for field in ("id", "orth", "tag", "pos", "lemma", "head", "dep", "ner"):
            self.assertIn(field, body["tokens"][0])

    def test_post_tokens_flattens_multiple_sentences(self):
        response = self.client.post(
            "/api/v1/nlp/tokens",
            json={"transcription": "私は学生です。日本語を勉強します。"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertGreaterEqual(len(body["ginza"]), 2)
        token_surfaces = [token["orth"] for token in body["tokens"]]
        self.assertTrue(any("私" in surface for surface in token_surfaces))
        self.assertTrue(any("日本語" in surface for surface in token_surfaces))

    def test_post_tokens_rejects_blank_transcription(self):
        response = self.client.post(
            "/api/v1/nlp/tokens",
            json={"transcription": "   "},
        )
        self.assertEqual(response.status_code, 422)

    def test_post_bunsetsu_accepts_raw_token_array(self):
        tokens = [
            {"id": 1, "orth": "日本語", "lemma": "日本語", "pos": "NOUN", "head_absolute": 4, "dep": "obj", "bunsetu_bi_label": "B", "bunsetu_position_type": "SEM_HEAD", "clause_head": 4},
            {"id": 2, "orth": "を", "lemma": "を", "pos": "ADP", "head_absolute": 1, "dep": "case", "bunsetu_bi_label": "I", "bunsetu_position_type": "SYN_HEAD", "clause_head": 4},
            {"id": 3, "orth": "勉強", "lemma": "勉強", "pos": "NOUN", "head_absolute": 4, "dep": "obl", "bunsetu_bi_label": "B", "bunsetu_position_type": "SEM_HEAD", "clause_head": 4},
            {"id": 4, "orth": "します", "lemma": "する", "pos": "VERB", "head_absolute": 0, "dep": "ROOT", "bunsetu_bi_label": "B", "bunsetu_position_type": "ROOT", "clause_head": 4},
        ]
        response = self.client.post("/api/v1/nlp/bunsetsu", json=tokens)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIsNone(body["video_id"])
        self.assertEqual([item["text"] for item in body["bunsetsu"]], ["日本語を", "勉強", "します"])
        self.assertEqual(
            [(item["from_bunsetsu"], item["to_bunsetsu"]) for item in body["relations"]],
            [(1, 3), (2, 3)],
        )
        self.assertEqual(body["roots"], [3])
        self.assertEqual(body["sentences"][0]["text"], "日本語を勉強します")

    def test_post_bunsetsu_rejects_missing_token_id(self):
        response = self.client.post("/api/v1/nlp/bunsetsu", json=[{"orth": "日本語", "head_absolute": 0, "dep": "ROOT"}])
        self.assertEqual(response.status_code, 422)

    def test_post_bunsetsu_rejects_duplicate_token_ids(self):
        response = self.client.post("/api/v1/nlp/bunsetsu", json=[{"id": 1, "orth": "日本語"}, {"id": 1, "orth": "を"}])
        self.assertEqual(response.status_code, 422)

    def test_post_bunsetsu_rejects_native_tokens_without_graph_metadata(self):
        response = self.client.post(
            "/api/v1/nlp/bunsetsu",
            json=[{"id": 1, "orth": "日本語", "head": 0, "dep": "ROOT"}],
        )
        self.assertEqual(response.status_code, 422)

    def test_post_bunsetsu_empty_array_returns_empty_graph(self):
        response = self.client.post("/api/v1/nlp/bunsetsu", json=[])
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["bunsetsu"], [])
        self.assertEqual(body["relations"], [])
        self.assertEqual(body["roots"], [])
        self.assertEqual(body["sentences"], [])

    def test_legacy_text_payload_is_rejected(self):
        response = self.client.post(
            "/api/v1/nlp/tokens",
            json={"text": "日本語を勉強します。"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
