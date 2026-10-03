"""Integration tests for Video Import & Processing Pipeline API endpoints."""
import os
import sys
import unittest
import importlib
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

api_server = importlib.import_module("src.api.api-server")
app = api_server.app


class TestPipelineApi(unittest.TestCase):
    """Test suite for video pipeline API endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_media = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "samples", "japanese_podcast_10s.mp4"
            )
        )

    def test_01_health_endpoints(self):
        """Verifies root and pipeline health check endpoints."""
        res_root = self.client.get("/health")
        self.assertEqual(res_root.status_code, 200)
        self.assertEqual(res_root.json()["status"], "UP")

        res_pipe = self.client.get("/api/v1/pipeline/health")
        self.assertEqual(res_pipe.status_code, 200)
        self.assertEqual(res_pipe.json()["status"], "UP")
        self.assertIn("youtube_url", res_pipe.json()["supported_inputs"])

    def test_02_process_url_validation_error(self):
        """Verifies 422 on missing or too short URL parameter."""
        res = self.client.post("/api/v1/pipeline/process-url", json={"url": "abc"})
        self.assertEqual(res.status_code, 422)

    def test_03_upload_video_unsupported_format(self):
        """Verifies 400 Bad Request when uploading non-media file format."""
        files = {"file": ("document.pdf", b"fake pdf content", "application/pdf")}
        res = self.client.post("/api/v1/pipeline/upload-video", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file format", res.json()["detail"])

    def test_04_process_media_path(self):
        """Verifies full end-to-end processing via the SSE pipeline endpoint."""
        if not os.path.exists(self.sample_media):
            raise unittest.SkipTest(f"Sample media not found: {self.sample_media}")

        payload = {"url": self.sample_media}
        res = self.client.post("/api/v1/pipeline/process-url", json=payload)

        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers["content-type"])
        self.assertIn("event: sentence_processed", res.text)
        self.assertIn("event: chunk_completed", res.text)
        self.assertIn("event: pipeline_completed", res.text)
        self.assertIn('"translation"', res.text)

    def test_05_upload_valid_video_file(self):
        """Verifies multipart file upload processing and temp file cleanup."""
        if not os.path.exists(self.sample_media):
            raise unittest.SkipTest(f"Sample media not found: {self.sample_media}")

        with open(self.sample_media, "rb") as f:
            files = {"file": ("test_upload.mp4", f, "video/mp4")}
            res = self.client.post("/api/v1/pipeline/upload-video", files=files)

        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers["content-type"])
        self.assertIn("event: sentence_processed", res.text)
        self.assertIn("event: chunk_completed", res.text)
        self.assertIn("event: pipeline_completed", res.text)
        self.assertIn('"translation"', res.text)


if __name__ == "__main__":
    unittest.main()
