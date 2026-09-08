"""Unit test suite for YouTubeService URL parsing, MP4 video download, and pipeline routing."""
import unittest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Adjust search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import YouTubeService, SubtitleSegment, PipelineResult


class TestYouTubeService(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.yt_svc = YouTubeService(download_dir=self.tmp_dir)

    def test_01_valid_youtube_urls(self):
        """Tests that various valid YouTube URL formats are correctly recognized."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
        ]
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertTrue(self.yt_svc.is_youtube_url(url), f"Failed for {url}")

    def test_02_invalid_youtube_urls(self):
        """Tests that local file paths and non-YouTube URLs are rejected."""
        invalid_inputs = [
            "samples/japanese_podcast_10s.mp4",
            "C:/files/podcast.wav",
            "/var/media/audio.mp3",
            "https://vimeo.com/12345678",
            "https://example.com/video.mp4",
            "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
            "",
            "   ",
            None,
            12345,
        ]
        for item in invalid_inputs:
            with self.subTest(item=item):
                self.assertFalse(self.yt_svc.is_youtube_url(item), f"Should fail for {item}")

    def test_03_extract_video_id(self):
        """Tests extracting the 11-character video ID across different URL formats."""
        url_map = {
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ": "dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/1A2B3C4D5E6": "1A2B3C4D5E6",
            "https://www.youtube.com/embed/zZ_X-y12345": "zZ_X-y12345",
        }
        for url, expected_id in url_map.items():
            self.assertEqual(self.yt_svc.extract_video_id(url), expected_id)

        self.assertIsNone(self.yt_svc.extract_video_id("samples/video.mp4"))

    @patch("yt_dlp.YoutubeDL")
    def test_04_get_metadata_mocked(self, mock_ydl_cls):
        """Tests get_metadata extracts key attributes from yt-dlp info dictionary."""
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "id": "dQw4w9WgXcQ",
            "title": "Japanese Podcast Episode 1",
            "uploader": "Rakushu Sensei",
            "duration": 60.5,
            "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
            "view_count": 1000,
        }

        meta = self.yt_svc.get_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(meta["video_id"], "dQw4w9WgXcQ")
        self.assertEqual(meta["title"], "Japanese Podcast Episode 1")
        self.assertEqual(meta["uploader"], "Rakushu Sensei")
        self.assertEqual(meta["duration"], 60.5)

    @patch("yt_dlp.YoutubeDL")
    def test_05_download_video_mp4(self, mock_ydl_cls):
        """Tests download_video saves directly as an MP4 file."""
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "id": "test_vid_01",
            "title": "Test Episode",
            "duration": 10.0,
        }

        dummy_mp4 = os.path.join(self.tmp_dir, "test_vid_01.mp4")
        with open(dummy_mp4, "wb") as f:
            f.write(b"dummy mp4 video bytes")

        file_path, meta = self.yt_svc.download_video(
            "https://www.youtube.com/watch?v=test_vid_01", max_height=720
        )

        self.assertEqual(meta["video_id"], "test_vid_01")
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.endswith(".mp4"))
    @patch("src.YouTubeService.download_video")
    @patch("src.AsrService.transcribe")
    @patch("src.LlmEnrichmentService.translate_full_transcription")
    @patch("src.LlmEnrichmentService.translate_sentence_with_context")
    def test_06_pipeline_youtube_routing(
        self, mock_s_trans, mock_f_trans, mock_asr, mock_yt_download
    ):
        """Tests that run_pipeline detects YouTube URL, downloads MP4, and populates video_path."""
        import importlib
        runner_mod = importlib.import_module("src.pipeline-runner")

        sample_url = "https://www.youtube.com/watch?v=abcdefghijk"
        mock_yt_download.return_value = (
            "downloads/video.mp4",
            {"video_id": "abcdefghijk", "title": "Podcast Title", "duration": 10.0},
        )
        mock_asr.return_value = SubtitleSegment(
            segment_id="seg_001",
            text="皆さん、こんにちは。",
            start_time=0.0,
            end_time=5.0,
        )
        mock_f_trans.return_value = "Xin chào mọi người."
        mock_s_trans.return_value = (
            "Xin chào mọi người.",
            {"皆さん": "mọi người", "こんにちは": "xin chào"},
            [],
        )

        result = runner_mod.run_pipeline(sample_url)

        mock_yt_download.assert_called_once()
        self.assertEqual(result.segment.video_id, "abcdefghijk")
        self.assertEqual(result.segment.video_title, "Podcast Title")
        self.assertEqual(result.segment.source_url, sample_url)
        self.assertEqual(result.segment.video_path, "downloads/video.mp4")


if __name__ == "__main__":
    unittest.main()
