"""Unit test suite for MediaInspector pre-flight validation and video type profiling."""
import os
import sys
import unittest
import tempfile
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    MediaInspector,
    VideoType,
    MediaInspectionResult,
    InvalidMediaError,
    InvalidLanguageError,
    ProhibitedContentError,
    AsrService,
)


class TestMediaInspector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_media = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples", "japanese_podcast_10s.mp4"))
        cls.inspector = MediaInspector()

    def test_01_audio_stream_and_duration_check(self):
        """Verifies that valid media stream is detected and duration is measured."""
        if not os.path.exists(self.sample_media):
            self.skipTest("Sample media not found.")
        has_audio, duration = self.inspector.check_audio_stream(self.sample_media)
        self.assertTrue(has_audio)
        self.assertAlmostEqual(duration, 10.0, delta=1.0)

    def test_02_missing_and_corrupt_media_handling(self):
        """Verifies that non-existent media raises InvalidMediaError."""
        with self.assertRaises(InvalidMediaError):
            self.inspector.check_audio_stream("non_existent_file.mp4")

    def test_03_prohibited_content_rejection(self):
        """Verifies that Red-list banned keywords are flagged."""
        is_banned, term = self.inspector.check_prohibited_content("最新のオンラインカジノ情報")
        self.assertTrue(is_banned)
        self.assertEqual(term, "オンラインカジノ")

        is_safe, _ = self.inspector.check_prohibited_content("今日のポッドキャストでは日本語を話します。")
        self.assertFalse(is_safe)

        # Title containing banned term should raise ProhibitedContentError
        with self.assertRaises(ProhibitedContentError):
            self.inspector.inspect(self.sample_media, title="違法オンラインカジノの紹介")

    def test_04_video_type_profiling(self):
        """Verifies dynamic classification and tuned Whisper configs by video category."""
        # 1. News profile
        v_type, cfg = self.inspector.profile_video_type("NHKニュース7 今日の報道")
        self.assertEqual(v_type, VideoType.NEWS_FORMAL)
        self.assertFalse(cfg["vad_filter"])
        self.assertIn("NHKニュース", cfg["initial_prompt"])

        # 2. Anime profile
        v_type, cfg = self.inspector.profile_video_type("人気アニメの感動シーン第1話")
        self.assertEqual(v_type, VideoType.ANIME_DRAMA)
        self.assertTrue(cfg["vad_filter"])
        self.assertEqual(cfg.get("no_speech_threshold"), 0.6)

        # 3. Vlog profile
        v_type, cfg = self.inspector.profile_video_type("東京散歩Vlogとゲーム実況")
        self.assertEqual(v_type, VideoType.CASUAL_VLOG)
        self.assertFalse(cfg.get("condition_on_previous_text", True))

        # 4. Podcast profile
        v_type, cfg = self.inspector.profile_video_type("日本語ポッドキャスト第10回")
        self.assertEqual(v_type, VideoType.PODCAST_TALKSHOW)
        self.assertTrue(cfg["vad_filter"])

    def test_05_full_inspection_on_japanese_sample(self):
        """Runs full end-to-end inspection on sample media with whisper model."""
        if not os.path.exists(self.sample_media):
            self.skipTest("Sample media not found.")

        asr = AsrService(model_size="tiny", auto_inspect=False)
        model = asr._get_whisper_model()
        res = self.inspector.inspect(
            self.sample_media, title="日本語ポッドキャスト 10s", whisper_model=model
        )
        self.assertTrue(res.is_valid)
        self.assertEqual(res.detected_language, "ja")
        self.assertGreaterEqual(res.language_probability, 0.70)
        self.assertEqual(res.video_type, VideoType.PODCAST_TALKSHOW)

    def test_06_source_based_moderation_distinction(self):
        """Verifies that YouTube source trusts platform moderation, while LOCAL_UPLOAD moderates metadata."""
        from src import MediaSourceType
        if not os.path.exists(self.sample_media):
            self.skipTest("Sample media not found.")

        # Local upload with prohibited title is blocked
        with self.assertRaises(ProhibitedContentError):
            self.inspector.inspect(self.sample_media, title="違法オンラインカジノ", source_type=MediaSourceType.LOCAL_UPLOAD)

        # YouTube URL source does not block at metadata stage (delegates to YouTube platform moderation)
        res = self.inspector.inspect(self.sample_media, title="Casino Royale Review", source_type=MediaSourceType.YOUTUBE_URL)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.source_type, MediaSourceType.YOUTUBE_URL)

    def test_07_native_language_detection_and_rejection(self):
        """Verifies that non-Japanese audio triggers InvalidLanguageError."""
        if not os.path.exists(self.sample_media):
            self.skipTest("Sample media not found.")

        class MockEnglishModel:
            def detect_language(self, audio):
                return "en", 0.98, [("en", 0.98), ("ja", 0.01)]

        with self.assertRaises(InvalidLanguageError):
            self.inspector.inspect(self.sample_media, title="English Media", whisper_model=MockEnglishModel())

    def test_08_instrumental_skip_and_supplementary_retry(self):
        """Verifies that instrumental slices (<0.50) are skipped and supplementary retry rescues valid JA."""
        if not os.path.exists(self.sample_media):
            self.skipTest("Sample media not found.")

        class MockMusicModel:
            def __init__(self):
                self.calls = 0

            def detect_language(self, audio):
                self.calls += 1
                # Round 1: low prob instrumental (prob 0.25) -> skipped
                # Round 2 (supplementary): detects Japanese (prob 0.95) -> passes!
                if self.calls <= 3:
                    return "en", 0.25, []
                return "ja", 0.95, []

        res = self.inspector.inspect(
            self.sample_media, title="Stay With Me Club Mix MV", whisper_model=MockMusicModel()
        )
        self.assertTrue(res.is_valid)
        self.assertEqual(res.detected_language, "ja")
        self.assertEqual(res.video_type, VideoType.MUSIC_SONG)


if __name__ == "__main__":
    unittest.main()



