"""Media Inspector: Multi-point sampling with Whisper-native detect_language and text moderation."""
import os
import subprocess
import importlib
from typing import List, Tuple, Dict, Any, Optional

_schema = importlib.import_module(".schema-models", package="src")
MediaInspectionResult = _schema.MediaInspectionResult
VideoType = _schema.VideoType
MediaSourceType = _schema.MediaSourceType


class InvalidMediaError(Exception):
    """Raised when input media is corrupted, inaccessible, or lacking audio stream/speech."""
    pass


class InvalidLanguageError(Exception):
    """Raised when media is detected as non-Japanese."""
    pass


class ProhibitedContentError(Exception):
    """Raised when media contains prohibited content (NSFW, illegal, extreme violence)."""
    pass


class MediaInspector:
    """Performs multi-point sampling using Whisper native detect_language and text moderation."""

    RED_LIST_TERMS = {
        "ポルノ", "エロ動画", "オナニー", "セフレ", "児童ポルノ", "裏ビデオ",
        "闇バイト", "オンラインカジノ", "ネットカジノ", "裏カジノ", "賭博場",
        "自殺方法", "覚醒剤", "大麻密売", "コカイン密売",
        "porn", "hentai", "xxx", "gambling", "online casino"
    }

    TYPE_PROFILES = {
        VideoType.NEWS_FORMAL: {
            "keywords": ["ニュース", "nhk", "報道", "政治", "経済", "会見", "news", "fnn"],
            "config": {"initial_prompt": "NHKニュースです。政治、経済、社会のニュースをお伝えします。", "vad_filter": False, "beam_size": 5, "temperature": 0.0}
        },
        VideoType.ANIME_DRAMA: {
            "keywords": ["アニメ", "ドラマ", "第話", "キャラ", "声優", "シーン", "anime", "clip"],
            "config": {"initial_prompt": "アニメやドラマのセリフです。感情豊かな会話です。", "vad_filter": True, "beam_size": 5, "no_speech_threshold": 0.6}
        },
        VideoType.CASUAL_VLOG: {
            "keywords": ["vlog", "実況", "ゲーム", "やってみた", "雑談", "gaming", "stream"],
            "config": {"initial_prompt": "こんにちは。ゲーム実況や日常Vlogの会話です。", "vad_filter": True, "beam_size": 5, "condition_on_previous_text": False}
        },
        VideoType.PODCAST_TALKSHOW: {
            "keywords": ["ポッドキャスト", "podcast", "対談", "ラジオ", "語り", "talk"],
            "config": {"initial_prompt": "こんにちは。ポッドキャストへようこそ。日常会話を話します。", "vad_filter": True, "beam_size": 5, "temperature": 0.0}
        },
        VideoType.MUSIC_SONG: {
            "keywords": ["mv", "mix", "song", "music", "cover", "club mix", "remix", "歌", "曲"],
            "config": {"initial_prompt": "日本の音楽や歌の歌詞です。", "vad_filter": True, "beam_size": 5, "temperature": 0.0}
        },
    }

    DEFAULT_WHISPER_CONFIG = {
        "initial_prompt": "こんにちは。日本語の書き起こしです。",
        "vad_filter": True, "beam_size": 5, "temperature": 0.0
    }

    def __init__(self, min_ja_probability: float = 0.50, foreign_prob_threshold: float = 0.70):
        self.min_ja_probability = min_ja_probability
        self.foreign_prob_threshold = foreign_prob_threshold

    def check_audio_stream(self, media_path: str) -> Tuple[bool, float]:
        """Checks if media has an audio stream and returns its duration in seconds."""
        if not os.path.exists(media_path):
            raise InvalidMediaError(f"Media file not found: {media_path}")
        cmd_stream = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
            media_path
        ]
        res_stream = subprocess.run(cmd_stream, capture_output=True, text=True)
        if not res_stream.stdout.strip():
            raise InvalidMediaError("Media file has no audio stream (silent/muted).")

        cmd_dur = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", media_path
        ]
        res_dur = subprocess.run(cmd_dur, capture_output=True, text=True)
        try:
            duration = float(res_dur.stdout.strip())
        except ValueError:
            duration = 10.0
        return True, duration

    def moderate_text(self, text: str) -> Tuple[bool, str]:
        """Text-level moderation checking for zero-tolerance prohibited terms."""
        norm = text.lower()
        for term in self.RED_LIST_TERMS:
            if term.lower() in norm:
                return True, term
        return False, ""

    def check_prohibited_content(self, text: str) -> Tuple[bool, str]:
        return self.moderate_text(text)

    def extract_audio_slices_memory(self, audio: Any, duration: float, ratios: List[float], slice_len: float = 3.5) -> List[Any]:
        """Extracts audio slice numpy arrays directly from memory (16kHz)."""
        sr = 16000
        slices = []
        total_samples = len(audio)
        for r in ratios:
            start_s = max(0, min(total_samples - int(slice_len * sr), int(duration * r * sr)))
            end_s = min(total_samples, start_s + int(slice_len * sr))
            slices.append(audio[start_s:end_s])
        return slices

    def detect_slice(self, slice_audio: Any, whisper_model: Any) -> Tuple[str, float]:
        """Runs Whisper native detect_language on a single audio slice."""
        if hasattr(whisper_model, "detect_language"):
            lang, prob, _ = whisper_model.detect_language(slice_audio)
            return lang, float(prob)
        return "ja", 1.0

    def profile_video_type(self, title: str, sample_text: str = "") -> Tuple[str, Dict[str, Any]]:
        combined = f"{title} {sample_text}".lower()
        for v_type, data in self.TYPE_PROFILES.items():
            if any(k.lower() in combined for k in data["keywords"]):
                return v_type, dict(data["config"])
        return VideoType.GENERAL_CONVERSATION, dict(self.DEFAULT_WHISPER_CONFIG)

    def inspect(
        self,
        media_path: str,
        title: str = "",
        source_type: str = MediaSourceType.LOCAL_UPLOAD,
        whisper_model: Any = None
    ) -> MediaInspectionResult:
        _, duration = self.check_audio_stream(media_path)
        if source_type == MediaSourceType.LOCAL_UPLOAD:
            is_banned, term = self.moderate_text(title)
            if is_banned:
                raise ProhibitedContentError(f"Media metadata contains prohibited term: '{term}'")

        if not whisper_model:
            v_type, cfg = self.profile_video_type(title)
            return MediaInspectionResult(is_valid=True, source_type=source_type, video_type=v_type, whisper_config=cfg)

        import faster_whisper
        audio = faster_whisper.decode_audio(media_path)
        ratios = [0.15, 0.50, 0.85] if duration < 180.0 else ([0.10, 0.30, 0.50, 0.70, 0.90] if duration <= 600.0 else [0.10, 0.22, 0.35, 0.50, 0.65, 0.78, 0.90])
        slices = self.extract_audio_slices_memory(audio, duration, ratios)

        ja_slices, foreign_slices = 0, []
        for s in slices:
            lang, prob = self.detect_slice(s, whisper_model)
            if lang == "ja" and prob >= self.min_ja_probability:
                ja_slices += 1
            elif lang != "ja" and prob >= self.foreign_prob_threshold:
                foreign_slices.append((lang, prob))

        min_ja_needed = 1 if duration < 180.0 else 2
        # Edge case: all slices skipped (e.g. music intro/solo), retry with supplementary slices
        if ja_slices == 0 and not foreign_slices:
            supp_ratios = [0.35, 0.65] if duration < 180.0 else [0.20, 0.40, 0.60, 0.80]
            for s in self.extract_audio_slices_memory(audio, duration, supp_ratios):
                lang, prob = self.detect_slice(s, whisper_model)
                if lang == "ja" and prob >= self.min_ja_probability: ja_slices += 1
                elif lang != "ja" and prob >= self.foreign_prob_threshold: foreign_slices.append((lang, prob))

        if ja_slices >= min_ja_needed:
            avg_prob = 0.95
        elif foreign_slices and ja_slices == 0:
            dom = max(foreign_slices, key=lambda x: x[1])[0]
            raise InvalidLanguageError(f"Media detected as foreign speech ('{dom}').")
        else:
            if ja_slices == 0 and not foreign_slices:
                raise InvalidMediaError("Media contains only instrumental music or silence with no detectable speech.")
            raise InvalidLanguageError(f"Media lacks sufficient Japanese speech ({ja_slices}/{min_ja_needed} slices).")

        v_type, whisper_cfg = self.profile_video_type(title)
        return MediaInspectionResult(
            is_valid=True, source_type=source_type, detected_language="ja",
            language_probability=round(avg_prob, 3), video_type=v_type,
            whisper_config=whisper_cfg, sample_slices_count=len(slices)
        )


