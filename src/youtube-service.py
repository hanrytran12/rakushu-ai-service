"""YouTube Ingestion Service: Parses YouTube URLs, extracts metadata, and downloads media."""
import os
import re
import tempfile
from typing import Dict, Any, Optional, Tuple


class YouTubeService:
    """Handles validation, metadata extraction, and media downloading from YouTube URLs."""

    URL_REGEX = re.compile(
        r"^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/|embed/)|youtu\.be/)(?P<id>[a-zA-Z0-9_-]{11})"
    )

    def __init__(self, download_dir: Optional[str] = None):
        self.download_dir = download_dir or os.path.join(tempfile.gettempdir(), "rakushu_youtube")
        os.makedirs(self.download_dir, exist_ok=True)

    def is_youtube_url(self, url_or_path: str) -> bool:
        """Checks whether the provided input string is a valid YouTube URL."""
        if not isinstance(url_or_path, str):
            return False
        return bool(self.URL_REGEX.match(url_or_path.strip()))

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extracts the canonical 11-character YouTube video ID."""
        m = self.URL_REGEX.match(url.strip())
        return m.group("id") if m else None

    def get_metadata(self, url: str) -> Dict[str, Any]:
        """Extracts video metadata without downloading full stream."""
        import yt_dlp
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "video_id": info.get("id") or self.extract_video_id(url) or "unknown",
                "title": info.get("title", ""),
                "uploader": info.get("uploader") or info.get("channel", ""),
                "duration": float(info.get("duration") or 0.0),
                "thumbnail_url": info.get("thumbnail", ""),
                "view_count": int(info.get("view_count") or 0),
            }

    def download_video(
        self, url: str, max_height: int = 720, custom_filename: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Downloads YouTube video as MP4 directly (capped at max_height for optimal performance)."""
        import yt_dlp
        meta = self.get_metadata(url)
        vid_id = meta["video_id"] or "youtube_video"
        base_name = custom_filename or vid_id
        out_template = os.path.join(self.download_dir, f"{base_name}.%(ext)s")
        final_path = os.path.join(self.download_dir, f"{base_name}.mp4")

        ydl_opts = {
            "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
            "outtmpl": out_template,
            "merge_output_format": "mp4",
            "retries": 10,
            "fragment_retries": 10,
            "continuedl": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(final_path):
            candidates = [
                os.path.join(self.download_dir, f) for f in os.listdir(self.download_dir)
                if f.startswith(base_name) and f.endswith((".mp4", ".mkv", ".webm"))
            ]
            if candidates:
                final_path = candidates[0]

        return final_path, meta

    def download_media(
        self, url: str, audio_only: bool = False, custom_filename: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Downloads media stream (MP4 by default, or 16kHz WAV if audio_only is True)."""
        if not audio_only:
            return self.download_video(url, custom_filename=custom_filename)

        import yt_dlp
        meta = self.get_metadata(url)
        vid_id = meta["video_id"] or "youtube_video"
        base_name = custom_filename or vid_id
        out_template = os.path.join(self.download_dir, f"{base_name}.%(ext)s")
        final_path = os.path.join(self.download_dir, f"{base_name}.wav")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }],
            "postprocessor_args": ["-ar", "16000", "-ac", "1"],
            "retries": 10,
            "fragment_retries": 10,
            "continuedl": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(final_path):
            candidates = [
                os.path.join(self.download_dir, f) for f in os.listdir(self.download_dir)
                if f.startswith(base_name)
            ]
            if candidates:
                final_path = candidates[0]

        return final_path, meta