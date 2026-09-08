"""Generates a 10-second Japanese podcast video sample with authentic Japanese speech."""
import os
import subprocess
from pathlib import Path


def create_sample_media(output_mp4: str = "samples/japanese_podcast_10s.mp4"):
    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    wav_path = samples_dir / "temp_speech.wav"
    ps_script_path = samples_dir / "synth_script.ps1"

    japanese_text = "皆さん、こんにちは。今日のポッドキャストでは、日本の面白い若者言葉について話します。"

    # PowerShell script to synthesize Japanese speech using native Haruka voice
    ps_code = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices()
$voiceName = ($voices | Where-Object {{ $_.VoiceInfo.Name -like "*Haruka*" -or $_.VoiceInfo.Name -like "*Ayumi*" }} | Select-Object -First 1).VoiceInfo.Name
if (-not $voiceName) {{
    $voiceName = ($voices | Select-Object -First 1).VoiceInfo.Name
}}
$synth.SelectVoice($voiceName)
$synth.SetOutputToWaveFile('{wav_path.as_posix()}')
$synth.Speak('{japanese_text}')
$synth.Dispose()
"""
    with open(ps_script_path, "w", encoding="utf-8-sig") as f:
        f.write(ps_code)

    print("[Media Generator] Synthesizing Japanese audio track...")
    subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps_script_path)],
        check=True
    )

    # Multiplex with FFmpeg: 10s video banner with background and audio
    print(f"[Media Generator] Encoding 10s MP4 video to {output_mp4}...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x1a1a2e:s=854x480:d=10",
        "-i", str(wav_path),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-t", "10",
        str(output_mp4)
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up temp script
    if ps_script_path.exists():
        ps_script_path.unlink()

    print(f"[Media Generator] Success! Generated 10s Japanese video: {output_mp4}")
    return output_mp4, japanese_text


if __name__ == "__main__":
    create_sample_media()
