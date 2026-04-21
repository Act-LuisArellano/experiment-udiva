"""
Video reader: extracts metadata and frames from video files using ffprobe/ffmpeg.

Adapted from the existing HHOI dataloader but simplified for the
experiment framework's needs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.core.schemas import VideoSample


def get_video_info(path: str) -> VideoSample:
    """Load video metadata via ffprobe.

    Returns a VideoSample with duration, fps, dimensions, and audio flag.
    Raises FileNotFoundError if the video doesn't exist.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    # Get stream info as JSON
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate,codec_name",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:300]}")

    data = json.loads(result.stdout)

    # Parse duration
    duration = float(data.get("format", {}).get("duration", 0))

    # Parse streams
    width = 0
    height = 0
    fps = 0.0
    has_audio = False

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type", "")
        if codec_type == "video" or (codec_type == "" and "width" in stream):
            width = stream.get("width", width)
            height = stream.get("height", height)
            # Parse frame rate (e.g., "5/1" or "30000/1001")
            fps_str = stream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 0.0
            else:
                fps = float(fps_str) if fps_str else 0.0
        if codec_type == "audio" or stream.get("codec_name", "") in ("aac", "mp3", "opus", "vorbis", "pcm_s16le"):
            has_audio = True

    return VideoSample(
        path=video_path,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
    )


def extract_frames(
    video_path: Path,
    start: float,
    duration: float,
    fps: float = 2.0,
    resize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Extract frames from a video segment as a numpy array.

    Returns array of shape (T, H, W, 3) in uint8 RGB.
    Uses ffmpeg with pipe output for efficiency.
    """
    vf_parts = [f"fps={fps}"]
    if resize is not None:
        h, w = resize
        vf_parts.append(f"scale={w}:{h}")
    vf_str = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vf", vf_str,
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed: {result.stderr.decode()[-300:]}"
        )

    raw = result.stdout
    if not raw:
        # No frames extracted (e.g., seek past end)
        if resize:
            h, w = resize
        else:
            # Try to get dimensions from video info
            h, w = 720, 1280  # default fallback
        return np.zeros((0, h, w, 3), dtype=np.uint8)

    # Determine frame dimensions
    if resize is not None:
        h, w = resize
    else:
        # Get dimensions from video
        info = get_video_info(str(video_path))
        h, w = info.height, info.width

    frame_bytes = h * w * 3
    n_frames = len(raw) // frame_bytes
    if n_frames == 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8)

    frames = np.frombuffer(raw[:n_frames * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(n_frames, h, w, 3)
    return frames


def frames_to_tensor(frames_np: np.ndarray) -> torch.Tensor:
    """Convert numpy frames (T, H, W, C) uint8 → torch (T, C, H, W) float32 [0, 1]."""
    if frames_np.shape[0] == 0:
        return torch.empty(0, dtype=torch.float32)
    tensor = torch.from_numpy(frames_np.copy()).permute(0, 3, 1, 2).float() / 255.0
    return tensor


def extract_audio(
    video_path: Path,
    start: float,
    duration: float,
    sr: int = 16000,
) -> torch.Tensor:
    """Extract audio from a video segment as a torch tensor.

    Uses ffmpeg to decode audio to raw PCM and converts to a normalized
    float32 tensor in the [-1, 1] range.

    Args:
        video_path: Path to the video file.
        start: Start time in seconds.
        duration: Duration in seconds.
        sr: Target sample rate in Hz (default 16000).

    Returns:
        1D torch.Tensor of shape (samples,) in float32 [-1, 1].
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 16-bit signed little-endian PCM
        "-ar", str(sr),         # target sample rate
        "-ac", "1",             # mono
        "-f", "s16le",          # raw PCM format
        "pipe:1",
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed: {result.stderr.decode()[-300:]}"
        )

    raw = result.stdout
    if not raw:
        return torch.zeros(0, dtype=torch.float32)

    # Convert raw int16 PCM bytes → normalized float32 tensor
    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio_np.copy())

