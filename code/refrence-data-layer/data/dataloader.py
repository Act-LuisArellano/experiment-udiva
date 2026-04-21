"""
UDIVA-HHOI Dataset & DataLoader.

Provides chunk-level access to the HHOI dyadic interaction dataset for training
and evaluation of LLM, VLM, and JEPA models.

Each sample is a *time chunk* from a session video, containing:
  - video_path  : path to the mosaic .mp4
    - video       : optional encoded chunk video bytes (MP4)
  - frames      : decoded video frames as float32 tensor (T, C, H, W)
  - transcript  : speaker-labelled dialogue within the chunk window
  - words       : word-level entries with timing, speaker, and confidence
  - metadata    : session info (participants, language, lego set, etc.)
  - chunk_info  : start/end times, duration, chunk index

The dataset handles:
  - Automatic split resolution (train / val / test) from metadata CSVs
  - Configurable chunk duration and stride (overlap)
  - On-the-fly frame extraction via ffmpeg (no pre-chunking needed)
  - Optional video cropping (right-side mosaic or full frame)
  - Transcript + word alignment to chunk time windows
  - Collate function for batched loading

Usage:
    from dataloader import HHOIDataset, create_dataloader

    # Quick start — all defaults
    train_loader = create_dataloader("train", chunk_duration=10, chunk_stride=5)
    for batch in train_loader:
        print(batch["frames"].shape)         # (B, T, C, H, W)
        print(batch["transcript"])            # list of transcript strings
        break

    # Full control
    dataset = HHOIDataset(
        split="train",
        data_root=".",
        chunk_duration=10,
        chunk_stride=5,
        fps=5,                # sample 5 frames/sec (instead of native 25)
        crop_right=True,      # crop to right 1280×720
        resize=(224, 224),    # resize frames
        load_video=True,      # include MP4 bytes for each chunk
    )
    sample = dataset[0]
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """All tuneable knobs for the dataset."""
    data_root: str = "."

    # Directories (relative to data_root)
    mosaics_dir: str = "mosaics"
    whisperx_dir: str = "whisperx_corrected"
    transcripts_dir: str = "transcriptions_filtered"
    metadata_dir: str = "metadata"
    annotations_dir: str = "annotations"

    # Chunking
    chunk_duration: float = 10.0   # seconds
    chunk_stride: float = 5.0     # seconds (overlap = duration - stride)

    # Video decoding
    fps: float = 5.0              # frames to sample per second
    crop_right: bool = False      # crop 1280×720 right side of 1920×720 mosaic
    resize: tuple[int, int] | None = None  # (H, W) to resize frames to

    # Filtering
    min_transcript_words: int = 0  # skip chunks with fewer words


# ─────────────────────────────────────────────────────────────────────────────
# SRT / JSON parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

_SRT_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    r"\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def _ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_srt_entries(path: Path) -> list[dict]:
    """Parse an SRT file into list of {start, end, text} dicts (times in seconds)."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    entries = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        match = _SRT_TS_RE.search(lines[1])
        if not match:
            continue
        g = match.groups()
        entries.append({
            "start": _ts_to_sec(*g[:4]),
            "end": _ts_to_sec(*g[4:]),
            "text": "\n".join(lines[2:]).strip(),
        })
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Metadata loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_sessions_csv(csv_path: Path) -> dict[str, dict]:
    """Load sessions CSV into {session_id: {column: value}} mapping."""
    sessions = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("ID", "")).strip().zfill(6)
            sessions[sid] = dict(row)
    return sessions


def _load_task_limits(json_path: Path) -> dict[str, dict]:
    """Load task_limits.json → {session_id: {task: [[start, end], ...]}}."""
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # Normalise session IDs to 6-digit zero-padded strings
    return {str(k).zfill(6): v for k, v in data.items()}


def load_split_session_ids(metadata_dir: Path) -> dict[str, list[str]]:
    """Return {split: [session_ids]} by reading all sessions CSVs."""
    splits: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        csv_path = metadata_dir / split / f"sessions_{split}.csv"
        if csv_path.exists():
            sessions = _load_sessions_csv(csv_path)
            splits[split] = list(sessions.keys())
    return splits


def load_annotation_splits(annotations_dir: Path) -> dict[str, list[str]]:
    """Determine train/test splits from annotation file naming.

    Files whose stem ends with ``_sp`` are **training**; the rest are **test**.
    Returns ``{split: [session_ids]}``.
    """
    splits: dict[str, list[str]] = {"train": [], "test": []}
    if not annotations_dir.exists():
        return splits
    seen: set[str] = set()
    for f in sorted(annotations_dir.glob("*_L_mosaic*.json")):
        sid = f.name.split("_")[0]
        if sid in seen:
            continue
        seen.add(sid)
        if f.stem.endswith("_sp"):
            splits["train"].append(sid)
        else:
            splits["test"].append(sid)
    return splits


def load_all_metadata(metadata_dir: Path) -> dict[str, dict]:
    """Merge session metadata + task limits across all splits.

    Returns {session_id: {split, session_meta, task_limits}}.
    """
    all_meta: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        csv_path = metadata_dir / split / f"sessions_{split}.csv"
        tl_path = metadata_dir / split / "task_limits.json"
        if not csv_path.exists():
            continue
        sessions = _load_sessions_csv(csv_path)
        task_limits = _load_task_limits(tl_path)
        for sid, meta in sessions.items():
            all_meta[sid] = {
                "split": split,
                "session_meta": meta,
                "task_limits": task_limits.get(sid, {}),
            }
    return all_meta


# ─────────────────────────────────────────────────────────────────────────────
# Video frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frames(
    video_path: Path,
    start: float,
    duration: float,
    fps: float = 5.0,
    crop_right: bool = False,
    resize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Extract frames from a video segment as a numpy array.

    Returns array of shape (T, H, W, 3) in uint8 RGB.
    Uses ffmpeg with pipe output for efficiency (no temp files).
    """
    # Build filter chain
    vf_parts = []
    if crop_right:
        # Crop right 1280×720 from 1920×720 mosaic
        vf_parts.append("crop=1280:720:640:0")
    vf_parts.append(f"fps={fps}")
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
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)

    # Determine frame dimensions
    if resize is not None:
        h, w = resize
    elif crop_right:
        h, w = 720, 1280
    else:
        h, w = 720, 1920

    frame_bytes = h * w * 3
    n_frames = len(raw) // frame_bytes
    if n_frames == 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8)

    frames = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(n_frames, h, w, 3)
    return frames


def extract_video_bytes(
    video_path: Path,
    start: float,
    duration: float,
    crop_right: bool = False,
    resize: tuple[int, int] | None = None,
) -> bytes:
    """Extract a video segment as MP4 bytes.

    Returns encoded bytes for the requested [start, start+duration] window.
    """
    vf_parts = []
    if crop_right:
        vf_parts.append("crop=1280:720:640:0")
    if resize is not None:
        h, w = resize
        vf_parts.append(f"scale={w}:{h}")

    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
    ]

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += [
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+frag_keyframe+empty_moov",
        "-f", "mp4",
        "pipe:1",
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg video extraction failed: {result.stderr.decode()[-300:]}"
        )

    return result.stdout or b""


# ─────────────────────────────────────────────────────────────────────────────
# Transcript windowing
# ─────────────────────────────────────────────────────────────────────────────

def _window_segments(
    whisperx_json: dict,
    start: float,
    end: float,
) -> tuple[list[dict], list[dict], str]:
    """Extract segments and words within [start, end] from WhisperX JSON.

    Returns (segments, words, formatted_transcript).
    """
    segments = []
    words = []
    transcript_lines = []

    for seg in whisperx_json.get("segments", []):
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        # Segment overlaps with window
        if seg_end > start and seg_start < end:
            segments.append(seg)
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "").strip()
            transcript_lines.append(f"[{speaker}] {text}")

            for w in seg.get("words", []):
                w_start = w.get("start", 0)
                w_end = w.get("end", w_start)
                if w_end > start and w_start < end:
                    words.append(w)

    return segments, words, "\n".join(transcript_lines)


def _window_manual_srt(
    entries: list[dict],
    start: float,
    end: float,
) -> str:
    """Extract manual SRT entries overlapping [start, end] as formatted text."""
    lines = []
    for e in entries:
        if e["end"] > start and e["start"] < end:
            lines.append(e["text"])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_session_annotations(annotations_dir: Path, session_id: str) -> list[dict]:
    """Load the annotation JSON for *session_id*.

    Looks for ``<sid>_L_mosaic_sp.json`` (train) first, then
    ``<sid>_L_mosaic.json`` (test).  Returns the annotation list, or ``[]``.
    """
    for suffix in ("_sp", ""):
        path = annotations_dir / f"{session_id}_L_mosaic{suffix}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("annotations", [])
    return []


def _window_annotations(
    annotations: list[dict],
    start: float,
    end: float,
) -> list[dict]:
    """Return annotations whose time span overlaps ``[start, end]``."""
    return [
        ann for ann in annotations
        if ann.get("end", 0) > start and ann.get("start", 0) < end
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HHOIDataset(Dataset):
    """PyTorch Dataset for UDIVA-HHOI dyadic interaction data.

    Each item is a time chunk from one session, containing video frames,
    speaker-labelled transcripts, and session metadata.
    """

    def __init__(
        self,
        split: str = "train",
        config: DatasetConfig | None = None,
        *,
        # Convenience kwargs (override config fields)
        data_root: str | None = None,
        chunk_duration: float | None = None,
        chunk_stride: float | None = None,
        fps: float | None = None,
        crop_right: bool | None = None,
        resize: tuple[int, int] | None = None,
        load_frames: bool = True,
        load_video: bool = False,
        sessions: list[str] | None = None,
    ):
        self.split = split
        self.load_frames = load_frames
        self.load_video = load_video

        # Build config
        cfg = config or DatasetConfig()
        if data_root is not None:
            cfg.data_root = data_root
        if chunk_duration is not None:
            cfg.chunk_duration = chunk_duration
        if chunk_stride is not None:
            cfg.chunk_stride = chunk_stride
        if fps is not None:
            cfg.fps = fps
        if crop_right is not None:
            cfg.crop_right = crop_right
        if resize is not None:
            cfg.resize = resize
        self.cfg = cfg

        self.root = Path(cfg.data_root)

        # Load metadata
        self._all_meta = load_all_metadata(self.root / cfg.metadata_dir)

        # Resolve available sessions for this split.
        # Annotation filenames are the source of truth:
        #   *_sp.json  → train      *  .json (no _sp) → test
        # Fall back to metadata CSVs only for splits without annotations
        # (e.g. "val").
        ann_dir = self.root / cfg.annotations_dir
        ann_splits = load_annotation_splits(ann_dir)
        if split in ann_splits and ann_splits[split]:
            split_sessions = set(ann_splits[split])
        else:
            split_map = load_split_session_ids(self.root / cfg.metadata_dir)
            split_sessions = set(split_map.get(split, []))

        # Find sessions that have both a mosaic video and WhisperX output
        mosaics_dir = self.root / cfg.mosaics_dir
        whisperx_dir = self.root / cfg.whisperx_dir
        available = []
        for session_dir in sorted(mosaics_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            sid = session_dir.name
            if sid not in split_sessions:
                continue
            if sessions is not None and sid not in sessions:
                continue
            video = session_dir / f"{sid}_L_mosaic.mp4"
            wx_json = whisperx_dir / sid / f"{sid}.json"
            if video.exists() and wx_json.exists():
                available.append(sid)

        # Pre-load per-session data and build chunk index
        self._sessions: dict[str, dict] = {}
        self._chunks: list[tuple[str, int, float, float]] = []  # (session, idx, start, end)

        for sid in available:
            video_path = mosaics_dir / sid / f"{sid}_L_mosaic.mp4"
            wx_json_path = whisperx_dir / sid / f"{sid}.json"
            manual_srt_path = self.root / cfg.transcripts_dir / sid / f"{sid}_lego.srt"

            # Get video duration
            duration = get_video_duration(video_path)

            # Load WhisperX JSON
            wx_data = json.loads(wx_json_path.read_text(encoding="utf-8"))

            # Load manual SRT if exists
            manual_entries = []
            if manual_srt_path.exists():
                manual_entries = _parse_srt_entries(manual_srt_path)

            # Load annotations
            annotations = _load_session_annotations(ann_dir, sid)

            # Session metadata
            meta = self._all_meta.get(sid, {})

            self._sessions[sid] = {
                "video_path": video_path,
                "duration": duration,
                "whisperx": wx_data,
                "manual_srt": manual_entries,
                "annotations": annotations,
                "metadata": meta,
            }

            # Generate chunks
            chunk_start = 0.0
            chunk_idx = 0
            while chunk_start < duration:
                chunk_end = min(chunk_start + cfg.chunk_duration, duration)
                # Skip very short tail chunks (< 1 second)
                if chunk_end - chunk_start < 1.0:
                    break

                # Optional: filter by minimum transcript words
                if cfg.min_transcript_words > 0:
                    _, words, _ = _window_segments(
                        wx_data, chunk_start, chunk_end
                    )
                    if len(words) < cfg.min_transcript_words:
                        chunk_start += cfg.chunk_stride
                        chunk_idx += 1
                        continue

                self._chunks.append((sid, chunk_idx, chunk_start, chunk_end))
                chunk_start += cfg.chunk_stride
                chunk_idx += 1

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid, chunk_idx, chunk_start, chunk_end = self._chunks[idx]
        session = self._sessions[sid]
        chunk_duration = chunk_end - chunk_start

        # ── Video frames ──
        if self.load_frames:
            frames_np = extract_frames(
                session["video_path"],
                start=chunk_start,
                duration=chunk_duration,
                fps=self.cfg.fps,
                crop_right=self.cfg.crop_right,
                resize=self.cfg.resize,
            )
            # (T, H, W, C) → (T, C, H, W) float32 [0, 1]
            frames = torch.from_numpy(frames_np.copy()).permute(0, 3, 1, 2).float() / 255.0
        else:
            frames = torch.empty(0)

        if self.load_video:
            video = extract_video_bytes(
                session["video_path"],
                start=chunk_start,
                duration=chunk_duration,
                crop_right=self.cfg.crop_right,
                resize=self.cfg.resize,
            )
        else:
            video = None

        # ── Transcripts ──
        segments, words, transcript_text = _window_segments(
            session["whisperx"], chunk_start, chunk_end
        )
        manual_text = _window_manual_srt(
            session["manual_srt"], chunk_start, chunk_end
        )

        # ── Annotations (ground-truth actions / utterances) ──
        annotations = _window_annotations(
            session["annotations"], chunk_start, chunk_end
        )

        # ── Metadata ──
        meta = session["metadata"]
        session_meta = meta.get("session_meta", {})

        return {
            # Identifiers
            "session_id": sid,
            "split": self.split,
            "chunk_index": chunk_idx,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,

            # Video
            "frames": frames,
            "video_path": str(session["video_path"]),
            "video": video,
            "video_mime": "video/mp4" if video is not None else None,

            # Transcripts (WhisperX corrected with speaker labels)
            "transcript": transcript_text,          # formatted string
            "segments": segments,                   # list of segment dicts
            "words": words,                         # list of word dicts

            # Manual ground-truth transcript
            "manual_transcript": manual_text,

            # Ground-truth annotations (actions, utterances, causality, etc.)
            "annotations": annotations,

            # Session metadata
            "language": session_meta.get("LANGUAGE", ""),
            "which_lego": session_meta.get("WHICH_LEGO", ""),
            "part1_id": session_meta.get("PART.1", ""),
            "part2_id": session_meta.get("PART.2", ""),
            "metadata": session_meta,
        }

    @property
    def session_ids(self) -> list[str]:
        """List of session IDs in this dataset."""
        return list(self._sessions.keys())

    @property
    def num_sessions(self) -> int:
        return len(self._sessions)

    def get_session_info(self, session_id: str) -> dict:
        """Get full session info (duration, metadata, etc.)."""
        s = self._sessions[session_id]
        return {
            "session_id": session_id,
            "duration": s["duration"],
            "num_segments": len(s["whisperx"].get("segments", [])),
            "num_manual_entries": len(s["manual_srt"]),
            "num_annotations": len(s["annotations"]),
            "metadata": s["metadata"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def hhoi_collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Custom collate that handles variable-length frames and nested dicts.

    Frames are zero-padded to the longest sequence in the batch.
    Non-tensor fields are collected into lists.
    """
    # Separate tensor and non-tensor fields
    tensor_keys = {"frames"}
    list_keys = set(batch[0].keys()) - tensor_keys

    collated: dict[str, Any] = {}

    # Non-tensor fields → simple lists
    for key in list_keys:
        collated[key] = [sample[key] for sample in batch]

    # Frames: pad to max length in batch → (B, T_max, C, H, W)
    frame_tensors = [sample["frames"] for sample in batch]
    if all(f.numel() == 0 for f in frame_tensors):
        collated["frames"] = torch.empty(len(batch), 0)
    else:
        max_t = max(f.shape[0] for f in frame_tensors if f.numel() > 0)
        padded = []
        masks = []
        for f in frame_tensors:
            if f.numel() == 0:
                # No frames loaded
                c = h = w = 3, 1, 1  # dummy
                padded.append(torch.zeros(max_t, 3, 1, 1))
                masks.append(torch.zeros(max_t, dtype=torch.bool))
            else:
                t, c, h, w = f.shape
                if t < max_t:
                    pad = torch.zeros(max_t - t, c, h, w)
                    padded.append(torch.cat([f, pad], dim=0))
                else:
                    padded.append(f[:max_t])
                mask = torch.zeros(max_t, dtype=torch.bool)
                mask[:min(t, max_t)] = True
                masks.append(mask)
        collated["frames"] = torch.stack(padded)       # (B, T, C, H, W)
        collated["frames_mask"] = torch.stack(masks)    # (B, T) — True = valid

    return collated


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloader(
    split: str = "train",
    data_root: str = ".",
    chunk_duration: float = 10.0,
    chunk_stride: float = 5.0,
    fps: float = 5.0,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool | None = None,
    crop_right: bool = False,
    resize: tuple[int, int] | None = None,
    load_frames: bool = True,
    load_video: bool = False,
    sessions: list[str] | None = None,
    **dataset_kwargs,
) -> DataLoader:
    """Create a DataLoader for the HHOI dataset.

    Args:
        split: One of "train", "val", "test".
        data_root: Root directory containing mosaics/, whisperx_corrected/, etc.
        chunk_duration: Length of each video chunk in seconds.
        chunk_stride: Step between chunk starts in seconds.
        fps: Frame sampling rate.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle. Defaults to True for train, False otherwise.
        crop_right: Crop to right 1280×720 of the mosaic.
        resize: (H, W) tuple to resize frames.
        load_frames: If False, skip frame extraction (text-only mode).
        load_video: If True, include MP4 bytes for each chunk.
        sessions: Restrict to specific session IDs.
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = HHOIDataset(
        split=split,
        data_root=data_root,
        chunk_duration=chunk_duration,
        chunk_stride=chunk_stride,
        fps=fps,
        crop_right=crop_right,
        resize=resize,
        load_frames=load_frames,
        load_video=load_video,
        sessions=sessions,
        **dataset_kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=hhoi_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI — quick test / inspection
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Quick inspection of the dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect HHOI Dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--chunk_duration", type=float, default=10.0)
    parser.add_argument("--chunk_stride", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--crop_right", action="store_true")
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                        metavar=("H", "W"))
    parser.add_argument("--no_frames", action="store_true",
                        help="Skip video frame loading (text-only inspect)")
    parser.add_argument("--with_video", action="store_true",
                        help="Include encoded MP4 bytes for each chunk")
    parser.add_argument("--sample", type=int, default=0,
                        help="Index of sample to inspect")
    parser.add_argument("--batch_test", action="store_true",
                        help="Test batched loading")
    args = parser.parse_args()

    resize = tuple(args.resize) if args.resize else None
    load_frames = not args.no_frames
    load_video = args.with_video

    dataset = HHOIDataset(
        split=args.split,
        data_root=args.data_root,
        chunk_duration=args.chunk_duration,
        chunk_stride=args.chunk_stride,
        fps=args.fps,
        crop_right=args.crop_right,
        resize=resize,
        load_frames=load_frames,
        load_video=load_video,
    )

    print(f"Split: {args.split}")
    print(f"Sessions: {dataset.num_sessions} ({', '.join(dataset.session_ids)})")
    print(f"Total chunks: {len(dataset)}")
    print(f"Config: duration={args.chunk_duration}s, stride={args.chunk_stride}s, fps={args.fps}")
    print()

    if len(dataset) == 0:
        print("No chunks available.")
        return

    # Show one sample
    idx = min(args.sample, len(dataset) - 1)
    sample = dataset[idx]

    print(f"--- Sample {idx} ---")
    print(f"Session:    {sample['session_id']}")
    print(f"Chunk:      [{sample['chunk_start']:.1f}s — {sample['chunk_end']:.1f}s] "
          f"(idx={sample['chunk_index']})")
    print(f"Frames:     {sample['frames'].shape}")
    if sample["video"] is not None:
        print(f"Video:      {len(sample['video'])} bytes ({sample['video_mime']})")
    print(f"Language:   {sample['language']}")
    print(f"Lego set:   {sample['which_lego']}")
    print(f"Words:      {len(sample['words'])}")
    print(f"Segments:   {len(sample['segments'])}")
    print(f"Annotations:{len(sample['annotations'])}")
    print()
    print("Transcript (WhisperX corrected):")
    for line in sample["transcript"].split("\n")[:8]:
        print(f"  {line}")
    if sample["transcript"].count("\n") > 7:
        print(f"  ... ({sample['transcript'].count(chr(10)) + 1} lines total)")
    print()
    print("Manual transcript:")
    for line in sample["manual_transcript"].split("\n")[:5]:
        print(f"  {line}")
    print()
    print(f"Annotations ({len(sample['annotations'])} in chunk):")
    for ann in sample["annotations"][:5]:
        subj = ann.get('subject', '?')
        act = ann.get('act', '?')
        hl = ann.get('high_level_action', 'none')
        ll = ann.get('low_level_action', 'none')
        ut = ann.get('utterance_type', 'none')
        t0, t1 = ann.get('start', 0), ann.get('end', 0)
        label = ut if act == 'V' else f"{hl}/{ll}"
        print(f"  [{t0:.1f}-{t1:.1f}] {subj} ({act}): {label}")
    if len(sample['annotations']) > 5:
        print(f"  ... ({len(sample['annotations'])} total)")

    # Batch test
    if args.batch_test:
        print()
        print("--- Batch Loading Test ---")
        loader = create_dataloader(
            split=args.split,
            data_root=args.data_root,
            chunk_duration=args.chunk_duration,
            chunk_stride=args.chunk_stride,
            fps=args.fps,
            batch_size=4,
            crop_right=args.crop_right,
            resize=resize,
            load_frames=load_frames,
            load_video=load_video,
        )
        for batch in loader:
            print(f"Batch frames:    {batch['frames'].shape}")
            if "frames_mask" in batch:
                print(f"Batch mask:      {batch['frames_mask'].shape}")
            if any(v is not None for v in batch["video"]):
                print(f"Batch video len: {[len(v) if v is not None else 0 for v in batch['video']]}")
            print(f"Batch sessions:  {batch['session_id']}")
            print(f"Batch chunks:    {list(zip(batch['chunk_start'], batch['chunk_end']))}")
            print(f"Batch words:     {[len(w) for w in batch['words']]}")
            break
        print(f"\nTotal batches: {len(loader)}")


if __name__ == "__main__":
    main()
