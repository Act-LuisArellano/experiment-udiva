#!/usr/bin/env python3
"""
Filter UDIVA v0.5 transcripts (SRT) to keep only phrases within the 'task'
time window, so they align with the UDIVA-HHOI mosaic videos.

UDIVA v0.5 transcripts span [before → after], while HHOI mosaics only
cover the [task_start → task_end] segment.  This script:

  1. Reads task_limits.json from each split's metadata.
  2. For each SRT file whose session appears in mosaics/, filters out
     subtitles outside the task window.
  3. Re-times the remaining subtitles so t=0 corresponds to task_start
     (matching the mosaic video timeline).
  4. Writes the filtered SRT to an output directory preserving folder
     structure.

Usage:
    python filter_transcripts.py [--output_dir <DIR>] [--tasks lego]
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

SRT_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    r"\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


@dataclass
class Subtitle:
    index: int
    start: float   # seconds
    end: float     # seconds
    text: str


def _ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _sec_to_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    sec %= 3600
    m = int(sec // 60)
    sec %= 60
    s = int(sec)
    ms = int(round((sec - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path: Path) -> list[Subtitle]:
    """Parse an SRT file into a list of Subtitle objects."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    subs: list[Subtitle] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        match = SRT_TS_RE.search(lines[1])
        if not match:
            continue
        g = match.groups()
        start = _ts_to_sec(g[0], g[1], g[2], g[3])
        end = _ts_to_sec(g[4], g[5], g[6], g[7])
        body = "\n".join(lines[2:]).strip()
        subs.append(Subtitle(index=int(lines[0]), start=start, end=end, text=body))
    return subs


def write_srt(subs: list[Subtitle], path: Path) -> None:
    """Write subtitles back to an SRT file with re-numbered indices."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subs, 1):
            f.write(f"{i}\n")
            f.write(f"{_sec_to_ts(sub.start)} --> {_sec_to_ts(sub.end)}\n")
            f.write(f"{sub.text}\n\n")


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------

def filter_and_retime(
    subs: list[Subtitle],
    task_start: float,
    task_end: float,
) -> list[Subtitle]:
    """
    Keep only subtitles overlapping [task_start, task_end] and shift times
    so that task_start becomes 0.
    """
    filtered: list[Subtitle] = []
    for sub in subs:
        # Keep if the subtitle overlaps the task window
        if sub.end <= task_start or sub.start >= task_end:
            continue
        # Clamp to task boundaries and shift
        new_start = max(sub.start, task_start) - task_start
        new_end = min(sub.end, task_end) - task_start
        filtered.append(Subtitle(
            index=0,        # will be re-numbered on write
            start=new_start,
            end=new_end,
            text=sub.text,
        ))
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all_task_limits(metadata_root: Path) -> dict:
    """Merge task_limits.json from all splits into one dict."""
    merged: dict = {}
    for split in ["train", "test", "val"]:
        p = metadata_root / split / "task_limits.json"
        if p.exists():
            merged.update(json.loads(p.read_text()))
    return merged


def find_srt(transcriptions_root: Path, session: str, task: str) -> Path | None:
    """Locate the SRT file for a given session and task across splits."""
    for split in ["train", "test", "val"]:
        candidate = transcriptions_root / split / session / f"{session}_{task}.srt"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter UDIVA v0.5 transcripts to align with HHOI mosaic videos."
    )
    parser.add_argument(
        "--transcriptions_dir",
        default="transcriptions",
        help="Root of downloaded transcriptions (default: transcriptions/)",
    )
    parser.add_argument(
        "--metadata_dir",
        default="metadata",
        help="Root of downloaded metadata (default: metadata/)",
    )
    parser.add_argument(
        "--mosaics_dir",
        default="mosaics",
        help="Root of HHOI mosaic videos (default: mosaics/)",
    )
    parser.add_argument(
        "--output_dir",
        default="transcriptions_filtered",
        help="Output directory for filtered SRTs (default: transcriptions_filtered/)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Tasks to filter (default: infer from mosaic filenames, e.g. lego)",
    )
    args = parser.parse_args()

    transcriptions_root = Path(args.transcriptions_dir)
    metadata_root = Path(args.metadata_dir)
    mosaics_root = Path(args.mosaics_dir)
    output_root = Path(args.output_dir)

    # Load task limits
    task_limits = load_all_task_limits(metadata_root)
    print(f"Loaded task limits for {len(task_limits)} sessions.")

    # Determine which sessions have mosaics
    mosaic_sessions = sorted(
        d.name for d in mosaics_root.iterdir() if d.is_dir()
    )
    print(f"Found {len(mosaic_sessions)} mosaic sessions.")

    # Determine tasks to process
    if args.tasks:
        tasks = args.tasks
    else:
        # Infer from mosaic filenames: <session>_<TASK_LETTER>_mosaic.mp4
        task_letters: set[str] = set()
        for sess in mosaic_sessions:
            for mp4 in (mosaics_root / sess).glob("*.mp4"):
                parts = mp4.stem.split("_")
                if len(parts) >= 2:
                    task_letters.add(parts[1])
        # Map single-letter codes to full task names
        letter_to_task = {"T": "talk", "A": "animals", "G": "ghost", "L": "lego"}
        tasks = sorted(letter_to_task.get(l, l.lower()) for l in task_letters)
        print(f"Inferred tasks from mosaic filenames: {tasks}")

    # Process
    total, written, skipped = 0, 0, 0
    for session in mosaic_sessions:
        if session not in task_limits:
            print(f"  [warn] {session}: no task limits found, skipping")
            skipped += 1
            continue

        for task in tasks:
            total += 1
            limits = task_limits[session].get(task)
            if not limits:
                print(f"  [warn] {session}/{task}: no limits in task_limits.json")
                skipped += 1
                continue

            # Use the first (and usually only) interval
            task_start, task_end = limits[0]

            srt_path = find_srt(transcriptions_root, session, task)
            if srt_path is None:
                print(f"  [warn] {session}/{task}: SRT not found")
                skipped += 1
                continue

            subs = parse_srt(srt_path)
            filtered = filter_and_retime(subs, task_start, task_end)

            out_path = output_root / session / f"{session}_{task}.srt"
            write_srt(filtered, out_path)
            written += 1

            orig_count = len(subs)
            filt_count = len(filtered)
            removed = orig_count - filt_count
            shift_str = f"shift={task_start:.2f}s" if task_start > 0 else "no shift"

            print(
                f"  {session}/{task}: {orig_count} → {filt_count} subs "
                f"({removed} removed, {shift_str})"
            )

    print(f"\nDone: {written} files written, {skipped} skipped out of {total} total.")


if __name__ == "__main__":
    main()
