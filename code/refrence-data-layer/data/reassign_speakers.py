"""
Reassign WhisperX word-level speakers using manual (ground-truth) subtitles.

WhisperX produces word-level timestamps (JSON + SRT) but without speaker
attribution.  Manual subtitles from UDIVA v0.5 have correct speaker labels
(PART.1, PART.2, SUPERVISOR) at the sentence level.

This script maps each WhisperX word to the correct speaker by checking which
manual subtitle's time range contains that word's midpoint timestamp.

Inputs consumed per session:
  - whisperx_out/<session>/<session>_L_mosaic.json   (WhisperX JSON)
  - transcriptions_filtered/<session>/<session>_lego.srt  (manual SRT)

Outputs written per session:
  - <output_dir>/<session>/<session>_words.srt      (word-level SRT with speakers)
  - <output_dir>/<session>/<session>_segments.srt   (segment-level SRT with speakers)
  - <output_dir>/<session>/<session>.json           (full JSON with speaker field)

Usage:
    # Batch — all sessions found under whisperx_out/ and transcriptions_filtered/
    python reassign_speakers.py

    # Single pair
    python reassign_speakers.py \\
        --whisperx_json whisperx_out/005013/005013_L_mosaic.json \\
        --manual_srt transcriptions_filtered/005013/005013_lego.srt

    # Custom dirs
    python reassign_speakers.py \\
        --whisperx_dir whisperx_out \\
        --manual_dir transcriptions_filtered \\
        --output_dir whisperx_corrected
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SRT helpers
# ─────────────────────────────────────────────────────────────────────────────

SRT_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    r"\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def _ts_to_ms(h: str, m: str, s: str, ms: str) -> float:
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def ms_to_srt_ts(ms: float) -> str:
    """Convert milliseconds to SRT timestamp string (HH:MM:SS,mmm)."""
    if ms < 0:
        ms = 0.0
    h = int(ms // 3600000)
    ms %= 3600000
    m = int(ms // 60000)
    ms %= 60000
    s = int(ms // 1000)
    ms_rem = int(ms % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"


def sec_to_ms(sec: float) -> float:
    return sec * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_manual_srt(srt_path: str | Path) -> list[dict]:
    """Parse manual subtitles into list of dicts with speaker, timing, text.

    Expected format per entry:
        PART.1: Some text here...
        PART.2: Another line...
        SUPERVISOR: Instructions...
    """
    text = Path(srt_path).read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    entries: list[dict] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        match = SRT_TS_RE.search(lines[1])
        if not match:
            continue
        g = match.groups()
        start_ms = _ts_to_ms(g[0], g[1], g[2], g[3])
        end_ms = _ts_to_ms(g[4], g[5], g[6], g[7])
        body = "\n".join(lines[2:]).strip()

        spk_match = re.match(r'^(PART\.\d+|SUPERVISOR):\s*(.*)', body, re.DOTALL)
        if spk_match:
            speaker = spk_match.group(1)
            content = spk_match.group(2).replace('\n', ' ')
        else:
            speaker = "UNKNOWN"
            content = body.replace('\n', ' ')

        entries.append({
            'start_ms': start_ms,
            'end_ms': end_ms,
            'speaker': speaker,
            'text': content,
        })
    return entries


def load_whisperx_json(json_path: str | Path) -> dict:
    """Load WhisperX JSON output (segments with nested words)."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Speaker re-attribution
# ─────────────────────────────────────────────────────────────────────────────

def find_speaker_for_timestamp(
    timestamp_ms: float,
    manual_entries: list[dict],
    tolerance_ms: float = 500,
) -> Optional[str]:
    """Find the manual subtitle speaker whose time range covers *timestamp_ms*.

    If multiple manual entries overlap (interruptions), pick the one whose
    midpoint is closest. If no exact overlap, use tolerance to find nearest.
    """
    # Exact overlaps
    overlapping = [
        e for e in manual_entries
        if e['start_ms'] <= timestamp_ms <= e['end_ms']
    ]

    if len(overlapping) == 1:
        return overlapping[0]['speaker']
    elif len(overlapping) > 1:
        best = min(
            overlapping,
            key=lambda e: abs((e['start_ms'] + e['end_ms']) / 2 - timestamp_ms),
        )
        return best['speaker']

    # No exact overlap — find nearest within tolerance
    best_entry = None
    best_dist = float('inf')
    for entry in manual_entries:
        dist = min(
            abs(entry['start_ms'] - timestamp_ms),
            abs(entry['end_ms'] - timestamp_ms),
        )
        if dist < best_dist:
            best_dist = dist
            best_entry = entry

    if best_entry and best_dist <= tolerance_ms:
        return best_entry['speaker']

    return None


def reassign_speakers_json(
    whisperx_data: dict,
    manual_entries: list[dict],
    tolerance_ms: float = 500,
) -> tuple[dict, dict]:
    """Assign a speaker to every word and segment in the WhisperX JSON.

    Returns (corrected_data, stats).
    """
    stats = {'words': 0, 'matched': 0, 'unmatched': 0}
    corrected_segments: list[dict] = []

    for seg in whisperx_data.get("segments", []):
        seg_copy = dict(seg)
        corrected_words: list[dict] = []
        seg_speakers: list[str] = []

        for w in seg.get("words", []):
            w_copy = dict(w)
            stats['words'] += 1

            # Use word midpoint for matching
            w_start = sec_to_ms(w.get("start", 0))
            w_end = sec_to_ms(w.get("end", w.get("start", 0)))
            mid_ms = (w_start + w_end) / 2

            speaker = find_speaker_for_timestamp(mid_ms, manual_entries, tolerance_ms)
            if speaker:
                stats['matched'] += 1
            else:
                stats['unmatched'] += 1
                speaker = "UNKNOWN"

            w_copy['speaker'] = speaker
            corrected_words.append(w_copy)
            seg_speakers.append(speaker)

        seg_copy['words'] = corrected_words

        # Segment-level speaker: majority vote among its words
        if seg_speakers:
            seg_copy['speaker'] = max(set(seg_speakers), key=seg_speakers.count)
        else:
            seg_copy['speaker'] = "UNKNOWN"

        corrected_segments.append(seg_copy)

    result = dict(whisperx_data)
    result['segments'] = corrected_segments
    return result, stats


# ─────────────────────────────────────────────────────────────────────────────
# Output generation
# ─────────────────────────────────────────────────────────────────────────────

def write_word_srt(segments: list[dict], path: Path) -> None:
    """Write word-level SRT: one entry per word with speaker label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    idx = 1
    for seg in segments:
        for w in seg.get("words", []):
            start = sec_to_ms(w.get("start", 0))
            end = sec_to_ms(w.get("end", w.get("start", 0)))
            speaker = w.get("speaker", "UNKNOWN")
            word = w.get("word", "")
            lines.append(str(idx))
            lines.append(f"{ms_to_srt_ts(start)} --> {ms_to_srt_ts(end)}")
            lines.append(f"[{speaker}] {word}")
            lines.append("")
            idx += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def write_segment_srt(segments: list[dict], path: Path) -> None:
    """Write segment-level SRT with speaker label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        start = sec_to_ms(seg.get("start", 0))
        end = sec_to_ms(seg.get("end", 0))
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        lines.append(str(i))
        lines.append(f"{ms_to_srt_ts(start)} --> {ms_to_srt_ts(end)}")
        lines.append(f"[{speaker}] {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_pair(
    whisperx_json_path: str | Path,
    manual_srt_path: str | Path,
    output_dir: str | Path,
    tolerance_ms: float = 500,
    verbose: bool = True,
) -> dict:
    """Process a single WhisperX JSON + manual SRT pair.

    Writes three output files and returns stats.
    """
    whisperx_json_path = Path(whisperx_json_path)
    manual_srt_path = Path(manual_srt_path)
    output_dir = Path(output_dir)

    session = whisperx_json_path.parent.name
    out_session = output_dir / session
    out_session.mkdir(parents=True, exist_ok=True)

    # Parse
    manual_entries = parse_manual_srt(manual_srt_path)
    whisperx_data = load_whisperx_json(whisperx_json_path)

    # Re-attribute
    corrected, stats = reassign_speakers_json(whisperx_data, manual_entries, tolerance_ms)

    # Write outputs
    write_word_srt(corrected["segments"], out_session / f"{session}_words.srt")
    write_segment_srt(corrected["segments"], out_session / f"{session}_segments.srt")
    write_json(corrected, out_session / f"{session}.json")

    if verbose:
        total_words = stats['words']
        matched = stats['matched']
        unmatched = stats['unmatched']
        pct = (matched / total_words * 100) if total_words else 0
        n_seg = len(corrected.get("segments", []))
        print(
            f"  {session}: {n_seg} segments, {total_words} words — "
            f"{matched} matched ({pct:.0f}%), {unmatched} unmatched"
        )

    return stats


def discover_pairs(
    whisperx_dir: Path,
    manual_dir: Path,
) -> list[tuple[Path, Path]]:
    """Find matching (whisperx_json, manual_srt) pairs across all sessions."""
    pairs: list[tuple[Path, Path]] = []
    for json_path in sorted(whisperx_dir.rglob("*.json")):
        session = json_path.parent.name
        # Try to find the manual SRT
        manual_srt = manual_dir / session / f"{session}_lego.srt"
        if manual_srt.exists():
            pairs.append((json_path, manual_srt))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reassign WhisperX word-level speakers using manual subtitles.",
    )
    # Single-pair mode
    parser.add_argument("--whisperx_json", default=None,
                        help="Path to a single WhisperX JSON file")
    parser.add_argument("--manual_srt", default=None,
                        help="Path to the corresponding manual SRT file")
    # Batch mode (default)
    parser.add_argument("--whisperx_dir", default="whisperx_out",
                        help="Root dir of WhisperX outputs (default: whisperx_out/)")
    parser.add_argument("--manual_dir", default="transcriptions_filtered",
                        help="Root dir of manual SRTs (default: transcriptions_filtered/)")
    parser.add_argument("--output_dir", "-o", default="whisperx_corrected",
                        help="Output directory (default: whisperx_corrected/)")
    parser.add_argument("--tolerance", "-t", type=float, default=500,
                        help="Tolerance in ms for matching words to manual entries")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.whisperx_json and args.manual_srt:
        # Single-pair mode
        print("Processing single pair...")
        process_pair(
            args.whisperx_json,
            args.manual_srt,
            output_dir=output_dir,
            tolerance_ms=args.tolerance,
        )
    else:
        # Batch mode
        pairs = discover_pairs(Path(args.whisperx_dir), Path(args.manual_dir))
        if not pairs:
            print("No matching pairs found. Check --whisperx_dir and --manual_dir.")
            return

        print(f"Found {len(pairs)} session(s) to process.\n")
        total_stats = {'words': 0, 'matched': 0, 'unmatched': 0}
        for wx_json, man_srt in pairs:
            stats = process_pair(
                wx_json, man_srt,
                output_dir=output_dir,
                tolerance_ms=args.tolerance,
            )
            for k in total_stats:
                total_stats[k] += stats[k]

        pct = (total_stats['matched'] / total_stats['words'] * 100) if total_stats['words'] else 0
        print(
            f"\nDone: {total_stats['words']} words total, "
            f"{total_stats['matched']} matched ({pct:.0f}%), "
            f"{total_stats['unmatched']} unmatched."
        )


if __name__ == "__main__":
    main()
