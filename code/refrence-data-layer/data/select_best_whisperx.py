"""
Select the best WhisperX output per session by comparing against manual transcripts.

For each session, compares two WhisperX runs (e.g. original vs audio-enhanced)
against the manual ground-truth transcript.  The version whose words better
match the manual transcript's content within each time window is selected.

Metric: For every manual subtitle entry, find time-overlapping WhisperX words
and compute word-level recall (what fraction of manual words appear in WhisperX).
The version with higher average recall per session wins.

Usage:
    python select_best_whisperx.py                         # defaults
    python select_best_whisperx.py \\
        --dir_a whisperx_out_original \\
        --dir_b whisperx_out_enhanced \\
        --manual_dir transcriptions_filtered \\
        --output_dir whisperx_out
"""

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation for comparison
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: str) -> list[str]:
    """Lower-case, strip accents, remove punctuation, split into word tokens."""
    text = text.lower()
    # Strip accents
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = _PUNCT_RE.sub("", text)
    return text.split()


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

SRT_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    r"\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def _ts_to_ms(h, m, s, ms):
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def parse_manual_srt(path: Path) -> list[dict]:
    """Parse manual SRT → list of {start_ms, end_ms, text, words}."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    entries = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        match = SRT_TS_RE.search(lines[1])
        if not match:
            continue
        g = match.groups()
        start_ms = _ts_to_ms(*g[:4])
        end_ms = _ts_to_ms(*g[4:])
        body = "\n".join(lines[2:]).strip()
        # Strip speaker prefix
        spk_match = re.match(r"^(PART\.\d+|SUPERVISOR):\s*(.*)", body, re.DOTALL)
        content = spk_match.group(2) if spk_match else body
        content = content.replace("\n", " ")
        words = normalise(content)
        if words:
            entries.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": content,
                "words": words,
            })
    return entries


def load_whisperx_words(json_path: Path) -> list[dict]:
    """Load WhisperX JSON and extract flat list of {start_ms, end_ms, word, norm_word}."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            start = w.get("start", 0)
            end = w.get("end", start)
            word = w.get("word", "")
            norm = normalise(word)
            if norm:
                words.append({
                    "start_ms": start * 1000,
                    "end_ms": end * 1000,
                    "word": word,
                    "norm_word": norm[0],
                })
    return words


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_session(
    manual_entries: list[dict],
    whisperx_words: list[dict],
    tolerance_ms: float = 1000,
) -> dict:
    """Score how well WhisperX words match the manual transcript.

    For each manual entry, collect WhisperX words that overlap its time range
    (with tolerance) and compute:
      - recall: fraction of manual words found in WhisperX words
      - precision: fraction of WhisperX words (in the time window) found in manual

    Returns aggregate metrics.
    """
    total_manual_words = 0
    total_recalled = 0
    total_wx_words_in_range = 0
    total_precise = 0
    entries_scored = 0

    for entry in manual_entries:
        e_start = entry["start_ms"] - tolerance_ms
        e_end = entry["end_ms"] + tolerance_ms
        manual_words = entry["words"]

        # Collect WhisperX words overlapping this time range
        wx_in_range = [
            w["norm_word"] for w in whisperx_words
            if w["start_ms"] >= e_start and w["end_ms"] <= e_end
        ]

        if not manual_words:
            continue

        entries_scored += 1
        total_manual_words += len(manual_words)
        total_wx_words_in_range += len(wx_in_range)

        # Recall: how many manual words are found in WhisperX?
        wx_set = set(wx_in_range)
        recalled = sum(1 for w in manual_words if w in wx_set)
        total_recalled += recalled

        # Precision: how many WhisperX words match manual?
        manual_set = set(manual_words)
        precise = sum(1 for w in wx_in_range if w in manual_set)
        total_precise += precise

    recall = total_recalled / total_manual_words if total_manual_words > 0 else 0
    precision = total_precise / total_wx_words_in_range if total_wx_words_in_range > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "entries_scored": entries_scored,
        "manual_words": total_manual_words,
        "wx_words_in_range": total_wx_words_in_range,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Select the best WhisperX output per session."
    )
    parser.add_argument("--dir_a", default="whisperx_out_original",
                        help="First WhisperX output dir (default: whisperx_out_original)")
    parser.add_argument("--dir_b", default="whisperx_out_enhanced",
                        help="Second WhisperX output dir (default: whisperx_out_enhanced)")
    parser.add_argument("--manual_dir", default="transcriptions_filtered",
                        help="Manual transcript dir (default: transcriptions_filtered)")
    parser.add_argument("--output_dir", "-o", default="whisperx_out",
                        help="Output directory for selected results (default: whisperx_out)")
    parser.add_argument("--tolerance", "-t", type=float, default=1000,
                        help="Time tolerance in ms for word matching (default: 1000)")
    args = parser.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    manual_dir = Path(args.manual_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover sessions
    sessions_a = {p.parent.name: p for p in sorted(dir_a.rglob("*.json"))}
    sessions_b = {p.parent.name: p for p in sorted(dir_b.rglob("*.json"))}
    all_sessions = sorted(set(sessions_a) | set(sessions_b))

    if not all_sessions:
        print("No sessions found.")
        return

    label_a = dir_a.name
    label_b = dir_b.name

    header = (
        f"{'SESSION':>8} | "
        f"{'Recall-A':>8} {'Prec-A':>7} {'F1-A':>6} {'Words-A':>7} | "
        f"{'Recall-B':>8} {'Prec-B':>7} {'F1-B':>6} {'Words-B':>7} | "
        f"{'WINNER':>10}"
    )
    print(f"A = {label_a}")
    print(f"B = {label_b}")
    print()
    print(header)
    print("-" * len(header))

    wins = {"A": 0, "B": 0, "only_A": 0, "only_B": 0}

    for session in all_sessions:
        manual_srt = manual_dir / session / f"{session}_lego.srt"
        if not manual_srt.exists():
            continue

        manual_entries = parse_manual_srt(manual_srt)
        if not manual_entries:
            continue

        has_a = session in sessions_a
        has_b = session in sessions_b

        # Score A
        if has_a:
            wx_words_a = load_whisperx_words(sessions_a[session])
            score_a = score_session(manual_entries, wx_words_a, args.tolerance)
        else:
            score_a = {"recall": 0, "precision": 0, "f1": 0, "wx_words_in_range": 0}

        # Score B
        if has_b:
            wx_words_b = load_whisperx_words(sessions_b[session])
            score_b = score_session(manual_entries, wx_words_b, args.tolerance)
        else:
            score_b = {"recall": 0, "precision": 0, "f1": 0, "wx_words_in_range": 0}

        # Select winner by F1 score
        if not has_a:
            winner = "B (only)"
            wins["only_B"] += 1
            src_dir = dir_b / session
        elif not has_b:
            winner = "A (only)"
            wins["only_A"] += 1
            src_dir = dir_a / session
        elif score_a["f1"] >= score_b["f1"]:
            winner = "A"
            wins["A"] += 1
            src_dir = dir_a / session
        else:
            winner = "B"
            wins["B"] += 1
            src_dir = dir_b / session

        # Copy winner to output
        dst_dir = output_dir / session
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

        # Format scores
        def fmt_score(s):
            return (
                f"{s['recall']:>8.1%} {s['precision']:>7.1%} "
                f"{s['f1']:>6.1%} {s['wx_words_in_range']:>7}"
            )

        score_a_str = fmt_score(score_a) if has_a else f"{'---':>8} {'---':>7} {'---':>6} {'---':>7}"
        score_b_str = fmt_score(score_b) if has_b else f"{'---':>8} {'---':>7} {'---':>6} {'---':>7}"

        print(f"{session:>8} | {score_a_str} | {score_b_str} | {winner:>10}")

    print()
    print(
        f"Selected: {wins['A']} × A ({label_a}), "
        f"{wins['B']} × B ({label_b}), "
        f"{wins.get('only_A',0)} only-A, "
        f"{wins.get('only_B',0)} only-B"
    )
    print(f"Output written to: {output_dir}/")


if __name__ == "__main__":
    main()
