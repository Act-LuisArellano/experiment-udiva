"""
Convert gf_baseline JSON predictions to the CSV format expected by compute_map.py.

Expected output structure:
  {out_dir}/{video_id}/recognition/chunk2.0/final_video_analysis.csv

CSV columns: subject, window_start, window_end,
             low_level_actions, high_level_actions, utterance_types

Usage (from code/ directory):
    python convert_predictions_to_csv.py
    python convert_predictions_to_csv.py --input-dir ../data-slow/results/gf_baseline \
                                          --output-dir ../data-slow/results/gf_baseline_csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SUBJECTS = ["participant_a", "participant_b"]
CHUNK_LABEL = "chunk2.0"
INPUT_DIR = Path("../data-slow/results/gf_baseline")
OUTPUT_DIR = Path("../data-slow/results/gf_baseline_csv")

VIDEO_IDS = ["005013", "020025", "027113", "035040", "041083", "044156", "066067"]


def load_json_segments(path: Path) -> dict[str, dict]:
    """Return {seg_id: {t_b, t_e, events: [...]}} from a prediction JSON."""
    raw = json.loads(path.read_text())
    # top level: {category: {video_id: {seg_id: {...}}}}
    category_data = next(iter(raw.values()))
    video_data = next(iter(category_data.values()))
    return video_data


def convert_video(video_id: str, input_dir: Path, output_dir: Path) -> None:
    verbal_path = input_dir / f"{video_id}_verbal.json"
    nonverbal_path = input_dir / f"{video_id}_nonverbal.json"

    verbal_segs = load_json_segments(verbal_path) if verbal_path.exists() else {}
    nonverbal_segs = load_json_segments(nonverbal_path) if nonverbal_path.exists() else {}

    # Union of all segment IDs (should be identical between files)
    all_seg_ids = sorted(set(verbal_segs) | set(nonverbal_segs))

    rows = []
    for seg_id in all_seg_ids:
        # Grab timing from whichever file has it
        seg = verbal_segs.get(seg_id) or nonverbal_segs.get(seg_id)
        t_b = seg["t_b"]
        t_e = seg["t_e"]

        # Collect per-subject labels for this segment
        utterance_per_subject: dict[str, list[str]] = {s: [] for s in SUBJECTS}
        high_level_per_subject: dict[str, list[str]] = {s: [] for s in SUBJECTS}
        low_level_per_subject: dict[str, list[str]] = {s: [] for s in SUBJECTS}

        for ev in verbal_segs.get(seg_id, {}).get("events", []):
            subj = ev.get("subject", "")
            if subj not in SUBJECTS:
                continue
            ut = ev.get("utterance_type", "").strip().lower().replace(" ", "_")
            if ut and ut != "none":
                utterance_per_subject[subj].append(ut)

        for ev in nonverbal_segs.get(seg_id, {}).get("events", []):
            subj = ev.get("subject", "")
            if subj not in SUBJECTS:
                continue
            hl = ev.get("highlevel_action", ev.get("high_level_action", "")).strip().lower().replace(" ", "_")
            ll = ev.get("lowlevel_action", ev.get("low_level_action", "")).strip().lower().replace(" ", "_")
            if hl and hl != "none":
                high_level_per_subject[subj].append(hl)
            if ll and ll != "none":
                low_level_per_subject[subj].append(ll)

        for subj in SUBJECTS:
            rows.append({
                "subject": subj,
                "window_start": t_b,
                "window_end": t_e,
                "low_level_actions": ",".join(sorted(set(low_level_per_subject[subj]))),
                "high_level_actions": ",".join(sorted(set(high_level_per_subject[subj]))),
                "utterance_types": ",".join(sorted(set(utterance_per_subject[subj]))),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["window_start", "subject"]).reset_index(drop=True)

    out_path = output_dir / video_id / "recognition" / CHUNK_LABEL / "final_video_analysis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  {video_id}: {len(df)} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert gf_baseline JSONs to eval CSVs.")
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    for video_id in VIDEO_IDS:
        convert_video(video_id, input_dir, output_dir)

    print(f"\nDone. Run evaluation with:")
    print(f"  python /home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl/compute_map.py \\")
    print(f"    --results-dir {output_dir} \\")
    print(f"    --gt-dir /home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl/ground_truth \\")
    print(f"    --task recognition --verbose")


if __name__ == "__main__":
    main()
