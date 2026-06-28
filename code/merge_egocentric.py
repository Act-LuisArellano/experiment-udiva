"""
Merge egocentric per-subject predictions into combined GF-format JSONs.

E1 produced participant_a events, E2 produced participant_b events (same segment
grid, since E1/E2/GF share the time base). For each video + category we union the
two event lists per segment, producing the same
    {category: {video_id: {segment_id: {t_b, t_e, events:[...]}}}}
layout the GF baseline uses — so convert_predictions_to_csv.py and compute_map.py
work unchanged.

Lightweight (json + pathlib only); run with host python3:
    python3 merge_egocentric.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

E1_DIR = Path("../data-slow/results/egocentric/e1")
E2_DIR = Path("../data-slow/results/egocentric/e2")
OUT_DIR = Path("../data-slow/results/gf_baseline_ego")

VIDEO_IDS = ["005013", "020025", "027113", "035040", "041083", "044156", "066067"]
CATEGORIES = ["verbal", "nonverbal"]


def _segments(report: dict, category: str, video_id: str) -> dict:
    """Return {segment_id: {t_b,t_e,events}} from a recognition report, or {}."""
    return report.get(category, {}).get(video_id, {})


def merge_video_category(video_id: str, category: str,
                         e1_dir: Path, e2_dir: Path) -> dict | None:
    e1_path = e1_dir / f"{video_id}_{category}.json"
    e2_path = e2_dir / f"{video_id}_{category}.json"
    if not e1_path.exists() or not e2_path.exists():
        missing = [str(p) for p in (e1_path, e2_path) if not p.exists()]
        print(f"  SKIP {video_id} {category}: missing {missing}")
        return None

    e1 = _segments(json.loads(e1_path.read_text()), category, video_id)  # subject A
    e2 = _segments(json.loads(e2_path.read_text()), category, video_id)  # subject B

    seg_ids = sorted(set(e1) | set(e2))
    merged_segs = {}
    for sid in seg_ids:
        a = e1.get(sid, {})
        b = e2.get(sid, {})
        meta = a or b  # whichever has timing
        merged_segs[sid] = {
            "t_b": meta.get("t_b"),
            "t_e": meta.get("t_e"),
            "events": list(a.get("events", [])) + list(b.get("events", [])),
        }
    return {category: {video_id: merged_segs}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-dir", default=str(E1_DIR))
    parser.add_argument("--e2-dir", default=str(E2_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    e1_dir, e2_dir = Path(args.e1_dir).resolve(), Path(args.e2_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"E1 (participant_a): {e1_dir}")
    print(f"E2 (participant_b): {e2_dir}")
    print(f"Out:                {out_dir}\n")

    n = 0
    for video_id in VIDEO_IDS:
        for category in CATEGORIES:
            merged = merge_video_category(video_id, category, e1_dir, e2_dir)
            if merged is None:
                continue
            out_path = out_dir / f"{video_id}_{category}.json"
            out_path.write_text(json.dumps(merged))
            segs = merged[category][video_id]
            n_ev = sum(len(s["events"]) for s in segs.values())
            print(f"  {video_id} {category}: {len(segs)} segs, {n_ev} events → {out_path.name}")
            n += 1

    print(f"\nMerged {n} files. Next: convert + score with --input-dir {out_dir}")


if __name__ == "__main__":
    main()
