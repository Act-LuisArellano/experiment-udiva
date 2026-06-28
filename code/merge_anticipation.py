"""
Merge anticipation shards into an official submission.json.

  --view exo : combine exo/shards/*.json -> exo/submission.json (each shard already
               carries both participants).
  --view ego : combine ego/e1/shards (participant_a) + ego/e2/shards (participant_b)
               per window -> ego/submission.json.

Any participant missing for a window is filled with one empty hypothesis so the
submission validates (both participants required).

Lightweight (json + pathlib); host python3:
    python3 merge_anticipation.py --view exo
    python3 merge_anticipation.py --view ego
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_ROOT = Path("../data-slow/results/gf_anticipation")
PARTICIPANTS = ("participant_a", "participant_b")


def _empty_part():
    return {"hypotheses": [{"events": []}]}


def _load_shards(shard_dir: Path) -> dict:
    """video_id -> {seg_id -> segment} merged across shards in a dir."""
    out: dict = {}
    for f in sorted(shard_dir.glob("*.json")):
        for vid, segs in json.loads(f.read_text()).get("anticipation", {}).items():
            out.setdefault(vid, {}).update(segs)
    return out


def merge_exo(out_root: Path) -> None:
    data = _load_shards(out_root / "exo" / "shards")
    submission = {"anticipation": {}}
    for vid, segs in data.items():
        submission["anticipation"][vid] = {}
        for sid, seg in segs.items():
            parts = dict(seg.get("participants", {}))
            for p in PARTICIPANTS:
                parts.setdefault(p, _empty_part())
            submission["anticipation"][vid][sid] = {
                "t_b": seg["t_b"], "t_e": seg["t_e"], "participants": parts}
    out = out_root / "exo" / "submission.json"
    out.write_text(json.dumps(submission))
    print(f"exo: {len(submission['anticipation'])} videos -> {out}")


def merge_ego(out_root: Path) -> None:
    e1 = _load_shards(out_root / "ego" / "e1" / "shards")  # participant_a
    e2 = _load_shards(out_root / "ego" / "e2" / "shards")  # participant_b
    submission = {"anticipation": {}}
    vids = sorted(set(e1) | set(e2))
    for vid in vids:
        segs_a, segs_b = e1.get(vid, {}), e2.get(vid, {})
        seg_ids = sorted(set(segs_a) | set(segs_b))
        submission["anticipation"][vid] = {}
        for sid in seg_ids:
            sa, sb = segs_a.get(sid), segs_b.get(sid)
            meta = sa or sb
            parts = {
                "participant_a": (sa or {}).get("participants", {}).get("participant_a", _empty_part()),
                "participant_b": (sb or {}).get("participants", {}).get("participant_b", _empty_part()),
            }
            submission["anticipation"][vid][sid] = {
                "t_b": meta["t_b"], "t_e": meta["t_e"], "participants": parts}
    out = out_root / "ego" / "submission.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(submission))
    print(f"ego: {len(submission['anticipation'])} videos -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True, choices=["exo", "ego"])
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()
    (merge_exo if args.view == "exo" else merge_ego)(out_root)


if __name__ == "__main__":
    main()
