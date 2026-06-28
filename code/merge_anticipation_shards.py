"""
Merge per-video anticipation shard files into a single official submission.json.

Each shard is `{"anticipation": {video_id: {...segments...}}}`. Merging is a
plain dict update over the per-video keys, so parallel runs that each wrote their
own shards combine cleanly.

Lightweight (json + pathlib only) so it runs with host python3.

Usage (from code/):
    python3 merge_anticipation_shards.py
    python3 merge_anticipation_shards.py --shard-dir ../data-slow/results/gf_anticipation/shards \
                                          --out ../data-slow/results/gf_anticipation/submission.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SHARD_DIR = Path("../data-slow/results/gf_anticipation/shards")
DEFAULT_OUT = Path("../data-slow/results/gf_anticipation/submission.json")


def merge(shard_dir: str | Path, out_path: str | Path) -> dict:
    shard_dir = Path(shard_dir)
    merged = {"anticipation": {}}
    shards = sorted(shard_dir.glob("*.json"))
    for f in shards:
        data = json.loads(f.read_text())
        merged["anticipation"].update(data.get("anticipation", {}))
    Path(out_path).write_text(json.dumps(merged))
    n_videos = len(merged["anticipation"])
    print(f"Merged {len(shards)} shard(s) -> {out_path}  ({n_videos} videos)")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default=str(DEFAULT_SHARD_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    merge(Path(args.shard_dir).resolve(), Path(args.out).resolve())


if __name__ == "__main__":
    main()
