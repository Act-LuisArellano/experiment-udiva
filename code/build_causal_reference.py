"""
Build the official-format causal `reference.json` from the GT spec
(data-slow/samples_gt/causal_samples_gt.json).

The GT spec is already in the right shape per effect (effect/options/causes),
nested under top key "SEGMENT"; the official scorer expects top key "causal".
This script just renames the root and writes the reference. video_id keys
(e.g. "005013.mp4") are kept verbatim so submission/reference align exactly.

Run from code/ with host python3:
    python3 build_causal_reference.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GT_SPEC = Path("../data-slow/samples_gt/causal_samples_gt.json")
OUT_DIR = Path("../data-slow/results/gf_causal")
ROOT_KEY = "causal"


def build(gt_spec: Path, out_dir: Path) -> None:
    spec = json.loads(gt_spec.read_text())
    # top key is "SEGMENT" in the GT spec
    segments = spec[next(iter(spec))]

    reference = {ROOT_KEY: {}}
    n_effects = n_causes = 0
    for video_id, effects in segments.items():
        reference[ROOT_KEY][video_id] = {}
        for effect_id, rec in effects.items():
            # scorer only needs `causes`; keep effect/options for readability
            reference[ROOT_KEY][video_id][effect_id] = {
                "effect": rec.get("effect", {}),
                "options": rec.get("options", {}),
                "causes": rec["causes"],
            }
            n_effects += 1
            n_causes += len(rec["causes"])
        print(f"  {video_id}: {len(effects)} effects")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference.json").write_text(json.dumps(reference))
    print(f"\n{n_effects} effects, {n_causes} causes → {out_dir / 'reference.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-spec", default=str(GT_SPEC))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    build(Path(args.gt_spec).resolve(), Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
