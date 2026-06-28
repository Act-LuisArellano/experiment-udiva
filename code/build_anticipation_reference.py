"""
Build the official-format anticipation `template.json` and `reference.json`
for the 7 GF videos from the qwen-3vl ground-truth annotations.

Approach 1 (fixed 2s windows): each video is tiled into non-overlapping 2s
windows [t_b, t_e]. For each window and participant, the reference event
sequence is that participant's annotations whose `start` falls in [t_b, t_e),
ordered by start time. Event encoding matches the official scorer:
    verbal     -> [utterance_type, target]
    non-verbal -> [highlevel_action, lowlevel_action, target]

Outputs (consumed by run_gf_anticipation.py and the official score.py):
    <out>/template.json    # segments with empty hypotheses (drives inference)
    <out>/reference.json   # GT event sequences (drives scoring)

Run from code/ with system python3:
    python3 build_anticipation_reference.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GT_DIR = Path("/home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl/ground_truth")
OUT_DIR = Path("../data-slow/results/gf_anticipation")
VIDEO_IDS = ["005013", "020025", "027113", "035040", "041083", "044156", "066067"]
PARTICIPANTS = ["participant_a", "participant_b"]

CHUNK = 2.0
STRIDE = 2.0
MIN_WINDOW = 1.0  # drop trailing windows shorter than this (matches recognition)


def norm(x: str) -> str:
    return (x or "").strip().lower().replace(" ", "_")


def event_from_ann(a: dict) -> list[str]:
    if a["act"] == "V":
        return [norm(a["utterance_type"]), norm(a["target"])]
    return [norm(a["high_level_action"]), norm(a["low_level_action"]), norm(a["target"])]


def build(out_dir: Path) -> None:
    reference = {"anticipation": {}}
    template = {"anticipation": {}}

    for vid in VIDEO_IDS:
        gt = json.loads((GT_DIR / f"{vid}_L_mosaic.json").read_text())
        anns = [a for a in gt["annotations"] if a["subject"] in PARTICIPANTS]
        duration = max((a["end"] for a in anns), default=0.0)

        ref_segs, tmpl_segs = {}, {}
        idx, start = 1, 0.0
        while start < duration:
            end = min(start + CHUNK, duration)
            if end - start < MIN_WINDOW:
                break
            t_b, t_e = round(start, 3), round(end, 3)
            seg_id = f"s_{idx:04d}"

            ref_parts, tmpl_parts = {}, {}
            for p in PARTICIPANTS:
                evs = sorted(
                    (a for a in anns if a["subject"] == p and t_b <= a["start"] < t_e),
                    key=lambda a: a["start"],
                )
                ref_parts[p] = {"events": [event_from_ann(a) for a in evs]}
                tmpl_parts[p] = {"hypotheses": [{"events": []}]}

            ref_segs[seg_id] = {"t_b": t_b, "t_e": t_e, "participants": ref_parts}
            tmpl_segs[seg_id] = {"t_b": t_b, "t_e": t_e, "participants": tmpl_parts}
            idx += 1
            start += STRIDE

        reference["anticipation"][vid] = ref_segs
        template["anticipation"][vid] = tmpl_segs
        n_ev = sum(len(s["participants"][p]["events"]) for s in ref_segs.values() for p in PARTICIPANTS)
        print(f"  {vid}: {len(ref_segs)} segments, {n_ev} reference events")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference.json").write_text(json.dumps(reference))
    (out_dir / "template.json").write_text(json.dumps(template))
    print(f"\nWrote:\n  {out_dir / 'template.json'}\n  {out_dir / 'reference.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    build(Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
