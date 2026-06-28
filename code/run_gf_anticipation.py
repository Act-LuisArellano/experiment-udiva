"""
Anticipation baseline over the GF videos, both views.

Windows come from data-slow/samples_gt/anticipation_samples_gt.json (which is also
the official reference). The model observes [max(0, t_b-60), t_b] and predicts
each participant's events in [t_b, t_e].

Views:
  --view exo : exocentric GF, one call predicts BOTH participants.
  --view e1  : egocentric E1, predicts participant_a only (camera wearer).
  --view e2  : egocentric E2, predicts participant_b only.

Writes per-video shards; merge with merge_anticipation.py (exo, or ego = e1+e2).

Usage (from code/):
    CUDA_VISIBLE_DEVICES=0 python run_gf_anticipation.py --view exo
    CUDA_VISIBLE_DEVICES=1 python run_gf_anticipation.py --view e1 --video 005013
    python run_gf_anticipation.py --view exo --mock
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

# ── trigger registry decorators ───────────────────────────────────────────────
import src.models.gemma_vlm          # noqa: F401
import src.models.qwen_omni          # noqa: F401
import src.experiments.recognition   # noqa: F401
import src.experiments.anticipation  # noqa: F401
import src.execution.single_device   # noqa: F401

from src.core.capabilities import validate_compatibility
from src.core.registry import MODEL_REGISTRY, EXPERIMENT_REGISTRY, EXECUTION_REGISTRY
from src.core.schemas import ExperimentConfig
from src.experiments.anticipation import PARTICIPANTS

BASE_CONFIG = Path("configs/experiments/anticipation_qwen.yaml")
DATA_ROOT   = Path("../data/udiva_hhoi")
OUTPUT_ROOT = Path("../data-slow/results/gf_anticipation")

# view -> (video subdir, predicted subject "" = both, shard dir under OUTPUT_ROOT)
VIEW_CFG = {
    "exo": ("GF", "",             "exo/shards"),
    "e1":  ("E1", "participant_a", "ego/e1/shards"),
    "e2":  ("E2", "participant_b", "ego/e2/shards"),
}


def events_to_arrays(events: list[dict]) -> list[list[str]]:
    arrays = []
    for ev in events:
        if ev.get("type") == "verbal":
            arrays.append([ev["utterance_type"], ev["target"]])
        else:
            arrays.append([ev["highlevel_action"], ev["lowlevel_action"], ev["target"]])
    return arrays


def build_config(base: dict, video_path: Path, video_id: str, subject: str) -> ExperimentConfig:
    cfg = {k: v for k, v in base.items()}
    cfg["data"] = dict(base["data"])
    cfg["data"]["video_path"] = str(video_path.resolve())
    cfg["extra"] = dict(base.get("extra", {}))
    cfg["extra"]["anticipation_video_id"] = video_id
    cfg["extra"]["anticipation_subject"] = subject
    cfg["extra"]["anticipation_template_path"] = str(
        Path(base["extra"]["anticipation_template_path"]).resolve())
    return ExperimentConfig.from_dict(cfg)


def wrap_backend_with_progress(backend, pbar: tqdm):
    original = backend.run_model

    def run_model_tracked(model, bundle, request):
        result = original(model, bundle, request)
        pbar.update(1)
        return result

    backend.run_model = run_model_tracked
    return lambda: setattr(backend, "run_model", original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True, choices=["exo", "e1", "e2"])
    parser.add_argument("--video", action="append", default=None,
                        help="Only run these video id(s), e.g. 005013. Repeatable.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    sub_dir, subject, shard_rel = VIEW_CFG[args.view]
    video_dir = (DATA_ROOT / sub_dir).resolve()
    shard_dir = (OUTPUT_ROOT / shard_rel).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(BASE_CONFIG.read_text())
    windows = json.loads(Path(base["extra"]["anticipation_template_path"]).read_text())["anticipation"]
    video_ids = sorted(windows.keys())
    if args.video:
        wanted = set(args.video)
        video_ids = [v for v in video_ids if v in wanted]
        missing = wanted - set(video_ids)
        if missing:
            sys.exit(f"Requested video id(s) not in windows: {sorted(missing)}")
    if args.skip_existing:
        video_ids = [v for v in video_ids if not (shard_dir / f"{v}.json").exists()]
    if not video_ids:
        sys.exit("No videos to run (after filters).")

    seg_counts = {v: len(windows[v]) for v in video_ids}
    total = sum(seg_counts.values())
    pred_subjects = (subject,) if subject else PARTICIPANTS

    # ── load model once ─────────────────────────────────────────────────────────
    if args.mock:
        from tests.conftest import MockModelAdapter
        label = ('{"participant_a": [], "participant_b": []}' if not subject
                 else f'{{"{subject}": []}}')
        model = MockModelAdapter(fixed_label=label)
        model.load("mock")
    else:
        model_cls = MODEL_REGISTRY.get(base["model"]["name"])
        model = model_cls()
        print(f"Loading model: {base['model']['checkpoint']} "
              f"(quantization={base['model'].get('quantization', 'none')})")
        model.load(base["model"].get("checkpoint", "auto"),
                   quantization=base["model"].get("quantization", "none"),
                   weights_path=base["model"].get("weights_path", ""),
                   **base["model"].get("load_kwargs", {}))

    experiment = EXPERIMENT_REGISTRY.get("anticipation")()
    errors = validate_compatibility(model.capabilities, experiment.requirements)
    if errors:
        sys.exit("Compatibility error:\n" + "\n".join(f"  - {e}" for e in errors))

    backend = EXECUTION_REGISTRY.get(base["execution"]["backend"])()
    backend.setup(model)

    print(f"\nAnticipation [{args.view}] subject={subject or 'both'} — "
          f"{len(video_ids)} videos, {total} windows\n")
    overall_bar = tqdm(total=total, unit="win", desc="Overall", position=0)
    t_start = time.time()

    for vi, video_id in enumerate(video_ids, 1):
        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            matches = list(video_dir.glob(f"{video_id}*.mp4"))
            if not matches:
                sys.exit(f"Video file for {video_id} not found in {video_dir}")
            video_path = matches[0]

        video_bar = tqdm(total=seg_counts[video_id], unit="win",
                         desc=f"[{vi}/{len(video_ids)}] {video_id}", position=1, leave=False)
        restore = wrap_backend_with_progress(backend, type("_dual", (), {
            "update": lambda self, n=1: (video_bar.update(n), overall_bar.update(n))
        })())

        config = build_config(base, video_path, video_id, subject)
        predictions = experiment.run(config, model, backend)

        restore()
        video_bar.close()

        segs = {}
        for p in predictions:
            parts = {
                subj: {"hypotheses": [{"events": events_to_arrays(p.extra["events"].get(subj, []))}]}
                for subj in pred_subjects
            }
            segs[p.extra["segment_id"]] = {"t_b": p.extra["t_b"], "t_e": p.extra["t_e"],
                                           "participants": parts}
        shard_path = shard_dir / f"{video_id}.json"
        shard_path.write_text(json.dumps({"anticipation": {video_id: segs}}))

        n_ev = sum(len(events_to_arrays(p.extra["events"].get(s, [])))
                   for p in predictions for s in pred_subjects)
        tqdm.write(f"  ✓ {video_id}  {seg_counts[video_id]} windows  {n_ev} events  "
                   f"→ {shard_path.name}  [{time.time() - t_start:.0f}s]")

    overall_bar.close()
    backend.teardown()
    if not args.mock:
        model.unload()
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Shards in {shard_dir}\n"
          f"Merge with: python merge_anticipation.py --view "
          f"{'exo' if args.view == 'exo' else 'ego'}")


if __name__ == "__main__":
    main()
