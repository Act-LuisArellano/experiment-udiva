"""
Causal QA baseline over the GF videos.

Loads Qwen once and runs the causal experiment for every effect in the GT spec,
writing a single official-format submission JSON to
data-slow/results/gf_causal/submission.json. Per-video shards allow safe
parallel runs (one model instance per GPU), merged at the end.

Usage (from code/ directory):
    CUDA_VISIBLE_DEVICES=2 python run_gf_causal.py
    CUDA_VISIBLE_DEVICES=2 python run_gf_causal.py --video 005013.mp4
    python run_gf_causal.py --mock
"""

from __future__ import annotations

import argparse
import glob
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
import src.experiments.causal        # noqa: F401
import src.execution.single_device   # noqa: F401

from src.core.capabilities import validate_compatibility
from src.core.registry import MODEL_REGISTRY, EXPERIMENT_REGISTRY, EXECUTION_REGISTRY
from src.core.schemas import ExperimentConfig

BASE_CONFIG = Path("configs/experiments/causal_qwen.yaml")
VIDEO_DIR   = Path("../data/udiva_hhoi/GF")
OUTPUT_DIR  = Path("../data-slow/results/gf_causal")


def build_config(base: dict, video_path: Path, video_id: str) -> ExperimentConfig:
    cfg = {k: v for k, v in base.items()}
    cfg["data"] = dict(base["data"])
    cfg["data"]["video_path"] = str(video_path.resolve())
    cfg["extra"] = dict(base.get("extra", {}))
    cfg["extra"]["causal_video_id"] = video_id
    cfg["extra"]["causal_spec_path"] = str(Path(base["extra"]["causal_spec_path"]).resolve())
    return ExperimentConfig.from_dict(cfg)


def merge_shards(shard_dir: Path, out_path: Path) -> None:
    merged = {"causal": {}}
    for f in sorted(shard_dir.glob("*.json")):
        merged["causal"].update(json.loads(f.read_text()).get("causal", {}))
    out_path.write_text(json.dumps(merged))
    print(f"Merged {len(list(shard_dir.glob('*.json')))} shard(s) -> {out_path} "
          f"({len(merged['causal'])} videos)")


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
    parser.add_argument("--video", action="append", default=None,
                        help="Only run these video id(s), e.g. 005013.mp4. Repeatable.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR.resolve()
    shard_dir = output_dir / "shards"

    if args.merge_only:
        merge_shards(shard_dir, output_dir / "submission.json")
        return

    base = yaml.safe_load(BASE_CONFIG.read_text())
    spec = json.loads(Path(base["extra"]["causal_spec_path"]).read_text())
    segments = spec[next(iter(spec))]
    video_ids = sorted(segments.keys())
    if args.video:
        wanted = set(args.video)
        video_ids = [v for v in video_ids if v in wanted]
        missing = wanted - set(video_ids)
        if missing:
            sys.exit(f"Requested video id(s) not in spec: {sorted(missing)}")
    if args.skip_existing:
        video_ids = [v for v in video_ids if not (shard_dir / f"{Path(v).stem}.json").exists()]
    if not video_ids:
        sys.exit("No videos to run (after filters).")

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    video_dir = VIDEO_DIR.resolve()

    seg_counts = {v: len(segments[v]) for v in video_ids}
    total_effects = sum(seg_counts.values())

    # ── load model once ─────────────────────────────────────────────────────────
    if args.mock:
        from tests.conftest import MockModelAdapter
        model = MockModelAdapter(fixed_label='{"option": "A", "timestamp": 1.0}')
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

    experiment = EXPERIMENT_REGISTRY.get("causal")()
    errors = validate_compatibility(model.capabilities, experiment.requirements)
    if errors:
        sys.exit("Compatibility error:\n" + "\n".join(f"  - {e}" for e in errors))

    backend = EXECUTION_REGISTRY.get(base["execution"]["backend"])()
    backend.setup(model)

    print(f"\nCausal QA — {len(video_ids)} videos, {total_effects} effects\n")
    overall_bar = tqdm(total=total_effects, unit="eff", desc="Overall", position=0)
    t_start = time.time()

    for vi, video_id in enumerate(video_ids, 1):
        video_path = video_dir / video_id
        if not video_path.exists():
            matches = list(video_dir.glob(f"{Path(video_id).stem}*.mp4"))
            if not matches:
                sys.exit(f"Video file for {video_id} not found in {video_dir}")
            video_path = matches[0]

        video_bar = tqdm(total=seg_counts[video_id], unit="eff",
                         desc=f"[{vi}/{len(video_ids)}] {video_id}", position=1, leave=False)
        restore = wrap_backend_with_progress(backend, type("_dual", (), {
            "update": lambda self, n=1: (video_bar.update(n), overall_bar.update(n))
        })())

        config = build_config(base, video_path, video_id)
        predictions = experiment.run(config, model, backend)

        restore()
        video_bar.close()

        recs = {p.extra["effect_id"]: {
                    "predicted_option": p.extra["predicted_option"],
                    "predicted_cause_timestamp": p.extra["predicted_cause_timestamp"],
                } for p in predictions}
        shard_path = shard_dir / f"{Path(video_id).stem}.json"
        shard_path.write_text(json.dumps({"causal": {video_id: recs}}))

        n_opt = sum(1 for p in predictions if p.extra["predicted_option"])
        tqdm.write(f"  ✓ {video_id}  {len(recs)} effects  {n_opt} options predicted  "
                   f"→ {shard_path.name}  [{time.time() - t_start:.0f}s]")

    overall_bar.close()
    backend.teardown()
    if not args.mock:
        model.unload()

    merge_shards(shard_dir, output_dir / "submission.json")
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
