"""
Egocentric recognition baseline.

Runs the recognition experiment in single-subject mode over the egocentric views:
  - E1 videos -> predict ONLY participant_a (the camera wearer)
  - E2 videos -> predict ONLY participant_b (the camera wearer)

Loads Qwen once and loops the requested videos for one view, writing one JSON per
video to data-slow/results/egocentric/{view}/{video_id}_{category}.json.
Merge E1+E2 afterwards with merge_egocentric.py, then score like the GF baseline.

Usage (from code/ directory):
    CUDA_VISIBLE_DEVICES=2 python run_egocentric_baseline.py --view e1 --category nonverbal
    CUDA_VISIBLE_DEVICES=3 python run_egocentric_baseline.py --view e2 --category verbal --video 035040
    python run_egocentric_baseline.py --view e1 --mock
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

# ── trigger registry decorators ───────────────────────────────────────────────
import src.models.gemma_vlm          # noqa: F401
import src.models.qwen_omni          # noqa: F401
import src.experiments.recognition   # noqa: F401
import src.execution.single_device   # noqa: F401

from src.core.capabilities import validate_compatibility
from src.core.registry import MODEL_REGISTRY, EXPERIMENT_REGISTRY, EXECUTION_REGISTRY
from src.core.schemas import ExperimentConfig
from src.output.recognition import save_report as save_recognition_report

from main import predictions_to_recognition_report


BASE_CONFIG = Path("configs/experiments/recognition_verbal_qwen.yaml")
DATA_ROOT   = Path("../data/udiva_hhoi")
OUTPUT_ROOT = Path("../data-slow/results/egocentric")

# Which subject each egocentric view records.
VIEW_SUBJECT = {"e1": "participant_a", "e2": "participant_b"}
VIEW_DIR     = {"e1": "E1", "e2": "E2"}


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def segment_count(duration: float, chunk_duration: float, chunk_stride: float) -> int:
    count, start = 0, 0.0
    while start < duration:
        end = min(start + chunk_duration, duration)
        if end - start >= 1.0:
            count += 1
        start += chunk_stride
    return count


def build_config(base: dict, video_path: Path, video_id: str, category: str,
                 subject: str, output_path: Path) -> ExperimentConfig:
    cfg = {k: v for k, v in base.items()}
    cfg["data"] = dict(base["data"])
    cfg["data"]["video_path"] = str(video_path.resolve())
    cfg["extra"] = dict(base.get("extra", {}))
    cfg["extra"]["recognition_category"] = category
    cfg["extra"]["recognition_video_id"] = video_id
    cfg["extra"]["recognition_subject"] = subject
    cfg["extra"].pop("recognition_template_path", None)
    cfg["output"] = dict(base.get("output", {}))
    cfg["output"]["path"] = str(output_path.resolve())
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
    parser.add_argument("--view", required=True, choices=["e1", "e2"],
                        help="e1 -> participant_a, e2 -> participant_b")
    parser.add_argument("--category", default="verbal", choices=["verbal", "nonverbal"])
    parser.add_argument("--video", action="append", default=None,
                        help="Only run these video id(s). Repeatable. Default: all.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    subject = VIEW_SUBJECT[args.view]
    video_dir = (DATA_ROOT / VIEW_DIR[args.view]).resolve()
    output_dir = (OUTPUT_ROOT / args.view).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    if args.video:
        wanted = set(args.video)
        videos = [v for v in videos if v.stem in wanted]
        missing = wanted - {v.stem for v in videos}
        if missing:
            sys.exit(f"Requested video id(s) not found in {video_dir}: {sorted(missing)}")
    if args.skip_existing:
        videos = [v for v in videos
                  if not (output_dir / f"{v.stem}_{args.category}.json").exists()]
    if not videos:
        sys.exit(f"No .mp4 files to run in {video_dir} (after filters)")

    base = yaml.safe_load(BASE_CONFIG.read_text())
    chunk_duration = base["data"].get("chunk_duration", 2.0)
    chunk_stride = base["data"].get("chunk_stride", 2.0)

    video_info = []
    for vp in videos:
        dur = video_duration(vp)
        n_seg = segment_count(dur, chunk_duration, chunk_stride)
        video_info.append((vp, dur, n_seg))
    total_segments = sum(n for _, _, n in video_info)

    # ── load model once ─────────────────────────────────────────────────────────
    if args.mock:
        from tests.conftest import MockModelAdapter
        model = MockModelAdapter(fixed_label='{"events": []}')
        model.load("mock")
    else:
        model_cls = MODEL_REGISTRY.get(base["model"]["name"])
        model = model_cls()
        checkpoint = base["model"].get("checkpoint", "auto")
        quantization = base["model"].get("quantization", "none")
        print(f"Loading model: {checkpoint} (quantization={quantization})")
        model.load(checkpoint, quantization=quantization,
                   weights_path=base["model"].get("weights_path", ""),
                   **base["model"].get("load_kwargs", {}))

    experiment = EXPERIMENT_REGISTRY.get("recognition")()
    errors = validate_compatibility(model.capabilities, experiment.requirements)
    if errors:
        sys.exit("Compatibility error:\n" + "\n".join(f"  - {e}" for e in errors))

    backend = EXECUTION_REGISTRY.get(base["execution"]["backend"])()
    backend.setup(model)

    print(f"\nEgocentric {args.view} ({subject}) — {args.category} — "
          f"{len(videos)} videos, {total_segments} segments\n")

    overall_bar = tqdm(total=total_segments, unit="seg", desc="Overall", position=0)
    t_start = time.time()

    for vi, (video_path, dur, n_seg) in enumerate(video_info, 1):
        video_id = video_path.stem
        output_path = output_dir / f"{video_id}_{args.category}.json"

        video_bar = tqdm(total=n_seg, unit="seg",
                         desc=f"[{vi}/{len(videos)}] {video_id} ({dur:.0f}s)",
                         position=1, leave=False)
        restore = wrap_backend_with_progress(backend, type("_dual", (), {
            "update": lambda self, n=1: (video_bar.update(n), overall_bar.update(n))
        })())

        config = build_config(base, video_path, video_id, args.category, subject, output_path)
        predictions = experiment.run(config, model, backend)

        restore()
        video_bar.close()

        report = predictions_to_recognition_report(predictions, config)
        save_recognition_report(report, config.output_path)

        n_events = sum(len(p.extra.get("events", [])) for p in predictions)
        tqdm.write(f"  ✓ {video_id}  {n_seg} segments  {n_events} events  "
                   f"→ {output_path.name}  [{time.time() - t_start:.0f}s]")

    overall_bar.close()
    backend.teardown()
    if not args.mock:
        model.unload()

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Results in {output_dir}")


if __name__ == "__main__":
    main()
