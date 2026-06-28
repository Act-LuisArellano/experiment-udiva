"""
Batch recognition baseline over all videos in data/udiva_hhoi/GF/.

Loads Qwen once and runs the recognition experiment for every .mp4,
writing one JSON output per video to data-slow/results/gf_baseline/.

Usage (from code/ directory):
    CUDA_VISIBLE_DEVICES=2,3,4 python run_gf_baseline.py
    CUDA_VISIBLE_DEVICES=2,3,4 python run_gf_baseline.py --category nonverbal
    CUDA_VISIBLE_DEVICES=2,3,4 python run_gf_baseline.py --mock
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
import src.experiments.chunk_classification  # noqa: F401
import src.experiments.recognition   # noqa: F401
import src.experiments.vqa           # noqa: F401
import src.execution.single_device   # noqa: F401

from src.core.capabilities import validate_compatibility
from src.core.registry import MODEL_REGISTRY, EXPERIMENT_REGISTRY, EXECUTION_REGISTRY
from src.core.schemas import ExperimentConfig
from src.output.recognition import save_report as save_recognition_report

from main import predictions_to_recognition_report


BASE_CONFIG = Path("configs/experiments/recognition_verbal_qwen.yaml")
VIDEO_DIR   = Path("../data/udiva_hhoi/GF")
OUTPUT_DIR  = Path("../data-slow/results/gf_baseline")


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


def build_config(base: dict, video_path: Path, video_id: str,
                 category: str, output_path: Path) -> ExperimentConfig:
    cfg = {k: v for k, v in base.items()}
    cfg["data"] = dict(base["data"])
    cfg["data"]["video_path"] = str(video_path.resolve())
    cfg["extra"] = dict(base.get("extra", {}))
    cfg["extra"]["recognition_category"] = category
    cfg["extra"]["recognition_video_id"] = video_id
    cfg["extra"].pop("recognition_template_path", None)
    cfg["output"] = dict(base.get("output", {}))
    cfg["output"]["path"] = str(output_path.resolve())
    return ExperimentConfig.from_dict(cfg)


def wrap_backend_with_progress(backend, pbar: tqdm):
    """Monkey-patch backend.run_model to tick the progress bar on each segment."""
    original = backend.run_model

    def run_model_tracked(model, bundle, request):
        result = original(model, bundle, request)
        pbar.update(1)
        return result

    backend.run_model = run_model_tracked
    return lambda: setattr(backend, "run_model", original)  # restore fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="verbal", choices=["verbal", "nonverbal"])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--video", action="append", default=None,
        help="Only run these video id(s) (stem, e.g. 035040). Repeatable. Default: all.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip videos whose output JSON already exists.",
    )
    args = parser.parse_args()

    video_dir  = VIDEO_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    if args.video:
        wanted = set(args.video)
        videos = [v for v in videos if v.stem in wanted]
        missing = wanted - {v.stem for v in videos}
        if missing:
            sys.exit(f"Requested video id(s) not found in {video_dir}: {sorted(missing)}")
    if args.skip_existing:
        videos = [
            v for v in videos
            if not (output_dir / f"{v.stem}_{args.category}.json").exists()
        ]
    if not videos:
        sys.exit(f"No .mp4 files to run in {video_dir} (after filters)")

    base           = yaml.safe_load(BASE_CONFIG.read_text())
    chunk_duration = base["data"].get("chunk_duration", 2.0)
    chunk_stride   = base["data"].get("chunk_stride", 2.0)

    # Pre-compute segment counts for accurate progress bars
    video_info = []
    for vp in videos:
        dur   = video_duration(vp)
        n_seg = segment_count(dur, chunk_duration, chunk_stride)
        video_info.append((vp, dur, n_seg))

    total_segments = sum(n for _, _, n in video_info)

    # ── load model once ───────────────────────────────────────────────────────
    if args.mock:
        from tests.conftest import MockModelAdapter
        model = MockModelAdapter(fixed_label='{"events": []}')
        model.load("mock")
    else:
        model_cls    = MODEL_REGISTRY.get(base["model"]["name"])
        model        = model_cls()
        checkpoint   = base["model"].get("checkpoint", "auto")
        quantization = base["model"].get("quantization", "none")
        weights_path = base["model"].get("weights_path", "")
        load_kwargs  = base["model"].get("load_kwargs", {})
        print(f"Loading model: {checkpoint} (quantization={quantization})")
        model.load(checkpoint, quantization=quantization,
                   weights_path=weights_path, **load_kwargs)

    experiment_cls = EXPERIMENT_REGISTRY.get("recognition")
    experiment     = experiment_cls()

    errors = validate_compatibility(model.capabilities, experiment.requirements)
    if errors:
        sys.exit("Compatibility error:\n" + "\n".join(f"  - {e}" for e in errors))

    backend_cls = EXECUTION_REGISTRY.get(base["execution"]["backend"])
    backend     = backend_cls()
    backend.setup(model)

    # ── loop videos ──────────────────────────────────────────────────────────
    print(f"\nRunning {args.category} recognition — "
          f"{len(videos)} videos, {total_segments} segments total\n")

    overall_bar = tqdm(total=total_segments, unit="seg", desc="Overall", position=0)
    t_start     = time.time()

    for video_idx, (video_path, dur, n_seg) in enumerate(video_info, 1):
        video_id    = video_path.stem
        output_path = output_dir / f"{video_id}_{args.category}.json"

        video_bar = tqdm(
            total=n_seg, unit="seg",
            desc=f"[{video_idx}/{len(videos)}] {video_id} ({dur:.0f}s)",
            position=1, leave=False,
        )

        restore = wrap_backend_with_progress(
            backend,
            pbar=type("_dual", (), {
                "update": lambda self, n=1: (video_bar.update(n), overall_bar.update(n))
            })(),
        )

        config      = build_config(base, video_path, video_id, args.category, output_path)
        predictions = experiment.run(config, model, backend)

        restore()
        video_bar.close()

        report = predictions_to_recognition_report(predictions, config)
        save_recognition_report(report, config.output_path)

        n_events = sum(len(p.extra.get("events", [])) for p in predictions)
        elapsed  = time.time() - t_start
        tqdm.write(
            f"  ✓ {video_id}  {n_seg} segments  {n_events} events  "
            f"saved → {output_path.name}  [{elapsed:.0f}s elapsed]"
        )

    overall_bar.close()
    backend.teardown()
    if not args.mock:
        model.unload()

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.1f} min. Results in {output_dir}")


if __name__ == "__main__":
    main()
