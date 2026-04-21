"""
UDIVA Experiment Framework — Main Entry Point

Orchestrates experiment execution by loading a YAML config, resolving
components from registries, and running the pipeline end-to-end.

Usage:
    python main.py --config configs/experiments/chunk_classification.yaml
    python main.py --config configs/experiments/vqa.yaml
    python main.py --config configs/experiments/chunk_classification.yaml --mock

The --mock flag uses a deterministic mock model (no GPU/download needed)
for quick testing and development.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.core.schemas import ExperimentConfig, CanonicalPrediction
from src.core.capabilities import validate_compatibility
from src.core.registry import MODEL_REGISTRY, EXPERIMENT_REGISTRY, EXECUTION_REGISTRY
from src.output.classification import ClassificationResult, ClassificationReport, save_report as save_classification_report
from src.output.vqa import VQAResult, VQAReport, save_report as save_vqa_report

# ── Import modules to trigger registry decorators ──────────────────────────
import src.models.gemma_vlm  # noqa: F401 — registers "gemma_vlm"
import src.models.qwen_omni  # noqa: F401 — registers "qwen_omni"
import src.experiments.chunk_classification  # noqa: F401 — registers "chunk_classification"
import src.experiments.vqa  # noqa: F401 — registers "vqa"
import src.execution.single_device  # noqa: F401 — registers "single_device"


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment config from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return ExperimentConfig.from_dict(data)


def predictions_to_classification_report(
    predictions: list[CanonicalPrediction],
    video_path: str,
) -> ClassificationReport:
    """Convert canonical predictions to a classification report."""
    results = [
        ClassificationResult(
            chunk_index=p.chunk_index,
            chunk_start=p.chunk_start,
            chunk_end=p.chunk_end,
            label=p.label,
            confidence=p.confidence,
            raw_text=p.raw_text,
        )
        for p in predictions
    ]
    return ClassificationReport(video_path=video_path, results=results)


# Keep backward compat alias
predictions_to_report = predictions_to_classification_report


def predictions_to_vqa_report(
    predictions: list[CanonicalPrediction],
    video_path: str,
) -> VQAReport:
    """Convert canonical predictions to a VQA report."""
    results = [
        VQAResult(
            chunk_index=p.chunk_index,
            chunk_start=p.chunk_start,
            chunk_end=p.chunk_end,
            question=p.extra.get("question", ""),
            answer=p.raw_text,
            formatted=p.extra.get("structured"),
        )
        for p in predictions
    ]
    return VQAReport(video_path=video_path, results=results)


def run_experiment(config: ExperimentConfig, use_mock: bool = False):
    """Execute a full experiment from config.

    Args:
        config: Loaded experiment configuration.
        use_mock: If True, use a MockModelAdapter instead of real model.

    Returns:
        ClassificationReport or VQAReport depending on experiment type.
    """
    # 1. Resolve components from registries
    experiment_cls = EXPERIMENT_REGISTRY.get(config.experiment_type)
    experiment = experiment_cls()

    backend_cls = EXECUTION_REGISTRY.get(config.backend)
    backend = backend_cls()

    # 2. Load model
    if use_mock:
        from tests.conftest import MockModelAdapter
        model = MockModelAdapter(fixed_label=config.labels[0] if config.labels else "mock answer")
        model.load("mock")
    else:
        model_cls = MODEL_REGISTRY.get(config.model_name)
        model = model_cls()
        print(f"Loading model: {config.model_checkpoint} (quantization={config.quantization})")
        if config.weights_path:
            print(f"Weights cache: {config.weights_path}")
        model.load(
            config.model_checkpoint,
            quantization=config.quantization,
            weights_path=config.weights_path,
        )

    # 3. Validate compatibility (hard error on mismatch)
    errors = validate_compatibility(model.capabilities, experiment.requirements)
    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(
            f"Model '{config.model_name}' is incompatible with experiment "
            f"'{config.experiment_type}':\n{error_msg}"
        )

    # 4. Setup execution backend
    backend.setup(model)

    # 5. Run experiment
    print(f"Running experiment: {config.experiment_type}")
    print(f"Video: {config.video_path}")
    print(f"Chunks: {config.chunk_duration}s duration, {config.chunk_stride}s stride")
    print()

    predictions = experiment.run(config, model, backend)

    # 6. Build and save report (type-specific)
    if config.experiment_type == "vqa":
        report = predictions_to_vqa_report(predictions, config.video_path)
        save_vqa_report(report, config.output_path)
        _print_vqa_summary(report)
    else:
        report = predictions_to_classification_report(predictions, config.video_path)
        save_classification_report(report, config.output_path)
        _print_classification_summary(report)

    print(f"\nResults saved to: {config.output_path}")

    # 7. Cleanup
    backend.teardown()
    if not use_mock:
        model.unload()

    return report


def _print_classification_summary(report: ClassificationReport) -> None:
    """Print classification-specific summary."""
    summary = report.summary()
    print(f"\n--- Classification Summary ---")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Avg confidence: {summary['avg_confidence']:.2f}")
    print()
    for r in report.results:
        print(f"  Chunk {r.chunk_index} [{r.chunk_start:.1f}s-{r.chunk_end:.1f}s]: "
              f"{r.label} (conf={r.confidence:.2f})")


def _print_vqa_summary(report: VQAReport) -> None:
    """Print VQA-specific summary."""
    summary = report.summary()
    print(f"\n--- VQA Summary ---")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Formatted answers: {summary['formatted_answers']}")
    print(f"Free-text answers: {summary['free_text_answers']}")
    print()
    for r in report.results:
        question_preview = r.question[:60] + "..." if len(r.question) > 60 else r.question
        answer_preview = r.answer[:80] + "..." if len(r.answer) > 80 else r.answer
        print(f"  Chunk {r.chunk_index} [{r.chunk_start:.1f}s-{r.chunk_end:.1f}s]:")
        print(f"    Q: {question_preview}")
        print(f"    A: {answer_preview}")
        if r.formatted:
            print(f"    Formatted: {r.formatted}")


def main():
    parser = argparse.ArgumentParser(description="UDIVA Experiment Framework")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--mock", action="store_true", help="Use mock model (no GPU/download needed)")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config, use_mock=args.mock)


if __name__ == "__main__":
    main()

