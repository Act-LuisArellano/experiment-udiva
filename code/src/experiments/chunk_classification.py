"""
Chunk classification experiment.

Classifies each video chunk into one of a set of predefined labels
using a VLM or classification model.
"""

from __future__ import annotations

from typing import Any

from src.core.capabilities import TaskRequirements
from src.core.interfaces import BaseExperiment, BaseExecutionBackend, BaseModelAdapter
from src.core.registry import EXPERIMENT_REGISTRY
from src.core.schemas import (
    CanonicalPrediction,
    ExperimentConfig,
    ModelRequest,
    RawModelOutput,
    VideoChunk,
)
from src.data.pipeline import DataPipeline


@EXPERIMENT_REGISTRY.register("chunk_classification")
class ChunkClassificationExperiment(BaseExperiment):
    """Classify each video chunk into one of a predefined set of labels."""

    @property
    def requirements(self) -> TaskRequirements:
        return TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            output_type="label",
        )

    def prepare_request(self, chunk: VideoChunk, config: ExperimentConfig) -> ModelRequest:
        """Build a classification request for a single chunk."""
        labels_str = ", ".join(config.labels)
        prompt = (
            f"Watch this video segment carefully (from {chunk.start:.1f}s to {chunk.end:.1f}s). "
            f"Classify the primary activity into exactly one of these labels: {labels_str}. "
            f"Respond with ONLY the label name, nothing else."
        )
        return ModelRequest(
            task="classify",
            labels=config.labels,
            prompt_template=prompt,
        )

    def postprocess(
        self,
        raw: RawModelOutput,
        chunk: VideoChunk,
        config: ExperimentConfig,
    ) -> CanonicalPrediction:
        """Parse model output text to extract a valid label."""
        text = (raw.text or "").strip().lower()
        matched_label = "unknown"

        # Try exact match first, then substring match
        for label in config.labels:
            if label.lower() == text:
                matched_label = label
                break
        else:
            # Substring match (for verbose responses)
            for label in config.labels:
                if label.lower() in text:
                    matched_label = label
                    break

        return CanonicalPrediction(
            chunk_index=chunk.index,
            chunk_start=chunk.start,
            chunk_end=chunk.end,
            label=matched_label,
            confidence=1.0 if matched_label != "unknown" else 0.0,
            raw_text=raw.text or "",
        )

    def run(
        self,
        config: ExperimentConfig,
        model: BaseModelAdapter,
        backend: BaseExecutionBackend,
    ) -> list[CanonicalPrediction]:
        """Execute the full classification pipeline.

        1. Load video
        2. Chunk it
        3. For each chunk: build modality bundle → predict → postprocess
        4. Return all predictions
        """
        pipeline = DataPipeline()

        # Load video and chunk it
        video = pipeline.load_video(config.video_path)
        chunks = pipeline.chunk(video, duration=config.chunk_duration, stride=config.chunk_stride)

        predictions: list[CanonicalPrediction] = []

        for chunk in chunks:
            # Prepare the model request
            request = self.prepare_request(chunk, config)

            # Build modality bundle (capability-aware)
            bundle = pipeline.build_modality_bundle(
                chunk,
                model.capabilities,
                request,
                fps=config.fps,
                resize=config.resize,
            )

            # Run inference through the execution backend
            raw_output = backend.run_model(model, bundle, request)

            # Postprocess into canonical prediction
            prediction = self.postprocess(raw_output, chunk, config)
            predictions.append(prediction)

        return predictions
