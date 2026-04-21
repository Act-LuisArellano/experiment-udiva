"""
VQA experiment.

Runs Visual Question Answering on video chunks using a prompt built
from a user-defined Python file. Supports both free-text and
structured JSON output modes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

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


def _load_prompt_function(file_path: str, function_name: str) -> Callable | None:
    """Dynamically load a function from a Python file.

    Args:
        file_path: Path to the .py file.
        function_name: Name of the function to load.

    Returns:
        The loaded function, or None if the file or function doesn't exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("prompt_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    func = getattr(module, function_name, None)
    if func is None:
        raise AttributeError(
            f"Function '{function_name}' not found in {file_path}. "
            f"Available: {[n for n in dir(module) if not n.startswith('_')]}"
        )
    return func


@EXPERIMENT_REGISTRY.register("vqa")
class VQAExperiment(BaseExperiment):
    """Visual Question Answering experiment.

    Asks a prompt (built from a user-defined Python file) per video chunk
    and collects the model's answers. Supports both free-text and
    structured JSON output based on the config's output_schema.
    """

    @property
    def requirements(self) -> TaskRequirements:
        return TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            needs_embedding=False,
            needs_classification=False,
            output_type="text",
        )

    def prepare_request(
        self,
        chunk: VideoChunk,
        config: ExperimentConfig,
        prompt_fn: Callable | None = None,
    ) -> ModelRequest:
        """Build a VQA request for a single chunk.

        Args:
            chunk: The video chunk.
            config: Experiment configuration.
            prompt_fn: The loaded build_prompt function. If None, uses a
                       default prompt.
        """
        if prompt_fn is not None:
            prompt = prompt_fn(
                chunk_start=chunk.start,
                chunk_end=chunk.end,
                chunk_index=chunk.index,
                output_schema=config.output_schema,
                labels=config.labels,
            )
        else:
            prompt = (
                f"Watch this video segment carefully "
                f"(from {chunk.start:.1f}s to {chunk.end:.1f}s). "
                f"Describe what you see."
            )

        # Load system prompt if available
        system_prompt = ""
        if hasattr(self, "_system_prompt_fn") and self._system_prompt_fn is not None:
            system_prompt = self._system_prompt_fn(
                output_schema=config.output_schema,
            )

        return ModelRequest(
            task="vqa",
            labels=config.labels,
            prompt_template=prompt,
            extra={"system_prompt": system_prompt} if system_prompt else {},
        )

    def postprocess(
        self,
        raw: RawModelOutput,
        chunk: VideoChunk,
        config: ExperimentConfig,
    ) -> CanonicalPrediction:
        """Parse model output into a canonical prediction.

        If output_schema is defined, attempts to parse JSON from the response.
        Structured fields are stored in extra["structured"].
        """
        text = (raw.text or "").strip()
        extra: dict[str, Any] = {}

        if config.output_schema:
            # Try to parse structured JSON output
            try:
                parsed = json.loads(text)
                extra["structured"] = parsed
            except json.JSONDecodeError:
                # Try to extract JSON from the text (model might add surrounding text)
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    try:
                        parsed = json.loads(text[json_start:json_end])
                        extra["structured"] = parsed
                    except json.JSONDecodeError:
                        extra["parse_error"] = "Could not parse structured output"
                else:
                    extra["parse_error"] = "No JSON found in response"

        return CanonicalPrediction(
            chunk_index=chunk.index,
            chunk_start=chunk.start,
            chunk_end=chunk.end,
            label="",  # VQA is open-ended, no label classification
            confidence=1.0,
            raw_text=text,
            extra=extra,
        )

    def run(
        self,
        config: ExperimentConfig,
        model: BaseModelAdapter,
        backend: BaseExecutionBackend,
    ) -> list[CanonicalPrediction]:
        """Execute the full VQA pipeline.

        1. Load prompt builder from config.prompt_file
        2. Load video and chunk it
        3. For each chunk: build prompt → predict → postprocess
        4. Return all predictions
        """
        # Load prompt functions
        prompt_fn = None
        self._system_prompt_fn = None

        if config.prompt_file:
            prompt_fn = _load_prompt_function(config.prompt_file, "build_prompt")

        if config.system_prompt_file:
            self._system_prompt_fn = _load_prompt_function(
                config.system_prompt_file, "build_system_prompt"
            )
        elif config.prompt_file:
            # Try to load build_system_prompt from the same file
            try:
                self._system_prompt_fn = _load_prompt_function(
                    config.prompt_file, "build_system_prompt"
                )
            except AttributeError:
                pass  # No system prompt function defined — that's fine

        pipeline = DataPipeline()

        # Load video and chunk it
        video = pipeline.load_video(config.video_path)
        chunks = pipeline.chunk(
            video, duration=config.chunk_duration, stride=config.chunk_stride
        )

        predictions: list[CanonicalPrediction] = []

        for chunk in chunks:
            # Prepare the model request with user-defined prompt
            request = self.prepare_request(chunk, config, prompt_fn)

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
            prediction.extra["question"] = request.prompt_template
            predictions.append(prediction)

        return predictions
