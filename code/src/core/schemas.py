"""
Canonical data types shared across all layers of the experiment framework.

These dataclasses define the contracts between layers so that experiments,
models, and execution backends can be composed independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoSample:
    """Full video metadata — represents one source video file."""
    path: Path
    duration: float  # seconds
    fps: float
    width: int
    height: int
    has_audio: bool = False


@dataclass
class VideoChunk:
    """A temporal segment of a video."""
    video: VideoSample
    index: int
    start: float  # seconds
    end: float    # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ModalityBundle:
    """
    All modalities prepared for a model from a single chunk.

    Fields are optional because not every model needs every modality.
    The data pipeline fills only what the model's capabilities require.
    """
    frames: Any | None = None       # torch.Tensor (T, C, H, W) float32 [0,1]
    text_prompt: str | None = None   # text prompt for the model
    audio: Any | None = None         # audio tensor or bytes
    chunk: VideoChunk | None = None  # reference back to source chunk


@dataclass
class ModelRequest:
    """What the experiment layer asks the model to do."""
    task: str                          # e.g. "classify", "predict_labels", "build_graph"
    labels: list[str] = field(default_factory=list)
    prompt_template: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawModelOutput:
    """Model-specific raw output before canonical normalization."""
    text: str | None = None             # generated text (for LLM/VLM)
    embeddings: Any | None = None       # latent representation (for JEPA-style)
    logits: Any | None = None           # classification logits
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalPrediction:
    """
    Normalized output used by the experiment layer.

    Every model output is converted into this shared schema through
    a postprocessor, regardless of the model's native output format.
    """
    chunk_index: int
    chunk_start: float
    chunk_end: float
    label: str = ""
    confidence: float = 0.0
    entities: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration loaded from a YAML file to drive an experiment run."""
    # Experiment
    experiment_type: str = ""
    labels: list[str] = field(default_factory=list)

    # VQA / Prompt configuration
    prompt_file: str = ""               # path to .py with build_prompt()
    system_prompt_file: str = ""        # path to .py with build_system_prompt()
    output_schema: dict[str, Any] | None = None  # optional structured output schema

    # Model
    model_name: str = ""
    model_checkpoint: str = ""
    weights_path: str = ""       # local path for model weights cache
    quantization: str = "none"   # "none", "4bit", "8bit"

    # Data
    video_path: str = ""
    chunk_duration: float = 3.0
    chunk_stride: float = 3.0
    fps: float = 2.0
    resize: tuple[int, int] | None = None

    # Execution
    backend: str = "single_device"

    # Output
    output_format: str = "json"
    output_path: str = "results/output.json"

    # Extra
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Build config from a nested dict (e.g. parsed YAML)."""
        exp = data.get("experiment", {})
        model = data.get("model", {})
        data_cfg = data.get("data", {})
        execution = data.get("execution", {})
        output = data.get("output", {})

        resize_raw = data_cfg.get("resize")
        resize = tuple(resize_raw) if resize_raw else None

        return cls(
            experiment_type=exp.get("type", ""),
            labels=exp.get("labels", []),
            prompt_file=exp.get("prompt_file", ""),
            system_prompt_file=exp.get("system_prompt_file", ""),
            output_schema=exp.get("output_schema"),
            model_name=model.get("name", ""),
            model_checkpoint=model.get("checkpoint", ""),
            weights_path=model.get("weights_path", ""),
            quantization=model.get("quantization", "none"),
            video_path=data_cfg.get("video_path", ""),
            chunk_duration=data_cfg.get("chunk_duration", 3.0),
            chunk_stride=data_cfg.get("chunk_stride", 3.0),
            fps=data_cfg.get("fps", 2.0),
            resize=resize,
            backend=execution.get("backend", "single_device"),
            output_format=output.get("format", "json"),
            output_path=output.get("path", "results/output.json"),
            extra=data.get("extra", {}),
        )

