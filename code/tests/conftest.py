"""
Shared pytest fixtures for the UDIVA experiment framework tests.

Provides mock models, sample data, and config fixtures so unit tests
run instantly without GPU or model downloads.
"""

import pytest
import torch
from pathlib import Path
from typing import Any

from src.core.schemas import (
    VideoSample,
    VideoChunk,
    ModalityBundle,
    ModelRequest,
    RawModelOutput,
    CanonicalPrediction,
    ExperimentConfig,
)
from src.core.capabilities import ModelCapabilities
from src.core.interfaces import BaseModelAdapter


# ── Paths ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # UDIVA-REFACTOR/
VIDEO_PATH = REPO_ROOT / "data-fast" / "videos" / "vid_00500-00660-remote.mp4"


@pytest.fixture
def sample_video_path() -> Path:
    """Path to the real 6.4-second test video."""
    return VIDEO_PATH


# ── Synthetic data ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_video_sample() -> VideoSample:
    """A VideoSample pointing to the test video."""
    return VideoSample(
        path=VIDEO_PATH,
        duration=6.4,
        fps=5.0,
        width=1280,
        height=720,
        has_audio=True,
    )


@pytest.fixture
def sample_chunk(sample_video_sample) -> VideoChunk:
    """A single 3-second chunk from the test video."""
    return VideoChunk(
        video=sample_video_sample,
        index=0,
        start=0.0,
        end=3.0,
    )


@pytest.fixture
def sample_frames() -> torch.Tensor:
    """Synthetic frames tensor: 6 frames at 224x224."""
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def sample_modality_bundle(sample_frames, sample_chunk) -> ModalityBundle:
    """A ModalityBundle with frames and a text prompt."""
    return ModalityBundle(
        frames=sample_frames,
        text_prompt="Classify this video chunk into one of: talking, building, idle",
        audio=None,
        chunk=sample_chunk,
    )


@pytest.fixture
def sample_model_request() -> ModelRequest:
    """A classification model request with 3 labels."""
    return ModelRequest(
        task="classify",
        labels=["talking", "building", "idle"],
        prompt_template="Classify this video chunk into one of: {labels}. Respond with only the label.",
    )


@pytest.fixture
def sample_raw_output() -> RawModelOutput:
    """Mock model output simulating a VLM text response."""
    return RawModelOutput(
        text="talking",
        embeddings=None,
        logits=None,
    )


@pytest.fixture
def sample_canonical_prediction() -> CanonicalPrediction:
    """A well-formed canonical prediction."""
    return CanonicalPrediction(
        chunk_index=0,
        chunk_start=0.0,
        chunk_end=3.0,
        label="talking",
        confidence=0.85,
        raw_text="talking",
    )


# ── Mock model ─────────────────────────────────────────────────────────────

class MockModelAdapter(BaseModelAdapter):
    """Deterministic mock model for unit testing.

    Always returns a fixed label so tests are fast and reproducible.
    """

    def __init__(self, fixed_label: str = "talking"):
        self.fixed_label = fixed_label
        self._loaded = False

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            modalities={"image", "text"},
            supports_generation=True,
            supports_embedding=False,
            supports_classification=False,
            train_modes={"inference"},
        )

    def load(self, checkpoint: str, **kwargs) -> None:
        self._loaded = True

    def predict(self, bundle: ModalityBundle, request: ModelRequest) -> RawModelOutput:
        return RawModelOutput(text=self.fixed_label)

    def unload(self) -> None:
        self._loaded = False


@pytest.fixture
def mock_model_adapter() -> MockModelAdapter:
    """A mock model adapter that always predicts 'talking'."""
    adapter = MockModelAdapter(fixed_label="talking")
    adapter.load("mock-checkpoint")
    return adapter


# ── Config ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_config() -> ExperimentConfig:
    """Minimal experiment config for testing."""
    return ExperimentConfig(
        experiment_type="chunk_classification",
        labels=["talking", "building", "idle"],
        model_name="mock",
        model_checkpoint="mock-checkpoint",
        weights_path="",
        quantization="none",
        video_path=str(VIDEO_PATH),
        chunk_duration=3.0,
        chunk_stride=3.0,
        fps=2.0,
        resize=(224, 224),
        backend="single_device",
        output_format="json",
        output_path="results/test_output.json",
    )


@pytest.fixture
def sample_config_dict() -> dict:
    """Raw dict matching YAML structure for ExperimentConfig.from_dict."""
    return {
        "experiment": {
            "type": "chunk_classification",
            "labels": ["talking", "building", "idle"],
        },
        "model": {
            "name": "gemma_vlm",
            "checkpoint": "google/gemma-3-4b-it",
            "weights_path": "../data-slow/models/Gemma/current-model-variation",
            "quantization": "4bit",
        },
        "data": {
            "video_path": str(VIDEO_PATH),
            "chunk_duration": 3.0,
            "chunk_stride": 3.0,
            "fps": 2.0,
            "resize": [224, 224],
        },
        "execution": {
            "backend": "single_device",
        },
        "output": {
            "format": "json",
            "path": "results/chunk_classification_output.json",
        },
    }
