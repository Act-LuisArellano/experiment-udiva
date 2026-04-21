"""
TDD tests for core schemas.

Tests all dataclass creation, field defaults, the VideoChunk.duration property,
and ExperimentConfig.from_dict YAML-dict loading.
"""

import pytest
from pathlib import Path

from src.core.schemas import (
    VideoSample,
    VideoChunk,
    ModalityBundle,
    ModelRequest,
    RawModelOutput,
    CanonicalPrediction,
    ExperimentConfig,
)


# ── VideoSample ────────────────────────────────────────────────────────────

class TestVideoSample:
    def test_creation(self):
        vs = VideoSample(path=Path("/tmp/v.mp4"), duration=10.0, fps=5.0, width=1280, height=720)
        assert vs.duration == 10.0
        assert vs.fps == 5.0
        assert vs.has_audio is False  # default

    def test_has_audio_flag(self):
        vs = VideoSample(path=Path("/tmp/v.mp4"), duration=5.0, fps=25.0, width=1920, height=1080, has_audio=True)
        assert vs.has_audio is True


# ── VideoChunk ─────────────────────────────────────────────────────────────

class TestVideoChunk:
    def test_duration_property(self, sample_video_sample):
        chunk = VideoChunk(video=sample_video_sample, index=0, start=1.0, end=4.0)
        assert chunk.duration == pytest.approx(3.0)

    def test_zero_duration(self, sample_video_sample):
        chunk = VideoChunk(video=sample_video_sample, index=0, start=2.0, end=2.0)
        assert chunk.duration == pytest.approx(0.0)

    def test_chunk_references_video(self, sample_video_sample):
        chunk = VideoChunk(video=sample_video_sample, index=1, start=3.0, end=6.0)
        assert chunk.video.path == sample_video_sample.path


# ── ModalityBundle ─────────────────────────────────────────────────────────

class TestModalityBundle:
    def test_defaults_are_none(self):
        bundle = ModalityBundle()
        assert bundle.frames is None
        assert bundle.text_prompt is None
        assert bundle.audio is None
        assert bundle.chunk is None

    def test_with_all_fields(self, sample_frames, sample_chunk):
        bundle = ModalityBundle(
            frames=sample_frames,
            text_prompt="classify this",
            audio=b"fake_audio",
            chunk=sample_chunk,
        )
        assert bundle.frames is not None
        assert bundle.text_prompt == "classify this"
        assert bundle.audio == b"fake_audio"


# ── ModelRequest ───────────────────────────────────────────────────────────

class TestModelRequest:
    def test_creation_with_labels(self):
        req = ModelRequest(task="classify", labels=["a", "b", "c"])
        assert req.task == "classify"
        assert len(req.labels) == 3

    def test_default_empty_lists(self):
        req = ModelRequest(task="embed")
        assert req.labels == []
        assert req.extra == {}


# ── RawModelOutput ─────────────────────────────────────────────────────────

class TestRawModelOutput:
    def test_text_output(self):
        out = RawModelOutput(text="talking")
        assert out.text == "talking"
        assert out.embeddings is None

    def test_all_none_by_default(self):
        out = RawModelOutput()
        assert out.text is None
        assert out.embeddings is None
        assert out.logits is None


# ── CanonicalPrediction ───────────────────────────────────────────────────

class TestCanonicalPrediction:
    def test_creation(self, sample_canonical_prediction):
        pred = sample_canonical_prediction
        assert pred.label == "talking"
        assert pred.confidence == pytest.approx(0.85)
        assert pred.chunk_index == 0

    def test_default_empty_collections(self):
        pred = CanonicalPrediction(chunk_index=0, chunk_start=0.0, chunk_end=3.0)
        assert pred.entities == []
        assert pred.events == []
        assert pred.relations == []
        assert pred.label == ""


# ── ExperimentConfig ──────────────────────────────────────────────────────

class TestExperimentConfig:
    def test_from_dict(self, sample_config_dict):
        config = ExperimentConfig.from_dict(sample_config_dict)
        assert config.experiment_type == "chunk_classification"
        assert config.labels == ["talking", "building", "idle"]
        assert config.model_name == "gemma_vlm"
        assert config.model_checkpoint == "google/gemma-3-4b-it"
        assert config.weights_path == "../data-slow/models/Gemma/current-model-variation"
        assert config.quantization == "4bit"
        assert config.chunk_duration == 3.0
        assert config.resize == (224, 224)
        assert config.backend == "single_device"

    def test_from_dict_defaults(self):
        config = ExperimentConfig.from_dict({})
        assert config.experiment_type == ""
        assert config.labels == []
        assert config.weights_path == ""
        assert config.quantization == "none"
        assert config.model_load_kwargs == {}
        assert config.resize is None
        assert config.backend == "single_device"

    def test_from_dict_model_load_kwargs(self):
        config = ExperimentConfig.from_dict(
            {
                "model": {
                    "load_kwargs": {
                        "profile": "server",
                        "device_map": "auto",
                    }
                }
            }
        )
        assert config.model_load_kwargs == {
            "profile": "server",
            "device_map": "auto",
        }

    def test_from_dict_no_resize(self):
        config = ExperimentConfig.from_dict({"data": {"resize": None}})
        assert config.resize is None

    def test_direct_construction(self, sample_config):
        assert sample_config.experiment_type == "chunk_classification"
        assert len(sample_config.labels) == 3
