"""
TDD tests for the data pipeline module.

Tests capability-aware modality bundle building and the full
load → chunk → bundle pipeline, including audio extraction.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.schemas import (
    VideoSample, VideoChunk, ModalityBundle, ModelRequest,
)
from src.data.pipeline import DataPipeline
from src.data.video_reader import extract_audio


@pytest.fixture
def pipeline():
    return DataPipeline()


# ── Audio extraction ──────────────────────────────────────────────────────

class TestExtractAudio:
    def test_returns_torch_tensor(self, sample_chunk):
        """extract_audio should return a torch.Tensor."""
        audio = extract_audio(
            sample_chunk.video.path,
            start=sample_chunk.start,
            duration=sample_chunk.duration,
        )
        assert isinstance(audio, torch.Tensor)

    def test_tensor_is_1d_or_2d(self, sample_chunk):
        """Audio tensor should be 1D (samples,) or 2D (channels, samples)."""
        audio = extract_audio(
            sample_chunk.video.path,
            start=sample_chunk.start,
            duration=sample_chunk.duration,
        )
        assert audio.ndim in (1, 2)

    def test_correct_sample_rate(self, sample_chunk):
        """Audio at 16kHz for 3 seconds should have ~48000 samples."""
        audio = extract_audio(
            sample_chunk.video.path,
            start=sample_chunk.start,
            duration=sample_chunk.duration,
            sr=16000,
        )
        # Allow some tolerance for ffmpeg rounding
        n_samples = audio.shape[-1]  # last dim is samples
        assert 47000 < n_samples < 49000

    def test_different_sample_rate(self, sample_chunk):
        """Changing sample rate should change the number of samples."""
        audio_16k = extract_audio(
            sample_chunk.video.path,
            start=sample_chunk.start,
            duration=sample_chunk.duration,
            sr=16000,
        )
        audio_8k = extract_audio(
            sample_chunk.video.path,
            start=sample_chunk.start,
            duration=sample_chunk.duration,
            sr=8000,
        )
        assert audio_8k.shape[-1] < audio_16k.shape[-1]

    def test_float32_dtype(self, sample_chunk):
        """Audio tensor should be float32."""
        audio = extract_audio(
            sample_chunk.video.path,
            start=sample_chunk.start,
            duration=sample_chunk.duration,
        )
        assert audio.dtype == torch.float32

    def test_values_normalized(self, sample_chunk):
        """Audio values should be in [-1, 1] range."""
        audio = extract_audio(
            sample_chunk.video.path,
            start=sample_chunk.start,
            duration=sample_chunk.duration,
        )
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0


# ── Build modality bundle ─────────────────────────────────────────────────

class TestBuildModalityBundle:
    def test_builds_bundle_with_frames(self, pipeline, sample_chunk, sample_model_request):
        """When model needs images, bundle should contain frames."""
        capabilities = {"modalities": {"image", "text"}}
        bundle = pipeline.build_modality_bundle(
            sample_chunk, capabilities, sample_model_request,
            fps=2.0, resize=(224, 224),
        )
        assert isinstance(bundle, ModalityBundle)
        assert bundle.frames is not None
        assert bundle.chunk is sample_chunk

    def test_text_prompt_included(self, pipeline, sample_chunk, sample_model_request):
        """When model needs text, prompt should be populated."""
        capabilities = {"modalities": {"image", "text"}}
        bundle = pipeline.build_modality_bundle(
            sample_chunk, capabilities, sample_model_request,
            fps=2.0, resize=(224, 224),
        )
        assert bundle.text_prompt is not None
        assert len(bundle.text_prompt) > 0

    def test_no_frames_if_text_only(self, pipeline, sample_chunk, sample_model_request):
        """If model only needs text, frames should be None."""
        capabilities = {"modalities": {"text"}}
        bundle = pipeline.build_modality_bundle(
            sample_chunk, capabilities, sample_model_request,
            fps=2.0, resize=(224, 224),
        )
        assert bundle.frames is None

    def test_prompt_contains_labels(self, pipeline, sample_chunk, sample_model_request):
        """Prompt should include the classification labels."""
        capabilities = {"modalities": {"image", "text"}}
        bundle = pipeline.build_modality_bundle(
            sample_chunk, capabilities, sample_model_request,
            fps=2.0, resize=(224, 224),
        )
        for label in sample_model_request.labels:
            assert label in bundle.text_prompt

    def test_audio_extracted_when_requested(self, pipeline, sample_chunk, sample_model_request):
        """When model declares 'audio' modality, bundle should contain audio tensor."""
        capabilities = {"modalities": {"image", "text", "audio"}}
        bundle = pipeline.build_modality_bundle(
            sample_chunk, capabilities, sample_model_request,
            fps=2.0, resize=(224, 224),
        )
        assert bundle.audio is not None
        assert isinstance(bundle.audio, torch.Tensor)
        assert bundle.audio.dtype == torch.float32

    def test_no_audio_if_not_requested(self, pipeline, sample_chunk, sample_model_request):
        """When model does NOT declare 'audio', bundle.audio should be None."""
        capabilities = {"modalities": {"image", "text"}}
        bundle = pipeline.build_modality_bundle(
            sample_chunk, capabilities, sample_model_request,
            fps=2.0, resize=(224, 224),
        )
        assert bundle.audio is None


class TestDataPipelineLoadVideo:
    def test_load_video(self, pipeline, sample_video_path):
        """Pipeline should load video metadata."""
        video = pipeline.load_video(str(sample_video_path))
        assert isinstance(video, VideoSample)
        assert video.duration > 0

    def test_chunk(self, pipeline, sample_video_sample):
        """Pipeline should chunk a video."""
        chunks = pipeline.chunk(sample_video_sample, duration=3.0, stride=3.0)
        assert len(chunks) > 0
        assert all(isinstance(c, VideoChunk) for c in chunks)

