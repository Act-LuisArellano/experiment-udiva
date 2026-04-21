"""
TDD tests for the video reader module.

Tests video info extraction, frame extraction, and tensor conversion
using the real 6.4-second test video.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.data.video_reader import get_video_info, extract_frames, frames_to_tensor


class TestGetVideoInfo:
    def test_returns_video_sample(self, sample_video_path):
        info = get_video_info(str(sample_video_path))
        assert info.path == sample_video_path
        assert info.duration == pytest.approx(6.4, abs=0.5)
        assert info.fps > 0
        assert info.width == 1280
        assert info.height == 720

    def test_detects_audio(self, sample_video_path):
        info = get_video_info(str(sample_video_path))
        assert info.has_audio is True

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            get_video_info("/nonexistent/video.mp4")


class TestExtractFrames:
    def test_returns_numpy_array(self, sample_video_path):
        frames = extract_frames(sample_video_path, start=0.0, duration=2.0, fps=2.0)
        assert isinstance(frames, np.ndarray)
        assert frames.ndim == 4  # (T, H, W, C)
        assert frames.dtype == np.uint8

    def test_frame_count_matches_fps(self, sample_video_path):
        frames = extract_frames(sample_video_path, start=0.0, duration=3.0, fps=2.0)
        # ~6 frames for 3s at 2fps (may vary ±1 due to encoding)
        assert 4 <= frames.shape[0] <= 8

    def test_resize(self, sample_video_path):
        frames = extract_frames(
            sample_video_path, start=0.0, duration=2.0, fps=2.0, resize=(224, 224)
        )
        assert frames.shape[1] == 224  # H
        assert frames.shape[2] == 224  # W
        assert frames.shape[3] == 3    # C

    def test_start_offset(self, sample_video_path):
        frames_start = extract_frames(sample_video_path, start=0.0, duration=2.0, fps=2.0)
        frames_mid = extract_frames(sample_video_path, start=3.0, duration=2.0, fps=2.0)
        # Both should return frames (content will differ but shape is similar)
        assert frames_start.shape[0] > 0
        assert frames_mid.shape[0] > 0

    def test_empty_result_for_past_end(self, sample_video_path):
        frames = extract_frames(sample_video_path, start=100.0, duration=1.0, fps=2.0)
        assert frames.shape[0] == 0


class TestFramesToTensor:
    def test_conversion_shape(self):
        # (T, H, W, C) uint8
        frames_np = np.random.randint(0, 255, (6, 64, 64, 3), dtype=np.uint8)
        tensor = frames_to_tensor(frames_np)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (6, 3, 64, 64)  # (T, C, H, W)
        assert tensor.dtype == torch.float32

    def test_normalization_range(self):
        frames_np = np.full((2, 32, 32, 3), 255, dtype=np.uint8)
        tensor = frames_to_tensor(frames_np)
        assert tensor.max() == pytest.approx(1.0)

        frames_np_zero = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        tensor_zero = frames_to_tensor(frames_np_zero)
        assert tensor_zero.min() == pytest.approx(0.0)

    def test_empty_input(self):
        frames_np = np.zeros((0, 64, 64, 3), dtype=np.uint8)
        tensor = frames_to_tensor(frames_np)
        assert tensor.shape[0] == 0
