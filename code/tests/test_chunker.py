"""
TDD tests for the video chunker module.

Tests chunking logic: exact divisions, remainders, short videos, tail filtering.
"""

import pytest
from pathlib import Path

from src.core.schemas import VideoSample, VideoChunk
from src.data.chunker import chunk_video


@pytest.fixture
def ten_second_video():
    """A synthetic 10-second video sample."""
    return VideoSample(
        path=Path("/tmp/fake.mp4"),
        duration=10.0,
        fps=5.0,
        width=1280,
        height=720,
    )


class TestChunkVideo:
    def test_exact_division(self, ten_second_video):
        """10s video, 5s chunks, 5s stride → 2 chunks."""
        chunks = chunk_video(ten_second_video, duration=5.0, stride=5.0)
        assert len(chunks) == 2
        assert chunks[0].start == pytest.approx(0.0)
        assert chunks[0].end == pytest.approx(5.0)
        assert chunks[1].start == pytest.approx(5.0)
        assert chunks[1].end == pytest.approx(10.0)

    def test_overlapping_chunks(self, ten_second_video):
        """10s video, 5s chunks, 2.5s stride → overlapping."""
        chunks = chunk_video(ten_second_video, duration=5.0, stride=2.5)
        # starts: 0, 2.5, 5, 7.5 → 4 chunks (last chunk 7.5→10 is 2.5s, ≥1s)
        # but 7.5+5=12.5 clamped to 10 → duration 2.5s which is ≥1s
        assert len(chunks) >= 3

    def test_remainder_chunk(self):
        """7s video, 3s chunks, 3s stride → 2 full + 1 remainder (1s)."""
        video = VideoSample(path=Path("/tmp/v.mp4"), duration=7.0, fps=5.0, width=640, height=480)
        chunks = chunk_video(video, duration=3.0, stride=3.0)
        assert len(chunks) == 3
        assert chunks[2].start == pytest.approx(6.0)
        assert chunks[2].end == pytest.approx(7.0)
        assert chunks[2].duration == pytest.approx(1.0)

    def test_tail_too_short_filtered(self):
        """6.5s video, 3s chunks, 3s stride → tail is 0.5s, dropped."""
        video = VideoSample(path=Path("/tmp/v.mp4"), duration=6.5, fps=5.0, width=640, height=480)
        chunks = chunk_video(video, duration=3.0, stride=3.0)
        # chunks at 0-3, 3-6, then 6-6.5 = 0.5s < 1.0s → filtered
        assert len(chunks) == 2

    def test_short_video_single_chunk(self):
        """2s video, 3s chunk → 1 chunk covering full video."""
        video = VideoSample(path=Path("/tmp/v.mp4"), duration=2.0, fps=5.0, width=640, height=480)
        chunks = chunk_video(video, duration=3.0, stride=3.0)
        assert len(chunks) == 1
        assert chunks[0].start == pytest.approx(0.0)
        assert chunks[0].end == pytest.approx(2.0)

    def test_very_short_video_filtered(self):
        """0.5s video → too short, no chunks."""
        video = VideoSample(path=Path("/tmp/v.mp4"), duration=0.5, fps=5.0, width=640, height=480)
        chunks = chunk_video(video, duration=3.0, stride=3.0)
        assert len(chunks) == 0

    def test_chunk_indices_sequential(self, ten_second_video):
        chunks = chunk_video(ten_second_video, duration=3.0, stride=3.0)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_references_video(self, ten_second_video):
        chunks = chunk_video(ten_second_video, duration=5.0, stride=5.0)
        for chunk in chunks:
            assert chunk.video is ten_second_video

    def test_real_video(self, sample_video_sample):
        """Chunk the real 6.4s test video."""
        chunks = chunk_video(sample_video_sample, duration=3.0, stride=3.0)
        assert len(chunks) == 2  # 0-3, 3-6 (6-6.4 = 0.4s < 1s → filtered)
        assert chunks[0].duration == pytest.approx(3.0)
