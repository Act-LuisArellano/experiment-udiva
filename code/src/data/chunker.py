"""
Video chunker: splits a VideoSample into temporal VideoChunks.
"""

from __future__ import annotations

from src.core.schemas import VideoSample, VideoChunk


def chunk_video(
    video: VideoSample,
    duration: float,
    stride: float,
    min_chunk_duration: float = 1.0,
) -> list[VideoChunk]:
    """Split a video into temporal chunks.

    Args:
        video: The source video sample.
        duration: Length of each chunk in seconds.
        stride: Step between chunk starts in seconds.
        min_chunk_duration: Minimum duration for a chunk to be kept (filters tails).

    Returns:
        List of VideoChunk objects with sequential indices.
    """
    chunks: list[VideoChunk] = []
    chunk_start = 0.0
    chunk_idx = 0

    while chunk_start < video.duration:
        chunk_end = min(chunk_start + duration, video.duration)
        actual_duration = chunk_end - chunk_start

        # Filter out chunks that are too short
        if actual_duration < min_chunk_duration:
            break

        chunks.append(VideoChunk(
            video=video,
            index=chunk_idx,
            start=chunk_start,
            end=chunk_end,
        ))

        chunk_start += stride
        chunk_idx += 1

    return chunks
