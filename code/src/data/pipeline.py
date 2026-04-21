"""
Data pipeline: orchestrates video loading, chunking, and modality bundle building.

This is the concrete implementation of BaseDataPipeline that other layers use.
It is capability-aware — only extracts modalities the model actually needs.
"""

from __future__ import annotations

from typing import Any

from src.core.capabilities import ModelCapabilities
from src.core.interfaces import BaseDataPipeline
from src.core.schemas import (
    ModalityBundle,
    ModelRequest,
    VideoChunk,
    VideoSample,
)
from src.data.video_reader import get_video_info, extract_frames, frames_to_tensor, extract_audio
from src.data.chunker import chunk_video


class DataPipeline(BaseDataPipeline):
    """Concrete data pipeline: video → chunks → modality bundles."""

    def load_video(self, path: str) -> VideoSample:
        """Load video metadata from file."""
        return get_video_info(path)

    def chunk(self, video: VideoSample, duration: float, stride: float) -> list[VideoChunk]:
        """Split video into temporal chunks."""
        return chunk_video(video, duration=duration, stride=stride)

    def build_modality_bundle(
        self,
        chunk: VideoChunk,
        model_capabilities: dict[str, Any] | ModelCapabilities,
        request: ModelRequest,
        fps: float = 2.0,
        resize: tuple[int, int] | None = None,
    ) -> ModalityBundle:
        """Build a ModalityBundle for a chunk, extracting only needed modalities.

        Args:
            chunk: The video chunk to extract data from.
            model_capabilities: ModelCapabilities instance or legacy dict
                with 'modalities' set (e.g. {"image", "text"}).
            request: The ModelRequest with task info and labels.
            fps: Frame sampling rate.
            resize: Optional (H, W) resize for frames.
        """
        if isinstance(model_capabilities, ModelCapabilities):
            modalities = model_capabilities.modalities
        else:
            modalities = model_capabilities.get("modalities", set())

        # Extract frames if model needs images
        frames = None
        if "image" in modalities:
            frames_np = extract_frames(
                chunk.video.path,
                start=chunk.start,
                duration=chunk.duration,
                fps=fps,
                resize=resize,
            )
            frames = frames_to_tensor(frames_np)

        # Build text prompt if model needs text
        text_prompt = None
        if "text" in modalities:
            text_prompt = self._build_prompt(request, chunk)

        # Extract audio if model needs it
        audio = None
        if "audio" in modalities:
            audio = extract_audio(
                chunk.video.path,
                start=chunk.start,
                duration=chunk.duration,
            )

        return ModalityBundle(
            frames=frames,
            text_prompt=text_prompt,
            audio=audio,
            chunk=chunk,
        )

    def _build_prompt(self, request: ModelRequest, chunk: VideoChunk) -> str:
        """Build a text prompt from the model request and chunk info."""
        if request.prompt_template:
            # Use the experiment-provided prompt; optionally substitute {labels}
            if request.labels and "{labels}" in request.prompt_template:
                labels_str = ", ".join(request.labels)
                return request.prompt_template.format(labels=labels_str)
            return request.prompt_template
        elif request.labels:
            labels_str = ", ".join(request.labels)
            return (
                f"Analyze this video segment (from {chunk.start:.1f}s to {chunk.end:.1f}s). "
                f"Classify the activity into one of: {labels_str}. "
                f"Respond with only the label name."
            )
        else:
            return f"Describe what happens in this video segment ({chunk.start:.1f}s to {chunk.end:.1f}s)."
