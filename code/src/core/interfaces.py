"""
Abstract base classes defining the interface contract for each layer.

Every model adapter, experiment, execution backend, and data pipeline
must implement these ABCs to be composable within the framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.capabilities import ModelCapabilities, TaskRequirements
from src.core.schemas import (
    CanonicalPrediction,
    ExperimentConfig,
    ModalityBundle,
    ModelRequest,
    RawModelOutput,
    VideoChunk,
    VideoSample,
)


class BaseModelAdapter(ABC):
    """Thin wrapper around a model (HuggingFace, custom, etc.)."""

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Declare what this model supports.

        Returns a ModelCapabilities instance describing modalities,
        generation/embedding/classification support, and train modes.
        """
        ...

    @abstractmethod
    def load(self, checkpoint: str, **kwargs) -> None:
        """Load model weights from *checkpoint*."""
        ...

    @abstractmethod
    def predict(self, bundle: ModalityBundle, request: ModelRequest) -> RawModelOutput:
        """Run inference on a single ModalityBundle."""
        ...

    def unload(self) -> None:
        """Optional: release resources (GPU memory, etc.)."""
        pass


class BaseExperiment(ABC):
    """Strategy that defines what an experiment does."""

    @property
    @abstractmethod
    def requirements(self) -> TaskRequirements:
        """Declare what this experiment needs from a model."""
        ...

    @abstractmethod
    def prepare_request(self, chunk: VideoChunk, config: ExperimentConfig) -> ModelRequest:
        """Build the ModelRequest for a single chunk."""
        ...

    @abstractmethod
    def postprocess(self, raw: RawModelOutput, chunk: VideoChunk, config: ExperimentConfig) -> CanonicalPrediction:
        """Convert raw model output into a canonical prediction."""
        ...

    @abstractmethod
    def run(self, config: ExperimentConfig, model: BaseModelAdapter, backend: "BaseExecutionBackend") -> list[CanonicalPrediction]:
        """Execute the full experiment pipeline."""
        ...


class BaseExecutionBackend(ABC):
    """Handles compute: device placement, multi-GPU wrapping, etc."""

    @abstractmethod
    def setup(self, model: BaseModelAdapter, **kwargs) -> None:
        """Prepare the model for execution (move to device, wrap, etc.)."""
        ...

    @abstractmethod
    def run_model(self, model: BaseModelAdapter, bundle: ModalityBundle, request: ModelRequest) -> RawModelOutput:
        """Execute model inference within this backend's compute context."""
        ...

    def teardown(self) -> None:
        """Optional: cleanup after experiment."""
        pass


class BaseDataPipeline(ABC):
    """Loads video data and produces ModalityBundles."""

    @abstractmethod
    def load_video(self, path: str) -> VideoSample:
        """Load video metadata from a file path."""
        ...

    @abstractmethod
    def chunk(self, video: VideoSample, duration: float, stride: float) -> list[VideoChunk]:
        """Split a video into temporal chunks."""
        ...

    @abstractmethod
    def build_modality_bundle(
        self,
        chunk: VideoChunk,
        model_capabilities: dict[str, Any],
        request: ModelRequest,
    ) -> ModalityBundle:
        """Extract and prepare modalities needed by the model for this chunk."""
        ...
