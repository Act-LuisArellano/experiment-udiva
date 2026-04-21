"""Core module: schemas, registry, interfaces, and capabilities for the experiment framework."""

from src.core.schemas import (
    VideoSample,
    VideoChunk,
    ModalityBundle,
    ModelRequest,
    RawModelOutput,
    CanonicalPrediction,
    ExperimentConfig,
)
from src.core.capabilities import ModelCapabilities, TaskRequirements, validate_compatibility
from src.core.registry import Registry, MODEL_REGISTRY, EXPERIMENT_REGISTRY, EXECUTION_REGISTRY
from src.core.interfaces import BaseModelAdapter, BaseExperiment, BaseExecutionBackend, BaseDataPipeline

__all__ = [
    "VideoSample",
    "VideoChunk",
    "ModalityBundle",
    "ModelRequest",
    "RawModelOutput",
    "CanonicalPrediction",
    "ExperimentConfig",
    "ModelCapabilities",
    "TaskRequirements",
    "validate_compatibility",
    "Registry",
    "MODEL_REGISTRY",
    "EXPERIMENT_REGISTRY",
    "EXECUTION_REGISTRY",
    "BaseModelAdapter",
    "BaseExperiment",
    "BaseExecutionBackend",
    "BaseDataPipeline",
]
