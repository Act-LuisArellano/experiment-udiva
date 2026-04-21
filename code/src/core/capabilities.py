"""
Typed capabilities and requirements system for task↔model matching.

Replaces the freeform dict[str, Any] capabilities with typed dataclasses
so that compatibility between tasks and models can be validated before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelCapabilities:
    """Declares what a model can do.

    Every model adapter returns one of these from its ``capabilities`` property
    so the framework can match it against task requirements.
    """
    modalities: set[str]              # e.g. {"image", "text", "audio"}
    supports_generation: bool         # can produce text output
    supports_embedding: bool = False  # can produce embedding vectors
    supports_classification: bool = False  # can do logit-based classification
    train_modes: set[str] = field(default_factory=lambda: {"inference"})
    max_images: int = 4               # max frames/images per call
    max_tokens: int = 128             # max generation tokens

    def to_dict(self) -> dict[str, Any]:
        """Backward-compatible dict representation.

        Produces the same shape as the old dict-based capabilities
        so DataPipeline.build_modality_bundle continues to work.
        """
        return {
            "modalities": self.modalities,
            "supports_generation": self.supports_generation,
            "supports_embedding": self.supports_embedding,
            "supports_classification": self.supports_classification,
            "supports_audio": "audio" in self.modalities,
            "train_modes": self.train_modes,
            "max_images": self.max_images,
            "max_tokens": self.max_tokens,
        }


@dataclass
class TaskRequirements:
    """Declares what a task needs from a model.

    Every experiment returns one of these from its ``requirements`` property
    so the framework can validate compatibility before execution.
    """
    required_modalities: set[str]          # e.g. {"image", "text"}
    needs_generation: bool                 # needs text generation
    needs_embedding: bool = False          # needs embedding vectors
    needs_classification: bool = False     # needs logit output
    output_type: str = "text"              # "text", "label", "embedding", "structured"


def validate_compatibility(
    capabilities: ModelCapabilities,
    requirements: TaskRequirements,
) -> list[str]:
    """Check if a model can fulfill a task's requirements.

    Returns a list of incompatibility reasons.
    An empty list means the model is fully compatible with the task.

    Args:
        capabilities: What the model offers.
        requirements: What the task demands.

    Returns:
        List of human-readable error strings. Empty = compatible.
    """
    errors: list[str] = []

    # Check modalities
    missing = requirements.required_modalities - capabilities.modalities
    if missing:
        errors.append(
            f"Model is missing required modalities: {sorted(missing)}. "
            f"Model has: {sorted(capabilities.modalities)}"
        )

    # Check generation
    if requirements.needs_generation and not capabilities.supports_generation:
        errors.append(
            "Task requires text generation but model does not support it."
        )

    # Check embedding
    if requirements.needs_embedding and not capabilities.supports_embedding:
        errors.append(
            "Task requires embedding output but model does not support it."
        )

    # Check classification
    if requirements.needs_classification and not capabilities.supports_classification:
        errors.append(
            "Task requires classification (logits) but model does not support it."
        )

    return errors
