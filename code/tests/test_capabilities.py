"""
TDD tests for the capabilities system.

Tests ModelCapabilities, TaskRequirements, and validate_compatibility
to ensure typed task↔model matching works correctly.
"""

import pytest

from src.core.capabilities import (
    ModelCapabilities,
    TaskRequirements,
    validate_compatibility,
)


# ── ModelCapabilities ──────────────────────────────────────────────────────

class TestModelCapabilities:
    def test_creation_with_defaults(self):
        caps = ModelCapabilities(
            modalities={"image", "text"},
            supports_generation=True,
        )
        assert caps.modalities == {"image", "text"}
        assert caps.supports_generation is True
        assert caps.supports_embedding is False
        assert caps.supports_classification is False
        assert caps.train_modes == {"inference"}
        assert caps.max_images == 4
        assert caps.max_tokens == 128

    def test_creation_full(self):
        caps = ModelCapabilities(
            modalities={"image", "text", "audio"},
            supports_generation=True,
            supports_embedding=True,
            supports_classification=True,
            train_modes={"inference", "finetune"},
            max_images=8,
            max_tokens=512,
        )
        assert "audio" in caps.modalities
        assert caps.supports_embedding is True
        assert caps.train_modes == {"inference", "finetune"}

    def test_to_dict_backward_compat(self):
        """The to_dict() method should produce the same shape as the old dict capabilities."""
        caps = ModelCapabilities(
            modalities={"image", "text"},
            supports_generation=True,
        )
        d = caps.to_dict()
        assert isinstance(d["modalities"], set)
        assert d["supports_generation"] is True
        assert d["supports_embedding"] is False
        assert "train_modes" in d

    def test_empty_modalities(self):
        caps = ModelCapabilities(modalities=set(), supports_generation=False)
        assert len(caps.modalities) == 0


# ── TaskRequirements ───────────────────────────────────────────────────────

class TestTaskRequirements:
    def test_creation(self):
        reqs = TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            output_type="text",
        )
        assert reqs.required_modalities == {"image", "text"}
        assert reqs.needs_generation is True
        assert reqs.needs_embedding is False
        assert reqs.needs_classification is False
        assert reqs.output_type == "text"

    def test_classification_requirements(self):
        reqs = TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            needs_classification=False,
            output_type="label",
        )
        assert reqs.output_type == "label"


# ── validate_compatibility ─────────────────────────────────────────────────

class TestValidateCompatibility:
    def test_compatible_pair(self):
        caps = ModelCapabilities(
            modalities={"image", "text"},
            supports_generation=True,
        )
        reqs = TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            output_type="text",
        )
        errors = validate_compatibility(caps, reqs)
        assert errors == []

    def test_missing_modality_fails(self):
        caps = ModelCapabilities(
            modalities={"text"},
            supports_generation=True,
        )
        reqs = TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            output_type="text",
        )
        errors = validate_compatibility(caps, reqs)
        assert len(errors) > 0
        assert any("image" in e for e in errors)

    def test_missing_generation_fails(self):
        caps = ModelCapabilities(
            modalities={"image", "text"},
            supports_generation=False,
        )
        reqs = TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            output_type="text",
        )
        errors = validate_compatibility(caps, reqs)
        assert len(errors) > 0
        assert any("generation" in e.lower() for e in errors)

    def test_missing_embedding_fails(self):
        caps = ModelCapabilities(
            modalities={"image", "text"},
            supports_generation=True,
            supports_embedding=False,
        )
        reqs = TaskRequirements(
            required_modalities={"image"},
            needs_generation=False,
            needs_embedding=True,
            output_type="embedding",
        )
        errors = validate_compatibility(caps, reqs)
        assert len(errors) > 0
        assert any("embedding" in e.lower() for e in errors)

    def test_superset_modalities_ok(self):
        """Model supports more modalities than task requires — should pass."""
        caps = ModelCapabilities(
            modalities={"image", "text", "audio"},
            supports_generation=True,
        )
        reqs = TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            output_type="text",
        )
        errors = validate_compatibility(caps, reqs)
        assert errors == []

    def test_multiple_failures(self):
        """Missing both modality and capability — should report all errors."""
        caps = ModelCapabilities(
            modalities={"text"},
            supports_generation=False,
        )
        reqs = TaskRequirements(
            required_modalities={"image", "text", "audio"},
            needs_generation=True,
            output_type="text",
        )
        errors = validate_compatibility(caps, reqs)
        assert len(errors) >= 2  # at least missing modalities + generation
