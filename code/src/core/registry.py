"""
Generic registry pattern for pluggable components.

Usage:
    MODEL_REGISTRY.register("gemma_vlm")(GemmaVLMAdapter)
    adapter_cls = MODEL_REGISTRY.get("gemma_vlm")
"""

from __future__ import annotations

from typing import Any


class Registry:
    """A name → class mapping with decorator-based registration."""

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, Any] = {}

    def register(self, key: str):
        """Decorator to register a class under *key*."""
        def decorator(cls):
            if key in self._registry:
                raise ValueError(
                    f"Duplicate registration in {self.name}: "
                    f"'{key}' is already registered to {self._registry[key].__name__}"
                )
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, key: str) -> Any:
        """Retrieve the class registered under *key*."""
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise KeyError(
                f"'{key}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[key]

    def list_registered(self) -> list[str]:
        """Return all registered keys."""
        return sorted(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        items = ", ".join(sorted(self._registry.keys()))
        return f"Registry({self.name!r}, [{items}])"


# ── Global registries ──────────────────────────────────────────────────────

MODEL_REGISTRY = Registry("models")
EXPERIMENT_REGISTRY = Registry("experiments")
EXECUTION_REGISTRY = Registry("execution_backends")
