"""
Single-device execution backend.

Handles inference on a single GPU (CUDA), MPS, or CPU.
This is the simplest backend — future backends (DDP, FSDP, DeepSpeed)
will follow the same interface.
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.interfaces import BaseExecutionBackend, BaseModelAdapter
from src.core.registry import EXECUTION_REGISTRY
from src.core.schemas import ModalityBundle, ModelRequest, RawModelOutput
from src.models.base import detect_device


@EXECUTION_REGISTRY.register("single_device")
class SingleDeviceBackend(BaseExecutionBackend):
    """Execute model inference on a single device (GPU/MPS/CPU)."""

    def __init__(self):
        self.device: str | None = None

    def setup(self, model: BaseModelAdapter, **kwargs) -> None:
        """Detect device and prepare for inference."""
        self.device = detect_device()

    def run_model(
        self,
        model: BaseModelAdapter,
        bundle: ModalityBundle,
        request: ModelRequest,
    ) -> RawModelOutput:
        """Run model.predict within a no_grad context."""
        with torch.no_grad():
            return model.predict(bundle, request)

    def teardown(self) -> None:
        """Cleanup — release any cached state."""
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
