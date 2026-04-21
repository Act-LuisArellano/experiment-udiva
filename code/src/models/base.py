"""
Base model utilities: device detection, quantization helpers.

Re-exports BaseModelAdapter for convenience.
"""

from __future__ import annotations

import torch

from src.core.interfaces import BaseModelAdapter  # noqa: F401


def detect_device() -> str:
    """Auto-detect the best available compute device.

    Returns one of: 'cuda', 'mps', 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def build_quantization_config(quantization: str, device: str):
    """Build a BitsAndBytes quantization config if applicable.

    Returns None if quantization is not supported on this device.
    4-bit and 8-bit quantization only work on CUDA.
    """
    if quantization == "none" or device != "cuda":
        return None

    try:
        from transformers import BitsAndBytesConfig

        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        pass
        
    if quantization == "2bit":
        try:
            from transformers import QuantoConfig
            return QuantoConfig(weights="int2")
        except ImportError:
            pass

    return None
