"""
Gemma VLM adapter: wraps a Gemma-family vision-language model.

Supports any HuggingFace Gemma VLM checkpoint. Uses 4-bit quantization
on CUDA and float16/float32 on MPS/CPU.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
from PIL import Image

from src.core.capabilities import ModelCapabilities
from src.core.interfaces import BaseModelAdapter
from src.core.registry import MODEL_REGISTRY
from src.core.schemas import ModalityBundle, ModelRequest, RawModelOutput
from src.models.base import detect_device, build_quantization_config


@MODEL_REGISTRY.register("gemma_vlm")
class GemmaVLMAdapter(BaseModelAdapter):
    """Adapter for Gemma Vision-Language Models (e.g., gemma-4-E2B).

    Supports image, text, and audio modalities.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = detect_device()
        self._loaded = False

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            modalities={"image", "text", "audio"},
            supports_generation=True,
            supports_embedding=False,
            supports_classification=False,
            train_modes={"inference", "finetune"},
            max_images=4,
            max_tokens=128,
        )

    def load(self, checkpoint: str, quantization: str = "none", weights_path: str = "", **kwargs) -> None:
        """Load Gemma VLM from HuggingFace.

        Args:
            checkpoint: HF model ID, e.g. 'google/gemma-3-4b-it'
            quantization: '4bit', '8bit', or 'none'
            weights_path: Local directory to cache/load model weights.
                          If empty, uses HuggingFace default cache.
        """
        from pathlib import Path
        from transformers import AutoProcessor, AutoModelForImageTextToText

        # Resolve cache directory for weights
        cache_dir = None
        if weights_path:
            cache_dir = Path(weights_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = str(cache_dir)

        quant_config = build_quantization_config(quantization, self.device)

        model_kwargs: dict[str, Any] = {}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        elif self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            cache_dir=cache_dir,
            device_map="auto" if self.device == "cuda" else None,
            **model_kwargs,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

    def predict(self, bundle: ModalityBundle, request: ModelRequest) -> RawModelOutput:
        """Run inference: frames + prompt → generated text."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build messages for the chat template
        messages = self._build_messages(bundle, request)

        # Process inputs
        try:
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        except ValueError:
            # Fallback for models that do not have a chat template (e.g., base models)
            user_msg = next((m for m in messages if m.get("role") == "user"), messages[-1])
            num_images = sum(
                1 for c in user_msg.get("content", []) if isinstance(c, dict) and c.get("type") == "image"
            )
            image_token = getattr(self.processor, "image_token", "<image>")
            image_prefix = image_token * num_images
            prompt = bundle.text_prompt or "Describe what you see."
            input_text = f"{image_prefix}{prompt}"

        # Prepare images from frames
        images = self._frames_to_pil(bundle.frames) if bundle.frames is not None else None

        # Prepare audio
        audios = None
        if bundle.audio is not None and bundle.audio.numel() > 0:
            # Processor expects list of raw audio arrays
            audios = [bundle.audio.numpy()]

        processor_kwargs = {
            "text": input_text,
            "return_tensors": "pt",
        }
        if images is not None:
            processor_kwargs["images"] = images
        if audios is not None:
            processor_kwargs["audios"] = audios

        inputs = self.processor(**processor_kwargs)

        # Move to device
        if self.device != "cuda":  # device_map="auto" handles CUDA
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        return RawModelOutput(text=text.strip())

    def unload(self) -> None:
        """Release model from memory."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_messages(self, bundle: ModalityBundle, request: ModelRequest) -> list[dict]:
        """Build chat messages with interleaved images and text."""
        messages = []

        # Add system prompt if provided by the experiment
        system_prompt = request.extra.get("system_prompt", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []

        # Add frames as images (sample a few key frames to avoid overload)
        if bundle.frames is not None and bundle.frames.numel() > 0:
            n_frames = bundle.frames.shape[0]
            # Select up to 4 evenly-spaced frames
            max_images = 4
            if n_frames <= max_images:
                indices = list(range(n_frames))
            else:
                indices = [int(i * (n_frames - 1) / (max_images - 1)) for i in range(max_images)]

            for idx in indices:
                content.append({"type": "image"})

        # Add audio token if audio is present
        if bundle.audio is not None and bundle.audio.numel() > 0:
            content.append({"type": "audio"})

        # Add text prompt
        prompt = bundle.text_prompt or "Describe what you see."
        content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content})
        return messages

    def _frames_to_pil(self, frames: torch.Tensor) -> list[Image.Image]:
        """Convert frames tensor (T, C, H, W) float32 [0,1] → list of PIL Images."""
        n_frames = frames.shape[0]
        max_images = 4
        if n_frames <= max_images:
            indices = list(range(n_frames))
        else:
            indices = [int(i * (n_frames - 1) / (max_images - 1)) for i in range(max_images)]

        images = []
        for idx in indices:
            frame = frames[idx]  # (C, H, W)
            # Convert to (H, W, C) uint8
            frame_np = (frame.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
            images.append(Image.fromarray(frame_np, mode="RGB"))

        return images
