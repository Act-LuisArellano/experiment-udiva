"""
Qwen 2.5 Omni model adapter.

Supports all three modalities: image, text, and audio.
Uses Qwen2_5OmniForConditionalGeneration + Qwen2_5OmniProcessor
from Hugging Face transformers.
"""

from __future__ import annotations

import gc

import torch
from PIL import Image

from src.core.capabilities import ModelCapabilities
from src.core.interfaces import BaseModelAdapter
from src.core.registry import MODEL_REGISTRY
from src.core.schemas import ModalityBundle, ModelRequest, RawModelOutput
from src.models.base import detect_device, build_quantization_config


QWEN_OMNI_LIGHT_CHECKPOINT = "tiny-random/qwen2.5-omni"
QWEN_OMNI_SERVER_CHECKPOINT = "Qwen/Qwen2.5-Omni-3B"


@MODEL_REGISTRY.register("qwen_omni")
class QwenOmniAdapter(BaseModelAdapter):
    """Adapter for Qwen 2.5 Omni models.

    Supports image, text, and audio modalities.
    Tested with:
    - tiny-random/qwen2.5-omni (local smoke tests)
    - Qwen/Qwen2.5-Omni-3B (server deployment profile)
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = detect_device()
        self.checkpoint = ""
        self._generate_audio = False
        self._loaded = False

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            modalities={"image", "text", "audio"},
            supports_generation=True,
            supports_embedding=False,
            supports_classification=False,
            train_modes={"inference"},
            max_images=4,
            max_tokens=128,
        )

    def load(self, checkpoint: str, quantization: str = "none", weights_path: str = "", **kwargs) -> None:
        """Load Qwen Omni model from HuggingFace.

        Args:
            checkpoint: HF model ID or "auto" for profile default
            quantization: '4bit', '8bit', '2bit', or 'none'
            weights_path: Optional local cache directory
            kwargs:
                profile: "light" (default) or "server"
                device_map: forwarded to HF loader (useful for multi-GPU)
                max_memory: forwarded to HF loader for per-GPU memory caps
                attn_implementation: forwarded to HF loader
                torch_dtype: dtype override ("float16", "bfloat16", ...)
        """
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        profile = kwargs.pop("profile", "light")
        resolved_checkpoint = self._resolve_checkpoint(checkpoint, profile=profile)
        cache_dir = weights_path if weights_path else None
        self._generate_audio = bool(kwargs.pop("generate_audio", False))
        disable_talker = bool(kwargs.pop("disable_talker", not self._generate_audio))

        # Build model kwargs
        model_kwargs = {}
        requested_device_map = kwargs.pop("device_map", None)
        if requested_device_map is not None:
            model_kwargs["device_map"] = requested_device_map
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = self._resolve_torch_dtype(torch_dtype)
        elif self.device == "cuda":
            model_kwargs["torch_dtype"] = "auto"
        else:
            # MPS / CPU: use float16 to avoid VRAM overflow
            model_kwargs["torch_dtype"] = torch.float16

        max_memory = kwargs.pop("max_memory", None)
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        attn_implementation = kwargs.pop("attn_implementation", None)
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        # Quantization
        quant_config = build_quantization_config(quantization, self.device)
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        # Forward any additional user-provided HF kwargs.
        model_kwargs.update(kwargs)

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            resolved_checkpoint,
            cache_dir=cache_dir,
        )
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            resolved_checkpoint,
            cache_dir=cache_dir,
            **model_kwargs,
        )
        self.checkpoint = resolved_checkpoint

        # Move to device if not using device_map
        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)

        if disable_talker and hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

        self.model.eval()
        self._loaded = True

    def _resolve_checkpoint(self, checkpoint: str, profile: str = "light") -> str:
        """Resolve local/server defaults when checkpoint is omitted or set to auto."""
        if checkpoint and checkpoint != "auto":
            return checkpoint
        if profile == "server":
            return QWEN_OMNI_SERVER_CHECKPOINT
        return QWEN_OMNI_LIGHT_CHECKPOINT

    def _resolve_torch_dtype(self, dtype_value: str | torch.dtype) -> str | torch.dtype:
        """Convert string dtype aliases to torch dtype objects."""
        if not isinstance(dtype_value, str):
            return dtype_value
        mapping = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(dtype_value.lower(), dtype_value)

    def predict(self, bundle: ModalityBundle, request: ModelRequest) -> RawModelOutput:
        """Run inference: multimodal inputs → generated text."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prepare images from frames
        images = self._frames_to_pil(bundle.frames) if bundle.frames is not None else None

        # Build messages with embedded PIL images (Qwen-specific)
        messages = self._build_messages(bundle, request, images)

        # Apply chat template to get the tokenizable text
        try:
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        except (ValueError, Exception) as e:
            # Fallback: if template fails, manually prepend image tokens
            num_images = len(images) if images is not None else 0
            # Qwen Omni models typically use <|IMAGE|> as placeholder
            image_token = "<|IMAGE|>"
            # Some versions use <|vision_bos|><|IMAGE|><|vision_eos|>
            image_prefix = f"<|vision_bos|>{image_token}<|vision_eos|>" * num_images
            prompt = bundle.text_prompt or "Describe what you see."
            input_text = f"{image_prefix}{prompt}"
            print(f"DEBUG: Qwen template failed ({e}), using fallback.")

        # print(f"DEBUG: Qwen input_text: {repr(input_text)}") # Keep for future if needed
        # We must ensure that the text contains at least one image token per image
        if images and "<|IMAGE|>" not in input_text:
            print("WARNING: No image tokens found in input_text, manually injecting.")
            image_prefix = "<|vision_bos|><|IMAGE|><|vision_eos|>" * len(images)
            input_text = f"{image_prefix}{input_text}"

        # Prepare audio
        audio = None
        if bundle.audio is not None and bundle.audio.numel() > 0:
            audio = [bundle.audio.numpy()]

        processor_kwargs = {
            "text": input_text,
            "return_tensors": "pt",
        }
        if images is not None:
            processor_kwargs["images"] = images
        if audio is not None:
            processor_kwargs["audio"] = audio

        inputs = self.processor(**processor_kwargs)

        # Move to device
        if self.device != "cuda":
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate
        generate_kwargs = {
            "max_new_tokens": 128,
            "do_sample": False,
        }
        if not self._generate_audio:
            generate_kwargs["return_audio"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Qwen Omni may return either tensor (text ids) or tuple(text ids, audio)
        text_ids = outputs[0] if isinstance(outputs, tuple) else outputs

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = text_ids[0][input_len:]
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

    def _build_messages(
        self,
        bundle: ModalityBundle,
        request: ModelRequest,
        images: list[Image.Image] | None = None,
    ) -> list[dict]:
        """Build chat messages with interleaved modalities.

        Qwen requires PIL images to be embedded directly in the message
        content dicts so that apply_chat_template can correctly insert
        the image placeholder tokens.
        """
        messages = []

        # Add system prompt if provided
        system_prompt = request.extra.get("system_prompt", "")
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        content = []

        # Add frames as images — Qwen needs the PIL image in the dict
        if images is not None:
            for img in images:
                content.append({"type": "image", "image": img})

        # Add audio if present
        if bundle.audio is not None and bundle.audio.numel() > 0:
            content.append({"type": "audio"})

        # Add text prompt
        prompt = bundle.text_prompt or "Describe what you see and hear."
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

        result = []
        for idx in indices:
            frame = frames[idx]  # (C, H, W)
            frame_np = (frame.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
            result.append(Image.fromarray(frame_np, mode="RGB"))

        return result
