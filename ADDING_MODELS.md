# Adding a New Model

This guide explains how to integrate a new model into the UDIVA experiment framework.

## Architecture Overview

The framework uses a **modular adapter pattern**. Each model is wrapped in an adapter class that:

1. **Declares its capabilities** (what modalities it accepts, what it can do)
2. **Implements a standard interface** (`load`, `predict`, `unload`)
3. **Registers itself** in the global model registry

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  YAML Config    │────▶│  Model Registry   │────▶│  Model Adapter  │
│  model.name:    │     │  MODEL_REGISTRY   │     │  (your code)    │
│  "my_model"     │     └──────────────────┘     └─────────────────┘
└─────────────────┘                                      │
                                                         ▼
                              ┌───────────────────────────────────────┐
                              │  ModelCapabilities                    │
                              │  modalities: {"image", "text", "audio"}│
                              │  supports_generation: True            │
                              └───────────────────────────────────────┘
```

## Step 1: Define Your Model's Capabilities

Capabilities are defined in `src/core/capabilities.py` as a typed dataclass. You don't modify this file — you return a `ModelCapabilities` instance from your adapter.

```python
from src.core.capabilities import ModelCapabilities

# What your model can do
ModelCapabilities(
    modalities={"image", "text", "audio"},  # Which inputs it accepts
    supports_generation=True,                # Can generate text
    supports_embedding=False,                # Can produce embeddings
    supports_classification=False,           # Can do logit-based classification
    train_modes={"inference"},               # "inference", "finetune", "train"
    max_images=4,                            # Max frames per call
    max_tokens=128,                          # Max generation tokens
)
```

### Available Modalities

| Modality | Pipeline Behavior | Data Source |
|----------|-------------------|-------------|
| `"image"` | Extracts video frames via ffmpeg → `ModalityBundle.frames` (torch.Tensor) | Video file |
| `"text"` | Builds a text prompt → `ModalityBundle.text_prompt` (str) | Experiment config |
| `"audio"` | Extracts audio via ffmpeg → `ModalityBundle.audio` (torch.Tensor, 16kHz mono) | Video file |

> **Key insight**: When you declare a modality in `capabilities.modalities`, the `DataPipeline` automatically extracts that data from the video. You don't need to write extraction code.

## Step 2: Create Your Adapter

Create a new file in `src/models/your_model.py`:

```python
"""
My Model adapter.
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


@MODEL_REGISTRY.register("my_model")  # ← Registry key used in YAML
class MyModelAdapter(BaseModelAdapter):
    """Adapter for My Model."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = detect_device()  # auto-detects cuda/mps/cpu
        self._loaded = False

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            modalities={"image", "text"},     # ← Declare what your model accepts
            supports_generation=True,
            supports_embedding=False,
            supports_classification=False,
            train_modes={"inference"},
            max_images=4,
            max_tokens=128,
        )

    def load(self, checkpoint: str, quantization: str = "none",
             weights_path: str = "", **kwargs) -> None:
        """Load model weights.

        Args:
            checkpoint: HF model ID or local path
            quantization: '4bit', '8bit', '2bit', or 'none'
            weights_path: Optional local cache directory
        """
        from transformers import AutoProcessor, AutoModelForCausalLM

        cache_dir = weights_path if weights_path else None

        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16

        quant_config = build_quantization_config(quantization)
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        self.processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, cache_dir=cache_dir, **model_kwargs,
        )

        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

    def predict(self, bundle: ModalityBundle, request: ModelRequest) -> RawModelOutput:
        """Run inference."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build messages for chat template
        messages = self._build_messages(bundle, request)

        # Apply chat template
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )

        # Prepare processor kwargs
        processor_kwargs = {"text": input_text, "return_tensors": "pt"}

        # Add images if present
        if bundle.frames is not None:
            processor_kwargs["images"] = self._frames_to_pil(bundle.frames)

        # Add audio if present
        if bundle.audio is not None and bundle.audio.numel() > 0:
            processor_kwargs["audios"] = [bundle.audio.numpy()]

        inputs = self.processor(**processor_kwargs)

        if self.device != "cuda":
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

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

    def _build_messages(self, bundle, request):
        """Build chat messages with interleaved modalities."""
        messages = []
        system_prompt = request.extra.get("system_prompt", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        if bundle.frames is not None and bundle.frames.numel() > 0:
            for _ in range(min(bundle.frames.shape[0], 4)):
                content.append({"type": "image"})
        if bundle.audio is not None and bundle.audio.numel() > 0:
            content.append({"type": "audio"})
        content.append({"type": "text", "text": bundle.text_prompt or "Describe what you see."})

        messages.append({"role": "user", "content": content})
        return messages

    def _frames_to_pil(self, frames):
        """Convert frames tensor → list of PIL Images."""
        n = min(frames.shape[0], 4)
        return [
            Image.fromarray(
                (frames[i].permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy(),
                mode="RGB",
            )
            for i in range(n)
        ]
```

## Step 3: Register in `main.py`

Add an import at the top of `code/main.py` to trigger registration:

```python
# ── Import modules to trigger registry decorators ──────────────────────────
import src.models.gemma_vlm    # noqa: F401 — registers "gemma_vlm"
import src.models.qwen_omni    # noqa: F401 — registers "qwen_omni"
import src.models.my_model     # noqa: F401 — registers "my_model"   ← ADD THIS
```

## Step 4: Create a YAML Config

Create `configs/experiments/my_experiment.yaml`:

```yaml
experiment:
  type: vqa
  prompt_file: "configs/prompts/vqa_prompt.py"

model:
  name: my_model                                    # ← Must match registry key
  checkpoint: "org/my-model-id"                     # ← HuggingFace model ID
  weights_path: "../data-slow/models/MyModel"       # ← Local cache (optional)
  quantization: "none"                              # ← "none", "4bit", "8bit", "2bit"

data:
  video_path: "../data-fast/videos/my_video.mp4"
  chunk_duration: 3.0
  chunk_stride: 3.0
  fps: 1.0
  resize: [224, 224]

execution:
  backend: single_device

output:
  format: json
  path: "../data-slow/results/my_output.json"
```

## Step 5: Write Tests (TDD)

Create `tests/test_my_model.py`:

```python
import pytest
from src.core.interfaces import BaseModelAdapter
from src.core.capabilities import ModelCapabilities

class TestMyModelAdapter:
    def test_import(self):
        from src.models.my_model import MyModelAdapter
        assert isinstance(MyModelAdapter(), BaseModelAdapter)

    def test_capabilities(self):
        from src.models.my_model import MyModelAdapter
        caps = MyModelAdapter().capabilities
        assert isinstance(caps, ModelCapabilities)
        assert "image" in caps.modalities
        assert caps.supports_generation is True

    def test_predict_raises_if_not_loaded(self):
        from src.models.my_model import MyModelAdapter
        from src.core.schemas import ModalityBundle, ModelRequest
        adapter = MyModelAdapter()
        with pytest.raises(RuntimeError):
            adapter.predict(ModalityBundle(), ModelRequest(task="test"))

    def test_registry(self):
        from src.core.registry import MODEL_REGISTRY
        import src.models.my_model
        assert MODEL_REGISTRY.get("my_model") is not None

@pytest.mark.integration
class TestMyModelIntegration:
    def test_load_and_predict(self):
        from src.models.my_model import MyModelAdapter
        from src.core.schemas import ModalityBundle, ModelRequest, RawModelOutput
        adapter = MyModelAdapter()
        adapter.load("org/my-model-id")
        result = adapter.predict(
            ModalityBundle(text_prompt="Hello"),
            ModelRequest(task="vqa"),
        )
        assert isinstance(result, RawModelOutput)
        adapter.unload()
```

## Step 6: Run & Verify

```bash
# Run unit tests (no model download)
uv run python -m pytest -sv tests/test_my_model.py -m "not integration"

# Run integration tests (downloads model)
uv run python -m pytest -sv tests/test_my_model.py -m integration

# Run experiment with mock model
uv run python main.py --config configs/experiments/my_experiment.yaml --mock

# Run experiment with real model
uv run python main.py --config configs/experiments/my_experiment.yaml
```

## Compatibility Validation

The framework automatically validates that your model's capabilities match the experiment's requirements before running. If your model is missing a required modality, you'll get a clear error:

```
RuntimeError: Model 'my_model' is incompatible with experiment 'vqa':
  - Model is missing required modalities: ['audio']. Model has: ['image', 'text']
```

This validation is defined in `src/core/capabilities.py:validate_compatibility()`.

## Existing Models

| Registry Key | File | Modalities | Model |
|-------------|------|------------|-------|
| `gemma_vlm` | `src/models/gemma_vlm.py` | image, text, audio | Google Gemma 4 |
| `qwen_omni` | `src/models/qwen_omni.py` | image, text, audio | Qwen 2.5 Omni |

## Tips

- **VRAM Management**: Use `torch.float16` on MPS, quantization on CPU/MPS. See `AGENTS.md`.
- **Audio Processing**: Audio is extracted as 16kHz mono float32 tensor via ffmpeg. Pass as `[audio.numpy()]` to the processor.
- **Frame Selection**: Limit to 4 evenly-spaced frames to avoid VRAM issues.
- **Testing**: Always write unit tests first (TDD). Mark real-model tests with `@pytest.mark.integration`.
