"""
TDD tests for the Qwen Omni model adapter.

Tests the QwenOmniAdapter conformance to BaseModelAdapter interface,
capabilities declaration, and predict behavior with mock/integration tests.
"""

import pytest
import torch
import sys
from types import SimpleNamespace

from src.core.schemas import ModalityBundle, ModelRequest, RawModelOutput
from src.core.interfaces import BaseModelAdapter
from src.core.capabilities import ModelCapabilities


class TestQwenOmniAdapter:
    """Unit tests — no model download needed."""

    def test_import(self):
        """QwenOmniAdapter should be importable."""
        from src.models.qwen_omni import QwenOmniAdapter
        adapter = QwenOmniAdapter()
        assert isinstance(adapter, BaseModelAdapter)

    def test_capabilities_modalities(self):
        """Qwen Omni should declare image, text, and audio modalities."""
        from src.models.qwen_omni import QwenOmniAdapter
        adapter = QwenOmniAdapter()
        caps = adapter.capabilities
        assert isinstance(caps, ModelCapabilities)
        assert {"image", "text", "audio"} == caps.modalities

    def test_capabilities_generation(self):
        """Qwen Omni should support generation."""
        from src.models.qwen_omni import QwenOmniAdapter
        adapter = QwenOmniAdapter()
        assert adapter.capabilities.supports_generation is True

    def test_not_loaded_by_default(self):
        """Adapter should not be loaded before calling load()."""
        from src.models.qwen_omni import QwenOmniAdapter
        adapter = QwenOmniAdapter()
        assert adapter._loaded is False

    def test_registry_registered(self):
        """qwen_omni should be registered in MODEL_REGISTRY."""
        from src.core.registry import MODEL_REGISTRY
        import src.models.qwen_omni  # noqa: F401 — triggers registration

        cls = MODEL_REGISTRY.get("qwen_omni")
        from src.models.qwen_omni import QwenOmniAdapter
        assert cls is QwenOmniAdapter

    def test_predict_raises_if_not_loaded(self):
        """Predict should raise if model not loaded."""
        from src.models.qwen_omni import QwenOmniAdapter
        adapter = QwenOmniAdapter()
        bundle = ModalityBundle(text_prompt="test")
        request = ModelRequest(task="vqa")
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.predict(bundle, request)

    def test_auto_checkpoint_light_profile(self):
        """Auto checkpoint should resolve to a lightweight local model."""
        from src.models.qwen_omni import QwenOmniAdapter

        adapter = QwenOmniAdapter()
        resolved = adapter._resolve_checkpoint("auto", profile="light")
        assert resolved == "tiny-random/qwen2.5-omni"

    def test_auto_checkpoint_server_profile(self):
        """Auto checkpoint should resolve to server-size model for deployment."""
        from src.models.qwen_omni import QwenOmniAdapter

        adapter = QwenOmniAdapter()
        resolved = adapter._resolve_checkpoint("", profile="server")
        assert resolved == "Qwen/Qwen2.5-Omni-3B"

    def test_load_forwards_distribution_kwargs(self, monkeypatch):
        """load() should pass device_map/max_memory to HF model loader."""
        from src.models.qwen_omni import QwenOmniAdapter

        captured = {}

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, checkpoint, cache_dir=None):
                captured["processor_checkpoint"] = checkpoint
                captured["processor_cache_dir"] = cache_dir
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, checkpoint, cache_dir=None, **kwargs):
                captured["model_checkpoint"] = checkpoint
                captured["model_cache_dir"] = cache_dir
                captured["model_kwargs"] = kwargs
                return cls()

            def to(self, device):
                captured["moved_to"] = device
                return self

            def eval(self):
                captured["eval_called"] = True

        fake_transformers = SimpleNamespace(
            Qwen2_5OmniForConditionalGeneration=FakeModel,
            Qwen2_5OmniProcessor=FakeProcessor,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        adapter = QwenOmniAdapter()
        adapter.device = "cuda"
        adapter.load(
            checkpoint="auto",
            quantization="none",
            weights_path="/tmp/qwen-cache",
            profile="server",
            device_map="auto",
            max_memory={"cuda:0": "46GiB", "cuda:1": "46GiB"},
        )

        assert adapter._loaded is True
        assert captured["processor_checkpoint"] == "Qwen/Qwen2.5-Omni-3B"
        assert captured["model_checkpoint"] == "Qwen/Qwen2.5-Omni-3B"
        assert captured["model_kwargs"]["device_map"] == "auto"
        assert captured["model_kwargs"]["max_memory"] == {
            0: "46GiB",
            1: "46GiB",
        }
        assert "moved_to" not in captured

    def test_normalize_max_memory_cuda_keys(self):
        """cuda:N and numeric string keys should be converted to GPU indices."""
        from src.models.qwen_omni import QwenOmniAdapter

        adapter = QwenOmniAdapter()
        normalized = adapter._normalize_max_memory_keys(
            {
                "cuda:0": "46GiB",
                "cuda:1": "46GiB",
                "2": "30GiB",
                "cpu": "128GiB",
                "disk": "200GiB",
            }
        )
        assert normalized == {
            0: "46GiB",
            1: "46GiB",
            2: "30GiB",
            "cpu": "128GiB",
            "disk": "200GiB",
        }

    def test_build_messages_wraps_system_prompt_as_text_content(self):
        """Qwen chat template expects system content as a list of typed items."""
        from src.models.qwen_omni import QwenOmniAdapter

        adapter = QwenOmniAdapter()
        request = ModelRequest(task="vqa", extra={"system_prompt": "You are helpful."})
        bundle = ModalityBundle(text_prompt="Describe this.")

        messages = adapter._build_messages(bundle, request, images=None)
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "You are helpful."

    def test_predict_uses_audio_key_and_disables_audio_generation(self):
        """predict() should send audio with 'audio' key and request text-only output."""
        from src.models.qwen_omni import QwenOmniAdapter

        captured = {}

        class FakeProcessor:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
                return "prompt"

            def __call__(self, **kwargs):
                captured["processor_kwargs"] = kwargs
                return {"input_ids": torch.tensor([[11, 12]])}

            def decode(self, tokens, skip_special_tokens=True):
                captured["decoded_tokens"] = tokens.tolist()
                return "answer"

        class FakeModel:
            def generate(self, **kwargs):
                captured["generate_kwargs"] = kwargs
                return torch.tensor([[11, 12, 13, 14]])

        adapter = QwenOmniAdapter()
        adapter.processor = FakeProcessor()
        adapter.model = FakeModel()
        adapter.device = "cpu"
        adapter._loaded = True

        bundle = ModalityBundle(
            text_prompt="What happens?",
            audio=torch.randn(1600),
        )
        request = ModelRequest(task="vqa")
        result = adapter.predict(bundle, request)

        assert isinstance(result, RawModelOutput)
        assert result.text == "answer"
        assert "audio" in captured["processor_kwargs"]
        assert "audios" not in captured["processor_kwargs"]
        assert captured["generate_kwargs"]["return_audio"] is False

    def test_predict_handles_tuple_generate_output(self):
        """When model returns (text_ids, audio), predict should decode text_ids only."""
        from src.models.qwen_omni import QwenOmniAdapter

        class FakeProcessor:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
                return "prompt"

            def __call__(self, **kwargs):
                return {"input_ids": torch.tensor([[1, 2]])}

            def decode(self, tokens, skip_special_tokens=True):
                return "tuple-answer"

        class FakeModel:
            def generate(self, **kwargs):
                text_ids = torch.tensor([[1, 2, 3]])
                fake_audio = torch.zeros(10)
                return (text_ids, fake_audio)

        adapter = QwenOmniAdapter()
        adapter.processor = FakeProcessor()
        adapter.model = FakeModel()
        adapter.device = "cpu"
        adapter._loaded = True

        result = adapter.predict(ModalityBundle(text_prompt="x"), ModelRequest(task="vqa"))
        assert result.text == "tuple-answer"


@pytest.mark.integration
class TestQwenOmniIntegration:
    """Integration tests — require model download.

    Run with: pytest -m integration -k qwen
    """

    def test_load_and_predict_text_only(self):
        """Test basic text-only prediction."""
        from src.models.qwen_omni import QwenOmniAdapter

        adapter = QwenOmniAdapter()
        adapter.load("yujiepan/qwen2.5-omni-tiny-random")

        bundle = ModalityBundle(
            text_prompt="Hello, what is this?",
        )
        request = ModelRequest(task="vqa")
        result = adapter.predict(bundle, request)

        assert isinstance(result, RawModelOutput)
        assert result.text is not None
        adapter.unload()

    def test_load_and_predict_with_images(self, sample_modality_bundle, sample_model_request):
        """Test prediction with image frames."""
        from src.models.qwen_omni import QwenOmniAdapter

        adapter = QwenOmniAdapter()
        adapter.load("yujiepan/qwen2.5-omni-tiny-random")

        result = adapter.predict(sample_modality_bundle, sample_model_request)
        assert isinstance(result, RawModelOutput)
        assert result.text is not None
        adapter.unload()

    def test_load_and_predict_with_audio(self, sample_chunk):
        """Test prediction with audio tensor."""
        from src.models.qwen_omni import QwenOmniAdapter
        from src.data.video_reader import extract_audio

        adapter = QwenOmniAdapter()
        adapter.load("yujiepan/qwen2.5-omni-tiny-random")

        audio = extract_audio(
            sample_chunk.video.path,
            start=sample_chunk.start,
            duration=sample_chunk.duration,
        )

        bundle = ModalityBundle(
            text_prompt="What sounds do you hear?",
            audio=audio,
            chunk=sample_chunk,
        )
        request = ModelRequest(task="vqa")
        result = adapter.predict(bundle, request)

        assert isinstance(result, RawModelOutput)
        assert result.text is not None
        adapter.unload()
