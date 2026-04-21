"""
TDD tests for the Qwen Omni model adapter.

Tests the QwenOmniAdapter conformance to BaseModelAdapter interface,
capabilities declaration, and predict behavior with mock/integration tests.
"""

import pytest
import torch

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
