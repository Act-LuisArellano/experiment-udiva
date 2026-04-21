"""
TDD tests for model adapters.

Tests the mock adapter conformance to the interface, and the real
GemmaVLM adapter behind an @integration marker.
"""

import pytest
from typing import Any

from src.core.schemas import ModalityBundle, ModelRequest, RawModelOutput
from src.core.interfaces import BaseModelAdapter
from src.models.base import detect_device
from tests.conftest import MockModelAdapter


class TestMockModelAdapter:
    def test_is_base_model_adapter(self, mock_model_adapter):
        assert isinstance(mock_model_adapter, BaseModelAdapter)

    def test_capabilities_has_required_keys(self, mock_model_adapter):
        caps = mock_model_adapter.capabilities
        assert isinstance(caps.modalities, set)
        assert "image" in caps.modalities
        assert caps.supports_generation is True

    def test_predict_returns_raw_model_output(self, mock_model_adapter, sample_modality_bundle, sample_model_request):
        result = mock_model_adapter.predict(sample_modality_bundle, sample_model_request)
        assert isinstance(result, RawModelOutput)
        assert result.text == "talking"

    def test_load_sets_loaded_flag(self):
        adapter = MockModelAdapter()
        assert adapter._loaded is False
        adapter.load("some-checkpoint")
        assert adapter._loaded is True

    def test_unload(self, mock_model_adapter):
        mock_model_adapter.unload()
        assert mock_model_adapter._loaded is False

    def test_predict_with_different_label(self):
        adapter = MockModelAdapter(fixed_label="building")
        adapter.load("test")
        bundle = ModalityBundle()
        request = ModelRequest(task="classify")
        result = adapter.predict(bundle, request)
        assert result.text == "building"


class TestDetectDevice:
    def test_returns_string(self):
        device = detect_device()
        assert isinstance(device, str)
        assert device in ("cuda", "mps", "cpu")


@pytest.mark.integration
class TestGemmaVLMAdapter:
    """Integration tests — require model download and GPU/MPS.

    Run with: pytest -m integration
    """

    def test_load_and_predict(self, sample_modality_bundle, sample_model_request):
        from src.models.gemma_vlm import GemmaVLMAdapter

        adapter = GemmaVLMAdapter()
        adapter.load("google/gemma-3-4b-it")

        result = adapter.predict(sample_modality_bundle, sample_model_request)
        assert isinstance(result, RawModelOutput)
        assert result.text is not None
        assert len(result.text) > 0

        adapter.unload()
