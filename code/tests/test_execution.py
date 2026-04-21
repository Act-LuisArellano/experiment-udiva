"""
TDD tests for the execution backend.

Tests device detection, model setup, and inference execution.
"""

import pytest

from src.core.schemas import ModalityBundle, ModelRequest, RawModelOutput
from src.execution.single_device import SingleDeviceBackend
from tests.conftest import MockModelAdapter


class TestSingleDeviceBackend:
    def test_setup(self):
        backend = SingleDeviceBackend()
        model = MockModelAdapter()
        model.load("test")
        backend.setup(model)
        assert backend.device is not None

    def test_run_model(self, mock_model_adapter, sample_modality_bundle, sample_model_request):
        backend = SingleDeviceBackend()
        backend.setup(mock_model_adapter)
        result = backend.run_model(mock_model_adapter, sample_modality_bundle, sample_model_request)
        assert isinstance(result, RawModelOutput)
        assert result.text == "talking"

    def test_teardown(self):
        backend = SingleDeviceBackend()
        model = MockModelAdapter()
        model.load("test")
        backend.setup(model)
        backend.teardown()  # should not raise

    def test_device_is_valid(self):
        backend = SingleDeviceBackend()
        model = MockModelAdapter()
        model.load("test")
        backend.setup(model)
        assert backend.device in ("cuda", "mps", "cpu")
