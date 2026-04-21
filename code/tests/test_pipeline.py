"""
End-to-end pipeline test and template for future experiment tests.

This file serves two purposes:
1. An integration test that verifies the full pipeline (config → experiment → output)
2. A template showing how to write tests for new experiments

Original test preserved below for reference.
"""

import pytest
import json
import torch
from pathlib import Path

from src.core.schemas import ExperimentConfig, CanonicalPrediction
from src.output.classification import ClassificationReport


# ── E2E Test: Full Pipeline with Mock Model ────────────────────────────────

class TestEndToEndPipeline:
    """Full pipeline test: config → experiment → report.

    Uses mock model so no GPU or model download is needed.
    This is the template for testing any new experiment type.
    """

    def test_full_pipeline_mock(self, sample_config, tmp_path):
        """Run the complete pipeline end-to-end with a mock model."""
        from tests.conftest import MockModelAdapter
        from src.experiments.chunk_classification import ChunkClassificationExperiment
        from src.execution.single_device import SingleDeviceBackend
        from src.output.classification import ClassificationResult, save_report
        from main import predictions_to_report

        # 1. Setup components
        experiment = ChunkClassificationExperiment()
        model = MockModelAdapter(fixed_label="talking")
        model.load("mock")
        backend = SingleDeviceBackend()
        backend.setup(model)

        # 2. Run experiment
        predictions = experiment.run(sample_config, model, backend)

        # 3. Verify predictions
        assert len(predictions) > 0
        for pred in predictions:
            assert isinstance(pred, CanonicalPrediction)
            assert pred.label in sample_config.labels + ["unknown"]

        # 4. Build and save report
        report = predictions_to_report(predictions, sample_config.video_path)
        output_path = tmp_path / "test_output.json"
        save_report(report, str(output_path))

        # 5. Verify output file
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "results" in data
        assert "summary" in data
        assert data["summary"]["total_chunks"] == len(predictions)

    def test_config_driven_pipeline(self, sample_config, tmp_path):
        """Test that the orchestrator can run from a config object."""
        from main import run_experiment

        sample_config.output_path = str(tmp_path / "config_driven_output.json")
        report = run_experiment(sample_config, use_mock=True)

        assert isinstance(report, ClassificationReport)
        assert len(report.results) > 0
        assert Path(sample_config.output_path).exists()


# ── Template: How to add tests for a new experiment ────────────────────────
#
# class TestNewExperiment:
#     """Template for testing a new experiment type.
#
#     Steps:
#     1. Create your experiment class in src/experiments/
#     2. Register it with @EXPERIMENT_REGISTRY.register("your_experiment")
#     3. Copy this template and modify for your experiment's specifics
#     """
#
#     def test_prepare_request(self):
#         from src.experiments.your_experiment import YourExperiment
#         experiment = YourExperiment()
#         request = experiment.prepare_request(chunk, config)
#         assert request.task == "your_task"
#
#     def test_postprocess(self):
#         experiment = YourExperiment()
#         raw = RawModelOutput(text="some output")
#         pred = experiment.postprocess(raw, chunk, config)
#         assert isinstance(pred, CanonicalPrediction)
#
#     def test_end_to_end(self):
#         # Use MockModelAdapter for unit tests
#         # Use @pytest.mark.integration for real model tests
#         pass


# ── Legacy test preserved as reference ─────────────────────────────────────

def test_model_output_shape():
    """Original processor test — kept as reference for simple model testing.

    This demonstrates how to test a minimal torch model directly.
    For the experiment framework, use the patterns above instead.
    """
    from src.processor import VideoModel

    print("Testing model output shape...")
    dummy_video = torch.randn(1, 3, 64, 64)

    model = VideoModel(model_name="tiny_stub")
    output = model.predict(dummy_video)

    assert output.shape == (1, 10)