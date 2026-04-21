"""
TDD tests for the chunk classification experiment.

Tests prompt construction, postprocessing, and end-to-end run with mock model.
"""

import pytest

from src.core.schemas import (
    ExperimentConfig,
    RawModelOutput,
    VideoChunk,
    VideoSample,
    CanonicalPrediction,
)
from src.experiments.chunk_classification import ChunkClassificationExperiment
from tests.conftest import MockModelAdapter


@pytest.fixture
def experiment():
    return ChunkClassificationExperiment()


class TestPrepareRequest:
    def test_request_has_classify_task(self, experiment, sample_chunk, sample_config):
        request = experiment.prepare_request(sample_chunk, sample_config)
        assert request.task == "classify"

    def test_request_has_labels(self, experiment, sample_chunk, sample_config):
        request = experiment.prepare_request(sample_chunk, sample_config)
        assert request.labels == ["talking", "building", "idle"]

    def test_prompt_template_includes_labels(self, experiment, sample_chunk, sample_config):
        request = experiment.prepare_request(sample_chunk, sample_config)
        assert "talking" in request.prompt_template
        assert "building" in request.prompt_template
        assert "idle" in request.prompt_template


class TestPostprocess:
    def test_extracts_valid_label(self, experiment, sample_chunk, sample_config):
        raw = RawModelOutput(text="talking")
        pred = experiment.postprocess(raw, sample_chunk, sample_config)
        assert isinstance(pred, CanonicalPrediction)
        assert pred.label == "talking"

    def test_extracts_label_from_verbose_text(self, experiment, sample_chunk, sample_config):
        raw = RawModelOutput(text="The activity in this chunk appears to be building with lego.")
        pred = experiment.postprocess(raw, sample_chunk, sample_config)
        assert pred.label == "building"

    def test_unknown_label_defaults(self, experiment, sample_chunk, sample_config):
        raw = RawModelOutput(text="something completely unrelated")
        pred = experiment.postprocess(raw, sample_chunk, sample_config)
        assert pred.label == "unknown"

    def test_preserves_chunk_info(self, experiment, sample_chunk, sample_config):
        raw = RawModelOutput(text="idle")
        pred = experiment.postprocess(raw, sample_chunk, sample_config)
        assert pred.chunk_index == sample_chunk.index
        assert pred.chunk_start == pytest.approx(sample_chunk.start)
        assert pred.chunk_end == pytest.approx(sample_chunk.end)

    def test_raw_text_preserved(self, experiment, sample_chunk, sample_config):
        raw = RawModelOutput(text="I think the answer is talking because they are communicating")
        pred = experiment.postprocess(raw, sample_chunk, sample_config)
        assert pred.raw_text == raw.text


class TestRunExperiment:
    def test_end_to_end_with_mock(self, experiment, sample_config):
        """Full run: load video → chunk → predict → collect results."""
        from src.execution.single_device import SingleDeviceBackend

        model = MockModelAdapter(fixed_label="talking")
        model.load("mock")
        backend = SingleDeviceBackend()
        backend.setup(model)

        predictions = experiment.run(sample_config, model, backend)

        assert isinstance(predictions, list)
        assert len(predictions) > 0
        for pred in predictions:
            assert isinstance(pred, CanonicalPrediction)
            assert pred.label == "talking"

    def test_predictions_cover_all_chunks(self, experiment, sample_config):
        """Each chunk of the video should have a prediction."""
        from src.execution.single_device import SingleDeviceBackend

        model = MockModelAdapter(fixed_label="building")
        model.load("mock")
        backend = SingleDeviceBackend()
        backend.setup(model)

        predictions = experiment.run(sample_config, model, backend)

        # 6.4s video, 3s chunks, 3s stride → 2 chunks (tail 0.4s filtered)
        assert len(predictions) == 2
        assert predictions[0].chunk_index == 0
        assert predictions[1].chunk_index == 1
