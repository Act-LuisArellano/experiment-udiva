"""
TDD tests for the VQA experiment.

Tests prompt loading, request building, postprocessing (both free-text
and structured JSON), and end-to-end runs with a mock model.
"""

import json
import pytest
from pathlib import Path
from typing import Any

from src.core.schemas import (
    ExperimentConfig,
    RawModelOutput,
    VideoChunk,
    VideoSample,
    CanonicalPrediction,
)
from src.experiments.vqa import VQAExperiment, _load_prompt_function
from tests.conftest import MockModelAdapter, VIDEO_PATH


# ── Fixtures ───────────────────────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "configs" / "prompts"


@pytest.fixture
def experiment():
    return VQAExperiment()


@pytest.fixture
def vqa_config():
    """VQA experiment config pointing to the example prompt file."""
    return ExperimentConfig(
        experiment_type="vqa",
        prompt_file=str(PROMPTS_DIR / "vqa_prompt.py"),
        system_prompt_file="",
        output_schema=None,
        model_name="mock",
        model_checkpoint="mock-checkpoint",
        video_path=str(VIDEO_PATH),
        chunk_duration=3.0,
        chunk_stride=3.0,
        fps=2.0,
        resize=(224, 224),
        backend="single_device",
        output_format="json",
        output_path="results/test_vqa_output.json",
    )


@pytest.fixture
def vqa_config_structured():
    """VQA config with structured output schema."""
    return ExperimentConfig(
        experiment_type="vqa",
        prompt_file=str(PROMPTS_DIR / "vqa_prompt.py"),
        output_schema={
            "fields": {
                "action": "string",
                "objects": "list",
            }
        },
        model_name="mock",
        model_checkpoint="mock-checkpoint",
        video_path=str(VIDEO_PATH),
        chunk_duration=3.0,
        chunk_stride=3.0,
        fps=2.0,
        resize=(224, 224),
        backend="single_device",
        output_format="json",
        output_path="results/test_vqa_structured_output.json",
    )


@pytest.fixture
def vqa_config_dict():
    """Raw dict matching YAML structure for VQA config."""
    return {
        "experiment": {
            "type": "vqa",
            "prompt_file": str(PROMPTS_DIR / "vqa_prompt.py"),
            "output_schema": {
                "fields": {
                    "action": "string",
                    "objects": "list",
                }
            },
        },
        "model": {
            "name": "gemma_vlm",
            "checkpoint": "google/gemma-4-E2B",
            "weights_path": "../data-slow/models/Gemma/current-model-variation",
            "quantization": "2bit",
        },
        "data": {
            "video_path": str(VIDEO_PATH),
            "chunk_duration": 3.0,
            "chunk_stride": 3.0,
            "fps": 1.0,
            "resize": [112, 112],
        },
        "execution": {
            "backend": "single_device",
        },
        "output": {
            "format": "json",
            "path": "../data-slow/results/vqa_output.json",
        },
    }


# ── Prompt loading ─────────────────────────────────────────────────────────

class TestPromptLoading:
    def test_load_build_prompt(self):
        fn = _load_prompt_function(str(PROMPTS_DIR / "vqa_prompt.py"), "build_prompt")
        assert callable(fn)

    def test_load_build_system_prompt(self):
        fn = _load_prompt_function(str(PROMPTS_DIR / "vqa_prompt.py"), "build_system_prompt")
        assert callable(fn)

    def test_build_prompt_returns_string(self):
        fn = _load_prompt_function(str(PROMPTS_DIR / "vqa_prompt.py"), "build_prompt")
        result = fn(chunk_start=0.0, chunk_end=3.0, chunk_index=0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_prompt_with_schema(self):
        fn = _load_prompt_function(str(PROMPTS_DIR / "vqa_prompt.py"), "build_prompt")
        result = fn(
            chunk_start=0.0,
            chunk_end=3.0,
            chunk_index=0,
            output_schema={"fields": {"action": "string"}},
        )
        assert "JSON" in result
        assert "action" in result

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_prompt_function("/nonexistent/prompt.py", "build_prompt")

    def test_missing_function_raises(self):
        with pytest.raises(AttributeError):
            _load_prompt_function(
                str(PROMPTS_DIR / "vqa_prompt.py"), "nonexistent_function"
            )


# ── VQA Config from_dict ──────────────────────────────────────────────────

class TestVQAConfigFromDict:
    def test_parses_prompt_file(self, vqa_config_dict):
        config = ExperimentConfig.from_dict(vqa_config_dict)
        assert config.experiment_type == "vqa"
        assert config.prompt_file.endswith("vqa_prompt.py")

    def test_parses_output_schema(self, vqa_config_dict):
        config = ExperimentConfig.from_dict(vqa_config_dict)
        assert config.output_schema is not None
        assert "fields" in config.output_schema

    def test_defaults_when_no_vqa_fields(self):
        config = ExperimentConfig.from_dict({})
        assert config.prompt_file == ""
        assert config.system_prompt_file == ""
        assert config.output_schema is None


# ── Prepare request ───────────────────────────────────────────────────────

class TestPrepareRequest:
    def test_request_has_vqa_task(self, experiment, sample_chunk, vqa_config):
        fn = _load_prompt_function(vqa_config.prompt_file, "build_prompt")
        request = experiment.prepare_request(sample_chunk, vqa_config, prompt_fn=fn)
        assert request.task == "vqa"

    def test_prompt_contains_chunk_times(self, experiment, sample_chunk, vqa_config):
        fn = _load_prompt_function(vqa_config.prompt_file, "build_prompt")
        request = experiment.prepare_request(sample_chunk, vqa_config, prompt_fn=fn)
        assert "0.0" in request.prompt_template
        assert "3.0" in request.prompt_template

    def test_default_prompt_when_no_file(self, experiment, sample_chunk, vqa_config):
        """When no prompt_fn is provided, uses a default prompt."""
        request = experiment.prepare_request(sample_chunk, vqa_config, prompt_fn=None)
        assert "Describe what you see" in request.prompt_template


# ── Postprocess ────────────────────────────────────────────────────────────

class TestPostprocess:
    def test_free_text_answer(self, experiment, sample_chunk, vqa_config):
        raw = RawModelOutput(text="The people are talking to each other.")
        pred = experiment.postprocess(raw, sample_chunk, vqa_config)
        assert isinstance(pred, CanonicalPrediction)
        assert pred.label == ""  # VQA has no label
        assert pred.raw_text == "The people are talking to each other."
        assert pred.confidence == 1.0

    def test_structured_json_output(self, experiment, sample_chunk, vqa_config_structured):
        json_response = json.dumps({"action": "talking", "objects": ["chair", "table"]})
        raw = RawModelOutput(text=json_response)
        pred = experiment.postprocess(raw, sample_chunk, vqa_config_structured)
        assert "structured" in pred.extra
        assert pred.extra["structured"]["action"] == "talking"
        assert pred.extra["structured"]["objects"] == ["chair", "table"]

    def test_structured_with_surrounding_text(self, experiment, sample_chunk, vqa_config_structured):
        """Model outputs JSON wrapped in text — should still parse."""
        text = 'Here is my answer: {"action": "building", "objects": ["lego"]}'
        raw = RawModelOutput(text=text)
        pred = experiment.postprocess(raw, sample_chunk, vqa_config_structured)
        assert "structured" in pred.extra
        assert pred.extra["structured"]["action"] == "building"

    def test_structured_parse_failure(self, experiment, sample_chunk, vqa_config_structured):
        """When model outputs unparseable text with schema expected."""
        raw = RawModelOutput(text="I'm not sure what they're doing.")
        pred = experiment.postprocess(raw, sample_chunk, vqa_config_structured)
        assert "parse_error" in pred.extra
        assert pred.raw_text == "I'm not sure what they're doing."

    def test_preserves_chunk_info(self, experiment, sample_chunk, vqa_config):
        raw = RawModelOutput(text="Some answer")
        pred = experiment.postprocess(raw, sample_chunk, vqa_config)
        assert pred.chunk_index == sample_chunk.index
        assert pred.chunk_start == pytest.approx(sample_chunk.start)
        assert pred.chunk_end == pytest.approx(sample_chunk.end)


# ── End-to-end run ─────────────────────────────────────────────────────────

class TestRunExperiment:
    def test_end_to_end_with_mock(self, experiment, vqa_config):
        """Full run: load video → chunk → predict → collect results."""
        from src.execution.single_device import SingleDeviceBackend

        model = MockModelAdapter(fixed_label="The people are chatting.")
        model.load("mock")
        backend = SingleDeviceBackend()
        backend.setup(model)

        predictions = experiment.run(vqa_config, model, backend)

        assert isinstance(predictions, list)
        assert len(predictions) > 0
        for pred in predictions:
            assert isinstance(pred, CanonicalPrediction)
            assert pred.label == ""  # VQA has no label
            assert "chatting" in pred.raw_text
            # Question should be stored in extra
            assert "question" in pred.extra
            assert len(pred.extra["question"]) > 0

    def test_predictions_cover_all_chunks(self, experiment, vqa_config):
        """Each chunk of the video should have exactly one prediction."""
        from src.execution.single_device import SingleDeviceBackend

        model = MockModelAdapter(fixed_label="answer")
        model.load("mock")
        backend = SingleDeviceBackend()
        backend.setup(model)

        predictions = experiment.run(vqa_config, model, backend)

        # 6.4s video, 3s chunks, 3s stride → 2 chunks
        assert len(predictions) == 2
        assert predictions[0].chunk_index == 0
        assert predictions[1].chunk_index == 1

    def test_question_comes_from_prompt_file(self, experiment, vqa_config):
        """The question stored should match the prompt built by vqa_prompt.py."""
        from src.execution.single_device import SingleDeviceBackend

        model = MockModelAdapter(fixed_label="answer")
        model.load("mock")
        backend = SingleDeviceBackend()
        backend.setup(model)

        predictions = experiment.run(vqa_config, model, backend)
        question = predictions[0].extra["question"]
        # Should contain the prompt from vqa_prompt.py, not the generic fallback
        assert "What are the people" in question

    def test_end_to_end_no_prompt_file(self, experiment):
        """Run VQA with no prompt_file — uses default prompt."""
        from src.execution.single_device import SingleDeviceBackend

        config = ExperimentConfig(
            experiment_type="vqa",
            model_name="mock",
            model_checkpoint="mock",
            video_path=str(VIDEO_PATH),
            chunk_duration=3.0,
            chunk_stride=3.0,
            fps=2.0,
            resize=(224, 224),
        )

        model = MockModelAdapter(fixed_label="default answer")
        model.load("mock")
        backend = SingleDeviceBackend()
        backend.setup(model)

        predictions = experiment.run(config, model, backend)
        assert len(predictions) > 0
        # Default prompt question should still be stored
        assert "question" in predictions[0].extra

