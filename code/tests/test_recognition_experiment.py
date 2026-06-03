"""
TDD tests for the recognition experiment.

The recognition task follows the Codabench UDIVA-HHOI format:
- segments come from a template/reference JSON manifest
- each segment may contain multiple structured events
- predictions are emitted as structured event lists with confidence scores
"""

from __future__ import annotations

import json

import pytest

from src.core.schemas import (
    CanonicalPrediction,
    ExperimentConfig,
    ModalityBundle,
    RawModelOutput,
    VideoSample,
)
from tests.conftest import MockModelAdapter, VIDEO_PATH


@pytest.fixture
def recognition_manifest(tmp_path):
    path = tmp_path / "recognition_template.json"
    path.write_text(
        json.dumps(
            {
                "verbal": {
                    "001080": {
                        "s_0001": {"t_b": 0.0, "t_e": 2.0, "events": []},
                        "s_0002": {"t_b": 3.0, "t_e": 5.0, "events": []},
                    }
                }
            }
        )
    )
    return path


@pytest.fixture
def recognition_config(recognition_manifest):
    return ExperimentConfig(
        experiment_type="recognition",
        model_name="mock",
        model_checkpoint="mock-checkpoint",
        video_path=str(VIDEO_PATH),
        chunk_duration=2.0,
        chunk_stride=2.0,
        fps=1.0,
        resize=(112, 112),
        backend="single_device",
        output_format="json",
        output_path="results/test_recognition_output.json",
        extra={
            "recognition_category": "verbal",
            "recognition_video_id": "001080",
            "recognition_template_path": str(recognition_manifest),
        },
    )


@pytest.fixture
def experiment():
    from src.experiments.recognition import RecognitionExperiment

    return RecognitionExperiment()


class TestPrepareRequest:
    def test_prompt_includes_category_schema(self, experiment, sample_chunk, recognition_config):
        request = experiment.prepare_request(sample_chunk, recognition_config)
        assert request.task == "recognition"
        assert "utterance_type" in request.prompt_template
        assert "participant_a" in request.prompt_template


class TestPostprocess:
    def test_parses_structured_events(self, experiment, sample_chunk, recognition_config):
        raw = RawModelOutput(
            text=json.dumps(
                {
                    "events": [
                        {
                            "subject": "participant_a",
                            "utterance_type": "suggest",
                            "target": "model",
                            "modifier": "none",
                            "score": 0.9,
                        }
                    ]
                }
            )
        )

        pred = experiment.postprocess(raw, sample_chunk, recognition_config)

        assert isinstance(pred, CanonicalPrediction)
        assert pred.label == ""
        assert pred.extra["events"][0]["utterance_type"] == "suggest"
        assert pred.extra["events"][0]["score"] == pytest.approx(0.9)


class TestRunExperiment:
    def test_end_to_end_with_mock(self, experiment, recognition_config, monkeypatch):
        from src.execution.single_device import SingleDeviceBackend
        from src.data.pipeline import DataPipeline

        fake_video = VideoSample(
            path=VIDEO_PATH,
            duration=5.0,
            fps=1.0,
            width=112,
            height=112,
            has_audio=False,
        )

        monkeypatch.setattr(DataPipeline, "load_video", lambda self, path: fake_video)
        monkeypatch.setattr(
            DataPipeline,
            "build_modality_bundle",
            lambda self, chunk, model_capabilities, request, fps=2.0, resize=None: ModalityBundle(
                frames=None,
                text_prompt=request.prompt_template,
                audio=None,
                chunk=chunk,
            ),
        )

        model = MockModelAdapter(
            fixed_label=json.dumps(
                {
                    "events": [
                        {
                            "subject": "participant_a",
                            "utterance_type": "suggest",
                            "target": "model",
                            "modifier": "none",
                            "score": 0.9,
                        }
                    ]
                }
            )
        )
        model.load("mock")
        backend = SingleDeviceBackend()
        backend.setup(model)

        predictions = experiment.run(recognition_config, model, backend)

        assert len(predictions) == 2
        assert predictions[0].extra["segment_id"] == "s_0001"
        assert predictions[1].extra["segment_id"] == "s_0002"
        assert predictions[0].extra["events"][0]["utterance_type"] == "suggest"