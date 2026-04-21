"""
End-to-end test for Qwen Omni VQA flow.

Validates that the VQA pipeline sends image + audio + text modalities to
the model adapter and produces a persisted VQA JSON report.
"""

from __future__ import annotations

from pathlib import Path

from src.core.capabilities import ModelCapabilities
from src.core.interfaces import BaseModelAdapter
from src.core.schemas import ExperimentConfig, ModalityBundle, ModelRequest, RawModelOutput
from src.experiments.vqa import VQAExperiment
from src.execution.single_device import SingleDeviceBackend
from src.output.vqa import load_report, save_report
from tests.conftest import VIDEO_PATH
from main import predictions_to_vqa_report


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "configs" / "prompts"


class SpyQwenOmniAdapter(BaseModelAdapter):
    """Test double that validates multimodal inputs and returns JSON text."""

    def __init__(self):
        self._loaded = False
        self.calls: list[tuple[ModalityBundle, ModelRequest]] = []

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

    def load(self, checkpoint: str, **kwargs) -> None:
        self._loaded = True

    def predict(self, bundle: ModalityBundle, request: ModelRequest) -> RawModelOutput:
        assert self._loaded is True
        assert bundle.frames is not None
        assert bundle.frames.numel() > 0
        assert bundle.audio is not None
        assert bundle.audio.numel() > 0
        assert bundle.text_prompt is not None
        assert request.task == "vqa"
        self.calls.append((bundle, request))
        return RawModelOutput(text='{"action":"talking","objects":["person"]}')

    def unload(self) -> None:
        self._loaded = False


def test_qwen_omni_vqa_end_to_end_multimodal_io(tmp_path):
    """Run VQA end-to-end with a Qwen-style multimodal adapter."""
    config = ExperimentConfig(
        experiment_type="vqa",
        prompt_file=str(PROMPTS_DIR / "vqa_prompt.py"),
        output_schema={"fields": {"action": "string", "objects": "list"}},
        model_name="qwen_omni",
        model_checkpoint="auto",
        video_path=str(VIDEO_PATH),
        chunk_duration=3.0,
        chunk_stride=3.0,
        fps=1.0,
        resize=(112, 112),
        backend="single_device",
        output_format="json",
        output_path=str(tmp_path / "qwen_omni_e2e_output.json"),
    )

    experiment = VQAExperiment()
    model = SpyQwenOmniAdapter()
    model.load("auto")
    backend = SingleDeviceBackend()
    backend.setup(model)

    predictions = experiment.run(config, model, backend)
    report = predictions_to_vqa_report(predictions, config.video_path)
    save_report(report, config.output_path)
    saved = load_report(config.output_path)

    assert len(predictions) > 0
    assert len(model.calls) == len(predictions)
    assert "structured" in predictions[0].extra
    assert predictions[0].extra["structured"]["action"] == "talking"
    assert saved.video_path == config.video_path
    assert len(saved.results) == len(predictions)
    assert len(saved.results[0].question) > 0
    assert len(saved.results[0].answer) > 0
