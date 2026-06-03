"""
Structured event recognition experiment.

This task mirrors the UDIVA-HHOI Codabench recognition format:
- predictions are grouped by segment
- each segment can contain zero or more events
- event fields depend on whether the category is verbal or nonverbal
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.capabilities import TaskRequirements
from src.core.interfaces import BaseExperiment, BaseExecutionBackend, BaseModelAdapter
from src.core.registry import EXPERIMENT_REGISTRY
from src.core.schemas import (
    CanonicalPrediction,
    ExperimentConfig,
    ModelRequest,
    RawModelOutput,
    VideoChunk,
)


VERBAL_UTTERANCE_TYPES = [
    "instruct", "explain", "clarify", "suggest", "command", "draw_attention",
    "agree", "disagree", "doubt", "resolve_conflict", "confirm_selection", "reject_selection", "discuss",
    "encourage", "reassure", "praise", "criticize", "assist",
    "seek_help", "seek_confirmation", "seek_clarification", "seek_information", "check_progress", "request",
    "declare_step", "declare_selection", "express_concern", "express_intent", "express_observation", "express_other",
    "positive_acknowledgement", "negative_acknowledgement", "other_acknowledgement",
]

NONVERBAL_HIGHLEVEL_ACTIONS = [
    "imitate", "request", "demonstrate", "positive_acknowledgement", "negative_acknowledgement", "other_acknowledgement",
    "open", "close", "assemble", "disassemble", "relocate", "select", "discard", "give", "receive", "correct", "take", "show", "play", "make_room", "organize", "prepare", "keep", "withdraw",
    "inspect_check", "draw_attention", "pay_attention", "verify",
    "search", "wait", "assist",
]

VERBAL_EVENT_KEYS = ("subject", "utterance_type", "target", "modifier", "score")
NONVERBAL_EVENT_KEYS = (
    "subject",
    "highlevel_action",
    "lowlevel_action",
    "target",
    "modifier",
    "score",
)


def _extract_json_object(text: str) -> dict[str, Any] | list[Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    obj_start = text.find("{")
    obj_end = text.rfind("}") + 1
    if obj_start >= 0 and obj_end > obj_start:
        try:
            return json.loads(text[obj_start:obj_end])
        except json.JSONDecodeError:
            pass

    arr_start = text.find("[")
    arr_end = text.rfind("]") + 1
    if arr_start >= 0 and arr_end > arr_start:
        try:
            return json.loads(text[arr_start:arr_end])
        except json.JSONDecodeError:
            pass

    return None


@EXPERIMENT_REGISTRY.register("recognition")
class RecognitionExperiment(BaseExperiment):
    """Structured event recognition over pre-defined or generated segments."""

    @property
    def requirements(self) -> TaskRequirements:
        return TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            output_type="structured",
        )

    def prepare_request(self, chunk: VideoChunk, config: ExperimentConfig) -> ModelRequest:
        category = self._get_category(config)
        prompt = self._build_prompt(category=category, chunk=chunk)
        return ModelRequest(task="recognition", prompt_template=prompt)

    def postprocess(
        self,
        raw: RawModelOutput,
        chunk: VideoChunk,
        config: ExperimentConfig,
    ) -> CanonicalPrediction:
        category = self._get_category(config)
        events = self._parse_events(raw.text or "", category=category)

        return CanonicalPrediction(
            chunk_index=chunk.index,
            chunk_start=chunk.start,
            chunk_end=chunk.end,
            label="",
            confidence=max((event.get("score", 0.0) for event in events), default=0.0),
            raw_text=raw.text or "",
            extra={
                "category": category,
                "events": events,
            },
        )

    def run(
        self,
        config: ExperimentConfig,
        model: BaseModelAdapter,
        backend: BaseExecutionBackend,
    ) -> list[CanonicalPrediction]:
        from src.data.pipeline import DataPipeline

        pipeline = DataPipeline()
        video = pipeline.load_video(config.video_path)

        segments = self._load_segments(config, video.duration)
        predictions: list[CanonicalPrediction] = []
        category = self._get_category(config)
        video_id = self._get_video_id(config)

        for index, segment in enumerate(segments):
            chunk = VideoChunk(
                video=video,
                index=index,
                start=segment["t_b"],
                end=segment["t_e"],
            )

            request = self.prepare_request(chunk, config)
            bundle = pipeline.build_modality_bundle(
                chunk,
                model.capabilities,
                request,
                fps=config.fps,
                resize=config.resize,
            )
            raw_output = backend.run_model(model, bundle, request)
            prediction = self.postprocess(raw_output, chunk, config)
            prediction.extra["segment_id"] = segment["segment_id"]
            prediction.extra["video_id"] = video_id
            prediction.extra["category"] = category
            predictions.append(prediction)

        return predictions

    def _build_prompt(self, category: str, chunk: VideoChunk) -> str:
        base = (
            f"Watch this video segment carefully (from {chunk.start:.3f}s to {chunk.end:.3f}s). "
            f"Return a JSON object with a single key 'events'. The value must be a list of zero or more event objects. "
            f"Each event must include a confidence field named 'score' with a value between 0 and 1. "
            f"If there are no events, return {{\"events\": []}}."
        )

        if category == "nonverbal":
            return (
                f"{base} For each event use the fields: subject, highlevel_action, lowlevel_action, target, modifier, score. "
                f"Allowed subject values include participant_a and participant_b. "
                f"Allowed highlevel_action values: {', '.join(NONVERBAL_HIGHLEVEL_ACTIONS)}. "
                f"Return JSON only."
            )

        return (
            f"{base} For each event use the fields: subject, utterance_type, target, modifier, score. "
            f"Allowed subject values include participant_a and participant_b. "
            f"Allowed utterance_type values: {', '.join(VERBAL_UTTERANCE_TYPES)}. "
            f"Return JSON only."
        )

    def _get_category(self, config: ExperimentConfig) -> str:
        return str(config.extra.get("recognition_category", "verbal")).strip().lower() or "verbal"

    def _get_video_id(self, config: ExperimentConfig) -> str:
        explicit_video_id = str(config.extra.get("recognition_video_id", "")).strip()
        if explicit_video_id:
            return explicit_video_id
        return Path(config.video_path).stem

    def _load_segments(self, config: ExperimentConfig, video_duration: float) -> list[dict[str, Any]]:
        manifest_path = str(config.extra.get("recognition_template_path", "")).strip()
        category = self._get_category(config)
        video_id = self._get_video_id(config)

        if not manifest_path:
            return self._generate_segments(config, video_duration)

        manifest = json.loads(Path(manifest_path).read_text())
        category_data = manifest.get(category, {})
        segment_data = category_data.get(video_id)
        if segment_data is None:
            raise KeyError(
                f"Video id '{video_id}' not found in recognition template for category '{category}'."
            )

        return [
            {
                "segment_id": segment_id,
                "t_b": float(segment_info["t_b"]),
                "t_e": float(segment_info["t_e"]),
            }
            for segment_id, segment_info in sorted(segment_data.items())
        ]

    def _generate_segments(self, config: ExperimentConfig, video_duration: float) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        start = 0.0
        index = 1
        while start < video_duration:
            end = min(start + config.chunk_duration, video_duration)
            if end - start < 1.0:
                break
            segments.append(
                {
                    "segment_id": f"s_{index:04d}",
                    "t_b": start,
                    "t_e": end,
                }
            )
            start += config.chunk_stride
            index += 1
        return segments

    def _parse_events(self, text: str, category: str) -> list[dict[str, Any]]:
        parsed = _extract_json_object(text)
        if isinstance(parsed, dict):
            events = parsed.get("events", [])
        elif isinstance(parsed, list):
            events = parsed
        else:
            return []

        if not isinstance(events, list):
            return []

        keys = NONVERBAL_EVENT_KEYS if category == "nonverbal" else VERBAL_EVENT_KEYS
        class_key = "highlevel_action" if category == "nonverbal" else "utterance_type"
        normalized: list[dict[str, Any]] = []

        for event in events:
            if not isinstance(event, dict):
                continue
            if not event.get(class_key):
                continue

            normalized_event = {key: event.get(key) for key in keys if key in event}
            normalized_event.setdefault("subject", event.get("subject", "participant_a"))
            normalized_event.setdefault("target", event.get("target", "unknown"))
            normalized_event.setdefault("modifier", event.get("modifier", "none"))
            if category == "nonverbal":
                normalized_event.setdefault("lowlevel_action", event.get("lowlevel_action", "none"))

            score = event.get("score", 1.0)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 1.0
            normalized_event["score"] = max(0.0, min(1.0, score))
            normalized.append(normalized_event)

        return normalized