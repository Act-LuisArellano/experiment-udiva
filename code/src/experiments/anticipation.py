"""
Event anticipation experiment (UDIVA-HHOI Codabench Tracks 3 & 4).

For each evaluation window [t_b, t_e] the model OBSERVES [max(0, t_b - obs), t_b]
of the video and predicts the ordered sequence of events each participant will
perform during [t_b, t_e]. Dense single-pass: N evenly-spaced frames over the
observation window + audio (raises the Qwen image budget via
request.extra["max_images"]).

Two views:
  - exocentric (GF): one call predicts BOTH participants (subjects grounded by
    frame side: participant_a = LEFT, participant_b = RIGHT).
  - egocentric (E1/E2): set extra["anticipation_subject"]; the first-person view
    predicts ONLY the camera wearer's own upcoming events.

Output (per participant) is later converted to the official submission schema
(ordered "hypotheses" of positional event arrays):
    verbal     -> [utterance_type, target]
    non-verbal -> [highlevel_action, lowlevel_action, target]
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
    ModalityBundle,
    ModelRequest,
    RawModelOutput,
    VideoChunk,
)
from src.experiments.recognition import (
    VERBAL_UTTERANCE_TYPES,
    NONVERBAL_HIGHLEVEL_ACTIONS,
    NONVERBAL_LOWLEVEL_ACTIONS,
    _extract_json_object,
)
from src.data.video_reader import extract_frames, frames_to_tensor, extract_audio

PARTICIPANTS = ("participant_a", "participant_b")
DEFAULT_OBS_SECONDS = 60.0
DEFAULT_NUM_FRAMES = 32
DEFAULT_RESIZE = 256


@EXPERIMENT_REGISTRY.register("anticipation")
class AnticipationExperiment(BaseExperiment):
    """Predict upcoming event sequences per participant per window."""

    @property
    def requirements(self) -> TaskRequirements:
        return TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            output_type="structured",
        )

    # Interface requirements; the real loop is in run() (per-window observation).
    def prepare_request(self, chunk: VideoChunk, config: ExperimentConfig) -> ModelRequest:
        return ModelRequest(task="anticipation",
                            prompt_template=self._build_prompt(chunk.end, chunk.end + 2.0, chunk.start, ""))

    def postprocess(self, raw: RawModelOutput, chunk: VideoChunk, config: ExperimentConfig) -> CanonicalPrediction:
        events = self._parse_events(raw.text or "", PARTICIPANTS)
        return CanonicalPrediction(
            chunk_index=chunk.index, chunk_start=chunk.start, chunk_end=chunk.end,
            raw_text=raw.text or "", extra={"events": events},
        )

    def run(
        self,
        config: ExperimentConfig,
        model: BaseModelAdapter,
        backend: BaseExecutionBackend,
    ) -> list[CanonicalPrediction]:
        segments = self._load_segments(config)
        obs_seconds = float(config.extra.get("anticipation_observation_seconds", DEFAULT_OBS_SECONDS))
        num_frames = int(config.extra.get("anticipation_num_frames", DEFAULT_NUM_FRAMES))
        resize = int(config.extra.get("anticipation_resize", DEFAULT_RESIZE))
        subject = str(config.extra.get("anticipation_subject", "")).strip().lower()
        subjects = (subject,) if subject else PARTICIPANTS
        want_audio = "audio" in model.capabilities.modalities
        video_path = Path(config.video_path)
        video_id = self._get_video_id(config)

        predictions: list[CanonicalPrediction] = []
        for index, seg in enumerate(segments):
            t_b, t_e = float(seg["t_b"]), float(seg["t_e"])
            obs_start = max(0.0, t_b - obs_seconds)

            if obs_start >= t_b:
                events = {p: [] for p in PARTICIPANTS}
                raw_text = ""
            else:
                duration = t_b - obs_start
                fps = num_frames / duration
                frames_np = extract_frames(video_path, obs_start, duration, fps=fps, resize=(resize, resize))
                n = frames_np.shape[0]
                frames = frames_to_tensor(frames_np)
                audio = extract_audio(video_path, obs_start, duration) if want_audio else None

                prompt = self._build_prompt(t_b, t_e, obs_start, subject)
                request = ModelRequest(task="anticipation", prompt_template=prompt,
                                       extra={"max_images": max(n, 1)})
                bundle = ModalityBundle(frames=frames, text_prompt=prompt, audio=audio)

                raw = backend.run_model(model, bundle, request)
                raw_text = raw.text or ""
                events = self._parse_events(raw_text, subjects)

            predictions.append(CanonicalPrediction(
                chunk_index=index, chunk_start=t_b, chunk_end=t_e, raw_text=raw_text,
                extra={"segment_id": seg["segment_id"], "video_id": video_id,
                       "t_b": t_b, "t_e": t_e, "events": events},
            ))

        return predictions

    # ── prompt ────────────────────────────────────────────────────────────────
    def _build_prompt(self, t_b: float, t_e: float, obs_start: float, subject: str) -> str:
        horizon = t_e - t_b
        vocab = (
            f"Each event is an object with a 'type' field that is 'verbal' or 'nonverbal'. "
            f"A verbal event has fields: type='verbal', utterance_type, target. "
            f"A nonverbal event has fields: type='nonverbal', highlevel_action, lowlevel_action, target. "
            f"You MUST choose label values only from these lists; do not invent new labels. "
            f"utterance_type one of: {', '.join(VERBAL_UTTERANCE_TYPES)}. "
            f"highlevel_action one of: {', '.join(NONVERBAL_HIGHLEVEL_ACTIONS)}. "
            f"lowlevel_action one of: {', '.join(NONVERBAL_LOWLEVEL_ACTIONS)}. "
            f"'target' is a short free-text object or person the action is directed at. "
        )
        if subject:
            return (
                f"This is the first-person egocentric video recorded by {subject}, "
                f"spanning {obs_start:.2f}s to {t_b:.2f}s. "
                f"Based ONLY on what you have observed, ANTICIPATE what {subject} (the camera "
                f"wearer) will do during the NEXT {horizon:.1f} seconds ({t_b:.2f}s-{t_e:.2f}s). "
                f"Return a JSON object with exactly one key '{subject}' whose value is an ordered "
                f"list (earliest first) of the events you predict; empty list if none. {vocab}"
                f"Return JSON only."
            )
        return (
            f"You are watching a video from {obs_start:.2f}s to {t_b:.2f}s of a two-person "
            f"interaction. participant_a is the person on the LEFT of the frame and participant_b "
            f"is on the RIGHT. Based ONLY on what you have observed, ANTICIPATE what each participant "
            f"will do during the NEXT {horizon:.1f} seconds ({t_b:.2f}s-{t_e:.2f}s). "
            f"Return a JSON object with exactly two keys: 'participant_a' and 'participant_b'. "
            f"Each value is an ordered list (earliest first) of predicted events; empty list if none. "
            f"{vocab}Return JSON only."
        )

    # ── parsing ─────────────────────────────────────────────────────────────--
    def _parse_events(self, text: str, subjects: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
        parsed = _extract_json_object(text)
        result: dict[str, list[dict[str, Any]]] = {p: [] for p in PARTICIPANTS}
        if not isinstance(parsed, dict):
            return result
        for p in subjects:
            raw_list = parsed.get(p, [])
            if not isinstance(raw_list, list):
                continue
            for ev in raw_list:
                if not isinstance(ev, dict):
                    continue
                etype = str(ev.get("type", "")).strip().lower()
                if etype == "verbal" and ev.get("utterance_type"):
                    result[p].append({
                        "type": "verbal",
                        "utterance_type": str(ev.get("utterance_type", "")),
                        "target": str(ev.get("target", "unclear")),
                    })
                elif etype == "nonverbal" and ev.get("highlevel_action"):
                    result[p].append({
                        "type": "nonverbal",
                        "highlevel_action": str(ev.get("highlevel_action", "")),
                        "lowlevel_action": str(ev.get("lowlevel_action", "none")),
                        "target": str(ev.get("target", "unclear")),
                    })
        return result

    # ── segments / ids ─────────────────────────────────────────────────────────
    def _load_segments(self, config: ExperimentConfig) -> list[dict[str, Any]]:
        template_path = str(config.extra.get("anticipation_template_path", "")).strip()
        if not template_path:
            raise ValueError("anticipation requires extra.anticipation_template_path")
        manifest = json.loads(Path(template_path).read_text())
        video_id = self._get_video_id(config)
        video_segs = manifest["anticipation"].get(video_id)
        if video_segs is None:
            raise KeyError(f"video_id '{video_id}' not in anticipation windows")
        return [
            {"segment_id": sid, "t_b": s["t_b"], "t_e": s["t_e"]}
            for sid, s in sorted(video_segs.items())
        ]

    def _get_video_id(self, config: ExperimentConfig) -> str:
        explicit = str(config.extra.get("anticipation_video_id", "")).strip()
        return explicit or Path(config.video_path).stem
