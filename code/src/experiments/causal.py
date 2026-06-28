"""
Causal QA experiment (UDIVA-HHOI Task 5).

For each annotated EFFECT, the model observes [max(0, t_b - lookback), t_e] of the
video and must answer:
  - which option (A-E) is the CAUSE of the effect, and
  - the absolute timestamp (seconds) when that cause occurs.

To let the model emit an absolute timestamp, every sampled frame has its absolute
time burned into the top-left corner; the model reads it off the chosen frame.
Dense single-pass: ~N evenly-spaced frames over the window (raises the Qwen image
budget via request.extra["max_images"]).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
from src.experiments.recognition import _extract_json_object
from src.data.video_reader import extract_frames, frames_to_tensor, extract_audio

OPTIONS = ("A", "B", "C", "D", "E")
DEFAULT_LOOKBACK = 60.0
DEFAULT_NUM_FRAMES = 32
DEFAULT_RESIZE = 320


def _overlay_timestamps(frames_np: np.ndarray, timestamps: list[float]) -> np.ndarray:
    """Burn 't=SS.ss' into the top-left corner of each (H,W,3) uint8 frame."""
    if frames_np.shape[0] == 0:
        return frames_np
    h, w = frames_np.shape[1:3]
    font_size = max(14, h // 16)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    out = []
    for frame, ts in zip(frames_np, timestamps):
        img = Image.fromarray(frame, mode="RGB")
        draw = ImageDraw.Draw(img)
        label = f"t={ts:.2f}s"
        # solid background box for legibility
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = font_size * len(label) // 2, font_size
        draw.rectangle([0, 0, tw + 8, th + 8], fill=(0, 0, 0))
        draw.text((4, 2), label, fill=(255, 255, 0), font=font)
        out.append(np.asarray(img))
    return np.stack(out, axis=0)


@EXPERIMENT_REGISTRY.register("causal")
class CausalExperiment(BaseExperiment):
    """Multiple-choice cause selection + temporal localization."""

    @property
    def requirements(self) -> TaskRequirements:
        return TaskRequirements(
            required_modalities={"image", "text"},
            needs_generation=True,
            output_type="structured",
        )

    # Interface requirements; the real loop is in run() (per-effect windows).
    def prepare_request(self, chunk: VideoChunk, config: ExperimentConfig) -> ModelRequest:
        return ModelRequest(task="causal")

    def postprocess(self, raw: RawModelOutput, chunk: VideoChunk, config: ExperimentConfig) -> CanonicalPrediction:
        option, ts = self._parse(raw.text or "")
        return CanonicalPrediction(
            chunk_index=chunk.index, chunk_start=chunk.start, chunk_end=chunk.end,
            raw_text=raw.text or "",
            extra={"predicted_option": option, "predicted_cause_timestamp": ts},
        )

    def run(
        self,
        config: ExperimentConfig,
        model: BaseModelAdapter,
        backend: BaseExecutionBackend,
    ) -> list[CanonicalPrediction]:
        spec = json.loads(Path(config.extra["causal_spec_path"]).read_text())
        segments = spec[next(iter(spec))]                  # top key "SEGMENT"
        video_id = str(config.extra["causal_video_id"])
        effects = segments[video_id]

        video_path = Path(config.video_path)
        lookback = float(config.extra.get("causal_lookback_seconds", DEFAULT_LOOKBACK))
        num_frames = int(config.extra.get("causal_num_frames", DEFAULT_NUM_FRAMES))
        resize = int(config.extra.get("causal_resize", DEFAULT_RESIZE))
        want_audio = "audio" in model.capabilities.modalities

        predictions: list[CanonicalPrediction] = []
        for index, (effect_id, rec) in enumerate(effects.items()):
            eff = rec["effect"]
            t_b, t_e = float(eff["t_b"]), float(eff["t_e"])
            start = max(0.0, t_b - lookback)
            duration = max(t_e - start, 0.5)
            fps = num_frames / duration

            frames_np = extract_frames(video_path, start, duration, fps=fps, resize=(resize, resize))
            n = frames_np.shape[0]
            timestamps = [round(start + k / fps, 2) for k in range(n)]
            frames_np = _overlay_timestamps(frames_np, timestamps)
            frames = frames_to_tensor(frames_np)

            audio = extract_audio(video_path, start, duration) if want_audio else None

            prompt = self._build_prompt(eff, rec["options"], start, t_b, t_e)
            request = ModelRequest(
                task="causal",
                prompt_template=prompt,
                extra={"max_images": max(n, 1)},
            )
            bundle = ModalityBundle(frames=frames, text_prompt=prompt, audio=audio)

            raw = backend.run_model(model, bundle, request)
            option, ts = self._parse(raw.text or "")

            predictions.append(CanonicalPrediction(
                chunk_index=index, chunk_start=t_b, chunk_end=t_e,
                raw_text=raw.text or "",
                extra={
                    "video_id": video_id,
                    "effect_id": effect_id,
                    "predicted_option": option,
                    "predicted_cause_timestamp": ts,
                    "n_frames": n,
                },
            ))
        return predictions

    # ── prompt ──────────────────────────────────────────────────────────────────
    def _build_prompt(self, effect: dict, options: dict, obs_start: float,
                      t_b: float, t_e: float) -> str:
        opts = "\n".join(f"{k}: {options[k]}" for k in OPTIONS if k in options)
        return (
            f"You are analyzing a two-person interaction to find the CAUSE of an event.\n"
            f"The frames span {obs_start:.2f}s to {t_e:.2f}s; each frame has its absolute "
            f"timestamp (in seconds) printed in yellow in its top-left corner.\n\n"
            f"EFFECT (occurs at {t_b:.2f}s-{t_e:.2f}s): {effect.get('description','')}\n\n"
            f"Which option below is the CAUSE of this effect? Choose exactly one.\n{opts}\n\n"
            f"The cause happens BEFORE the effect. Identify the frame where the cause occurs "
            f"and read its printed timestamp.\n"
            f"Return ONLY a JSON object: {{\"option\": \"<one of A,B,C,D,E>\", "
            f"\"timestamp\": <seconds as a number>}}."
        )

    # ── parsing ─────────────────────────────────────────────────────────────────
    def _parse(self, text: str) -> tuple[str, float | None]:
        parsed = _extract_json_object(text)
        option, ts = "", None
        if isinstance(parsed, dict):
            opt = str(parsed.get("option", "")).strip().upper()
            if opt in OPTIONS:
                option = opt
            raw_ts = parsed.get("timestamp", None)
            try:
                ts = float(raw_ts) if raw_ts is not None else None
            except (TypeError, ValueError):
                ts = None
        return option, ts
