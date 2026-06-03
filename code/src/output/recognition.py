"""
Recognition output: segment-level event submission format and JSON serialization.

The serialized JSON matches the Codabench UDIVA-HHOI recognition layout:
{
  "verbal": {
    "001080": {
      "s_0001": {"t_b": 0.0, "t_e": 2.0, "events": [...]}
    }
  }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class RecognitionResult:
    """Predicted events for one recognition segment."""

    segment_id: str
    chunk_index: int
    chunk_start: float
    chunk_end: float
    events: list[dict[str, Any]]
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecognitionReport:
    """Recognition submission bundle for one category and one video."""

    category: str
    video_id: str
    results: list[RecognitionResult]

    def summary(self) -> dict[str, Any]:
        total_events = sum(len(result.events) for result in self.results)
        segments_with_events = sum(1 for result in self.results if result.events)
        return {
            "total_segments": len(self.results),
            "total_events": total_events,
            "segments_with_events": segments_with_events,
        }

    def to_submission_dict(self) -> dict[str, Any]:
        return {
            self.category: {
                self.video_id: {
                    result.segment_id: {
                        "t_b": result.chunk_start,
                        "t_e": result.chunk_end,
                        "events": result.events,
                    }
                    for result in self.results
                }
            }
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_submission_dict()


def save_report(report: RecognitionReport, path: str) -> None:
    """Save a recognition submission report as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_submission_dict(), indent=2))


def load_report(path: str) -> RecognitionReport:
    """Load a recognition submission report from JSON."""
    data = json.loads(Path(path).read_text())
    category = next(iter(data.keys()))
    video_id = next(iter(data[category].keys()))
    segments = data[category][video_id]
    results = [
        RecognitionResult(
            segment_id=segment_id,
            chunk_index=index,
            chunk_start=segment_data["t_b"],
            chunk_end=segment_data["t_e"],
            events=segment_data.get("events", []),
        )
        for index, (segment_id, segment_data) in enumerate(sorted(segments.items()))
    ]
    return RecognitionReport(category=category, video_id=video_id, results=results)