"""
Classification output: result dataclasses and JSON serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ClassificationResult:
    """A single chunk classification result."""
    chunk_index: int
    chunk_start: float
    chunk_end: float
    label: str
    confidence: float
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationReport:
    """Collection of classification results for a video."""
    video_path: str
    results: list[ClassificationResult]

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics."""
        label_counts: dict[str, int] = {}
        for r in self.results:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1

        avg_confidence = (
            sum(r.confidence for r in self.results) / len(self.results)
            if self.results else 0.0
        )

        return {
            "total_chunks": len(self.results),
            "label_counts": label_counts,
            "avg_confidence": avg_confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary(),
        }


def save_report(report: ClassificationReport, path: str) -> None:
    """Save a classification report as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))


def load_report(path: str) -> ClassificationReport:
    """Load a classification report from JSON."""
    data = json.loads(Path(path).read_text())
    results = [
        ClassificationResult(**r)
        for r in data["results"]
    ]
    return ClassificationReport(
        video_path=data["video_path"],
        results=results,
    )
