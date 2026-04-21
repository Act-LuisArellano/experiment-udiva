"""
VQA output: result dataclasses and JSON serialization.

Parallel to classification.py but for open-ended Q&A results.
Supports both free-text answers and structured JSON output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class VQAResult:
    """A single VQA result for one video chunk."""
    chunk_index: int
    chunk_start: float
    chunk_end: float
    question: str                                  # the prompt sent to the model
    answer: str                                    # raw text response from the model
    formatted: dict[str, Any] | None = None        # parsed structured output, null if not requested

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VQAReport:
    """Collection of VQA results for a video."""
    video_path: str
    results: list[VQAResult]

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics."""
        n_formatted = sum(1 for r in self.results if r.formatted is not None)
        return {
            "total_chunks": len(self.results),
            "formatted_answers": n_formatted,
            "free_text_answers": len(self.results) - n_formatted,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary(),
        }


def save_report(report: VQAReport, path: str) -> None:
    """Save a VQA report as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))


def load_report(path: str) -> VQAReport:
    """Load a VQA report from JSON."""
    data = json.loads(Path(path).read_text())
    results = [
        VQAResult(
            chunk_index=r["chunk_index"],
            chunk_start=r["chunk_start"],
            chunk_end=r["chunk_end"],
            question=r.get("question", ""),
            answer=r["answer"],
            formatted=r.get("formatted"),
        )
        for r in data["results"]
    ]
    return VQAReport(
        video_path=data["video_path"],
        results=results,
    )

