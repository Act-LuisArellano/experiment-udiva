"""
TDD tests for the output layer.

Tests classification result creation, report building, and JSON serialization.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.output.classification import (
    ClassificationResult,
    ClassificationReport,
    save_report,
    load_report,
)


@pytest.fixture
def sample_results():
    return [
        ClassificationResult(chunk_index=0, chunk_start=0.0, chunk_end=3.0, label="talking", confidence=0.9, raw_text="talking"),
        ClassificationResult(chunk_index=1, chunk_start=3.0, chunk_end=6.0, label="building", confidence=0.7, raw_text="building"),
        ClassificationResult(chunk_index=2, chunk_start=6.0, chunk_end=9.0, label="talking", confidence=0.85, raw_text="talking"),
    ]


class TestClassificationResult:
    def test_creation(self):
        result = ClassificationResult(
            chunk_index=0, chunk_start=0.0, chunk_end=3.0,
            label="talking", confidence=0.9, raw_text="talking",
        )
        assert result.label == "talking"
        assert result.confidence == 0.9

    def test_to_dict(self):
        result = ClassificationResult(
            chunk_index=0, chunk_start=0.0, chunk_end=3.0,
            label="idle", confidence=0.5, raw_text="idle",
        )
        d = result.to_dict()
        assert d["label"] == "idle"
        assert d["chunk_index"] == 0
        assert "confidence" in d


class TestClassificationReport:
    def test_creation(self, sample_results):
        report = ClassificationReport(
            video_path="test.mp4",
            results=sample_results,
        )
        assert len(report.results) == 3
        assert report.video_path == "test.mp4"

    def test_summary(self, sample_results):
        report = ClassificationReport(video_path="test.mp4", results=sample_results)
        summary = report.summary()
        assert summary["total_chunks"] == 3
        assert "talking" in summary["label_counts"]
        assert summary["label_counts"]["talking"] == 2
        assert summary["label_counts"]["building"] == 1

    def test_to_dict(self, sample_results):
        report = ClassificationReport(video_path="test.mp4", results=sample_results)
        d = report.to_dict()
        assert "video_path" in d
        assert "results" in d
        assert "summary" in d
        assert len(d["results"]) == 3


class TestSaveLoadReport:
    def test_save_and_load_roundtrip(self, sample_results, tmp_path):
        report = ClassificationReport(video_path="test.mp4", results=sample_results)
        path = tmp_path / "output.json"
        save_report(report, str(path))

        # Verify file exists and is valid JSON
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["video_path"] == "test.mp4"
        assert len(data["results"]) == 3

    def test_load_report(self, sample_results, tmp_path):
        report = ClassificationReport(video_path="test.mp4", results=sample_results)
        path = tmp_path / "output.json"
        save_report(report, str(path))

        loaded = load_report(str(path))
        assert isinstance(loaded, ClassificationReport)
        assert len(loaded.results) == 3
        assert loaded.video_path == "test.mp4"

    def test_save_creates_directories(self, sample_results, tmp_path):
        report = ClassificationReport(video_path="test.mp4", results=sample_results)
        path = tmp_path / "nested" / "dir" / "output.json"
        save_report(report, str(path))
        assert path.exists()
