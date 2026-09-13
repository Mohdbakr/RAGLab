"""Tests for raglab_common.benchmarking, written before the implementation.

Run with the implementation absent first to confirm these fail for the
right reason (ImportError / AttributeError), then again once
raglab_common/benchmarking.py exists, to confirm green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from raglab_common.benchmarking import BenchmarkEvent, BenchmarkLogger


def make_event(**overrides: object) -> BenchmarkEvent:
    """Build a valid BenchmarkEvent, overriding only the fields under test."""
    fields: dict[str, object] = {
        "project_id": "01-rag-from-scratch",
        "operation": "retrieve",
        "latency_ms": 42.5,
    }
    fields.update(overrides)
    return BenchmarkEvent(**fields)  # type: ignore[arg-type]


class TestBenchmarkEvent:
    def test_accepts_minimal_valid_payload(self) -> None:
        event = make_event()
        assert event.project_id == "01-rag-from-scratch"
        assert event.operation == "retrieve"
        assert event.latency_ms == 42.5
        assert event.vector_store_id is None
        assert event.event_id  # auto-generated, non-empty
        assert event.timestamp is not None

    def test_rejects_unknown_operation(self) -> None:
        with pytest.raises(ValidationError):
            make_event(operation="not-a-real-operation")

    def test_accepts_embed_operation(self) -> None:
        event = make_event(operation="embed")
        assert event.operation == "embed"

    def test_rejects_negative_latency(self) -> None:
        with pytest.raises(ValidationError):
            make_event(latency_ms=-1.0)

    def test_carries_optional_kpi_fields(self) -> None:
        event = make_event(
            vector_store_id="chroma",
            operation="generate",
            prompt_tokens=120,
            completion_tokens=48,
            estimated_cost_usd=0.0021,
            retrieved_k=5,
            extra={"model": "gpt-4o-mini"},
        )
        assert event.vector_store_id == "chroma"
        assert event.prompt_tokens == 120
        assert event.completion_tokens == 48
        assert event.estimated_cost_usd == pytest.approx(0.0021)
        assert event.retrieved_k == 5
        assert event.extra == {"model": "gpt-4o-mini"}

    def test_event_ids_are_unique(self) -> None:
        assert make_event().event_id != make_event().event_id


class TestBenchmarkLogger:
    def test_creates_output_directory_if_missing(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "does" / "not" / "exist"
        BenchmarkLogger(project_id="01-rag-from-scratch", output_dir=output_dir)
        assert output_dir.is_dir()

    def test_log_appends_one_jsonl_line_per_event(self, tmp_path: Path) -> None:
        logger = BenchmarkLogger(project_id="01-rag-from-scratch", output_dir=tmp_path)

        logger.log(make_event(latency_ms=10.0))
        logger.log(make_event(latency_ms=20.0))

        log_file = tmp_path / "01-rag-from-scratch.jsonl"
        lines = log_file.read_text().splitlines()
        assert len(lines) == 2

        first, second = (json.loads(line) for line in lines)
        assert first["latency_ms"] == 10.0
        assert second["latency_ms"] == 20.0

    def test_log_rejects_event_for_a_different_project(self, tmp_path: Path) -> None:
        logger = BenchmarkLogger(project_id="01-rag-from-scratch", output_dir=tmp_path)
        mismatched_event = make_event(project_id="03-production-rag-reference")

        with pytest.raises(ValueError, match="project_id"):
            logger.log(mismatched_event)

    def test_measure_context_manager_records_latency_and_logs(
        self, tmp_path: Path
    ) -> None:
        logger = BenchmarkLogger(project_id="01-rag-from-scratch", output_dir=tmp_path)

        with logger.measure("retrieve", vector_store_id="chroma") as fields:
            fields["retrieved_k"] = 3

        log_file = tmp_path / "01-rag-from-scratch.jsonl"
        (recorded,) = (json.loads(line) for line in log_file.read_text().splitlines())
        assert recorded["operation"] == "retrieve"
        assert recorded["vector_store_id"] == "chroma"
        assert recorded["retrieved_k"] == 3
        assert recorded["latency_ms"] >= 0.0

    def test_measure_still_logs_when_the_block_raises(self, tmp_path: Path) -> None:
        logger = BenchmarkLogger(project_id="01-rag-from-scratch", output_dir=tmp_path)

        with pytest.raises(RuntimeError), logger.measure("generate"):
            raise RuntimeError("boom")

        log_file = tmp_path / "01-rag-from-scratch.jsonl"
        assert log_file.exists()
        (recorded,) = (json.loads(line) for line in log_file.read_text().splitlines())
        assert recorded["operation"] == "generate"
        assert recorded["extra"].get("error") == "boom"
