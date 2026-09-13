"""Shared KPI logging so every RAGLab project reports comparable metrics.

Each project calls :class:`BenchmarkLogger` around its retrieval and
generation calls. Events land as JSON Lines under a per-project file so the
platform frontend's Benchmarks page can load every project's history with a
single glob, without any project depending on the others.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

BenchmarkOperation = Literal["ingest", "retrieve", "generate", "embed"]


def _utcnow() -> datetime:
    """Return the current time in UTC.

    Returns:
        The current UTC timestamp.
    """
    return datetime.now(UTC)


class BenchmarkEvent(BaseModel):
    """One measured operation, ready to be compared across projects/stores.

    Attributes:
        event_id: Unique identifier, generated automatically.
        timestamp: UTC time the event was recorded.
        project_id: Catalog id of the project that recorded this event, e.g.
            ``"03-production-rag-reference"``.
        vector_store_id: Catalog id of the vector store involved, if any.
        operation: Which pipeline stage this measures.
        latency_ms: Wall-clock duration of the operation, in milliseconds.
        prompt_tokens: Prompt tokens consumed, if this was an LLM call.
        completion_tokens: Completion tokens produced, if this was an LLM
            call.
        estimated_cost_usd: Estimated USD cost of the call, if known.
        retrieved_k: Number of chunks/documents retrieved, if applicable.
        extra: Free-form additional context (model name, error message,
            etc.) that doesn't warrant its own column.
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=_utcnow)
    project_id: str
    vector_store_id: str | None = None
    operation: BenchmarkOperation
    latency_ms: float = Field(ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    retrieved_k: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] = Field(default_factory=dict)


class BenchmarkLogger:
    """Appends :class:`BenchmarkEvent` records to a per-project JSONL file."""

    def __init__(self, project_id: str, output_dir: Path | str = Path("benchmarks")) -> None:
        """Create a logger scoped to one project.

        Args:
            project_id: Catalog id of the project doing the logging. Events
                logged through this instance must carry the same id.
            output_dir: Directory the JSONL file is written into. Created
                if it doesn't already exist.
        """
        self._project_id = project_id
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._output_dir / f"{project_id}.jsonl"

    def log(self, event: BenchmarkEvent) -> None:
        """Append one event to this project's JSONL file.

        Args:
            event: The event to record. Its ``project_id`` must match the
                project this logger was constructed for.

        Raises:
            ValueError: If the event's ``project_id`` doesn't match this
                logger's project.
        """
        if event.project_id != self._project_id:
            raise ValueError(
                f"project_id mismatch: logger is scoped to {self._project_id!r}, "
                f"event has {event.project_id!r}"
            )
        with self._log_file.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json())
            fh.write("\n")

    @contextmanager
    def measure(
        self,
        operation: BenchmarkOperation,
        *,
        vector_store_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Time a block of code and log it as a :class:`BenchmarkEvent`.

        The yielded dict accepts any :class:`BenchmarkEvent` field
        (``prompt_tokens``, ``retrieved_k``, ...) to be filled in by the
        caller before the block exits; unrecognized keys are stored under
        ``extra``. The event is logged even if the block raises, with the
        exception message recorded under ``extra["error"]``.

        Args:
            operation: Which pipeline stage this measures.
            vector_store_id: Catalog id of the vector store involved, if any.

        Yields:
            A mutable dict the caller fills in with additional event fields.
        """
        fields: dict[str, Any] = {}
        started_at = time.perf_counter()
        try:
            yield fields
        except Exception as exc:
            fields.setdefault("extra", {})["error"] = str(exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - started_at) * 1000
            known_fields = {
                "prompt_tokens",
                "completion_tokens",
                "estimated_cost_usd",
                "retrieved_k",
            }
            extra = dict(fields.pop("extra", {}))
            for key in list(fields):
                if key not in known_fields:
                    extra[key] = fields.pop(key)
            self.log(
                BenchmarkEvent(
                    project_id=self._project_id,
                    vector_store_id=vector_store_id,
                    operation=operation,
                    latency_ms=latency_ms,
                    extra=extra,
                    **fields,
                )
            )
