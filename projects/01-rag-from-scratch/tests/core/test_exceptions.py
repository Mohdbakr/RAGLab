"""Tests for plainrag.core.exceptions."""

from __future__ import annotations

import pytest

from plainrag.core.exceptions import (
    EmbeddingError,
    EmptyIndexError,
    GenerationError,
    IndexPersistenceError,
    PlainRAGError,
)


@pytest.mark.parametrize(
    "exc_type", [EmbeddingError, GenerationError, IndexPersistenceError, EmptyIndexError]
)
class TestExceptionHierarchy:
    def test_is_a_plainrag_error(self, exc_type: type[PlainRAGError]) -> None:
        assert issubclass(exc_type, PlainRAGError)

    def test_carries_its_message(self, exc_type: type[PlainRAGError]) -> None:
        assert str(exc_type("something failed")) == "something failed"
