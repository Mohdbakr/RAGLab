"""Exceptions for failures talking to the network or disk.

Argument-validation errors on pure functions (e.g. chunk_text's bad
chunk_size/overlap) stay plain ValueError — this hierarchy is
specifically for runtime/I/O failures at a service boundary.
"""

from __future__ import annotations


class PlainRAGError(Exception):
    """Base class for all plainrag errors."""


class EmbeddingError(PlainRAGError):
    """An embedding provider call failed."""


class GenerationError(PlainRAGError):
    """A chat completion call failed."""


class IndexPersistenceError(PlainRAGError):
    """The on-disk index could not be loaded or saved."""


class EmptyIndexError(PlainRAGError):
    """Asked a question against an index with nothing ingested."""
