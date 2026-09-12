"""Loads backend/vector-store specs out of the YAML catalog files.

The catalog is the single source of truth the orchestrator, and nothing
else, needs to change when a new project or vector store ships: add a
folder plus one entry here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.catalog.models import BackendSpec, EmbeddingServiceSpec, VectorStoreSpec


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a plain dict. Its shape is validated by the
    Pydantic models built from it, not here, so `Any` is the honest type
    for this stage."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _require_unique_ids(ids: list[str], *, kind: str) -> None:
    seen: set[str] = set()
    for entry_id in ids:
        if entry_id in seen:
            raise ValueError(f"duplicate {kind} id in catalog: {entry_id!r}")
        seen.add(entry_id)


def load_backends(catalog_path: Path) -> list[BackendSpec]:
    """Load and validate every backend entry from a ``backends.yaml`` file.

    Args:
        catalog_path: Path to the YAML file.

    Returns:
        The catalog's backends, in file order. An empty list if the file
        doesn't exist.

    Raises:
        pydantic.ValidationError: If an entry fails schema validation
            (e.g. a container backend missing ``compose_file``).
        ValueError: If two entries share the same ``id``.
    """
    raw = _read_yaml(catalog_path)
    specs = [BackendSpec.model_validate(entry) for entry in raw.get("backends", [])]
    _require_unique_ids([spec.id for spec in specs], kind="backend")
    return specs


def load_vectorstores(catalog_path: Path) -> list[VectorStoreSpec]:
    """Load and validate every vector-store entry from a ``vectorstores.yaml`` file.

    Args:
        catalog_path: Path to the YAML file.

    Returns:
        The catalog's vector stores, in file order. An empty list if the
        file doesn't exist or declares none yet.

    Raises:
        pydantic.ValidationError: If an entry fails schema validation.
        ValueError: If two entries share the same ``id``.
    """
    raw = _read_yaml(catalog_path)
    specs = [VectorStoreSpec.model_validate(entry) for entry in raw.get("vectorstores", [])]
    _require_unique_ids([spec.id for spec in specs], kind="vector store")
    return specs


def load_embedding_services(catalog_path: Path) -> list[EmbeddingServiceSpec]:
    """Load and validate every entry from an ``embedding_services.yaml`` file.

    Args:
        catalog_path: Path to the YAML file.

    Returns:
        The catalog's embedding services, in file order. An empty list if
        the file doesn't exist or declares none yet.

    Raises:
        pydantic.ValidationError: If an entry fails schema validation.
        ValueError: If two entries share the same ``id``.
    """
    raw = _read_yaml(catalog_path)
    specs = [
        EmbeddingServiceSpec.model_validate(entry)
        for entry in raw.get("embedding_services", [])
    ]
    _require_unique_ids([spec.id for spec in specs], kind="embedding service")
    return specs
