"""Tests for app.catalog.loader, written before the implementation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.catalog.loader import load_backends, load_vectorstores
from app.catalog.models import BackendSpec, VectorStoreSpec

REPO_ROOT = Path(__file__).resolve().parents[3]


def write_yaml(path: Path, data: object) -> None:
    path.write_text(yaml.safe_dump(data))


class TestLoadBackends:
    def test_loads_a_well_formed_catalog(self, tmp_path: Path) -> None:
        write_yaml(
            tmp_path / "backends.yaml",
            {
                "backends": [
                    {
                        "id": "01-rag-from-scratch",
                        "name": "RAG From Scratch",
                        "level": "basic",
                        "summary": "No frameworks.",
                        "path": "projects/01-rag-from-scratch",
                        "needs_container": False,
                        "base_url": "http://localhost:9001",
                        "health_path": "/",
                        "compatible_vector_stores": [],
                        "stateful": False,
                        "status": "planned",
                    }
                ]
            },
        )

        backends = load_backends(tmp_path / "backends.yaml")

        assert backends == [
            BackendSpec(
                id="01-rag-from-scratch",
                name="RAG From Scratch",
                level="basic",
                summary="No frameworks.",
                path="projects/01-rag-from-scratch",
                needs_container=False,
                base_url="http://localhost:9001",
                health_path="/",
                compatible_vector_stores=[],
                stateful=False,
                status="planned",
            )
        ]

    def test_rejects_a_container_backend_missing_compose_file(
        self, tmp_path: Path
    ) -> None:
        write_yaml(
            tmp_path / "backends.yaml",
            {
                "backends": [
                    {
                        "id": "broken",
                        "name": "Broken",
                        "level": "basic",
                        "summary": "x",
                        "path": "projects/broken",
                        "needs_container": True,
                        # compose_file omitted on purpose
                        "base_url": "http://localhost:1",
                        "health_path": "/",
                    }
                ]
            },
        )

        with pytest.raises(ValidationError, match="compose_file"):
            load_backends(tmp_path / "backends.yaml")

    def test_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        entry = {
            "id": "dup",
            "name": "Dup",
            "level": "basic",
            "summary": "x",
            "path": "projects/dup",
            "needs_container": False,
            "base_url": "http://localhost:1",
            "health_path": "/",
        }
        write_yaml(tmp_path / "backends.yaml", {"backends": [entry, entry]})

        with pytest.raises(ValueError, match="duplicate"):
            load_backends(tmp_path / "backends.yaml")

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        assert load_backends(tmp_path / "does-not-exist.yaml") == []

    def test_loads_the_real_repo_catalog(self) -> None:
        """Guard against schema drift between this loader and the real file."""
        backends = load_backends(REPO_ROOT / "catalog" / "backends.yaml")
        assert any(b.id == "02-document-qa-foundation" for b in backends)


class TestLoadVectorStores:
    def test_loads_a_well_formed_catalog(self, tmp_path: Path) -> None:
        write_yaml(
            tmp_path / "vectorstores.yaml",
            {
                "vectorstores": [
                    {
                        "id": "chroma",
                        "name": "Chroma",
                        "compose_path": "platform/vectorstores/chroma/docker-compose.yml",
                        "host": "localhost",
                        "port": 8000,
                        "health_path": "/api/v1/heartbeat",
                        "status": "planned",
                    }
                ]
            },
        )

        stores = load_vectorstores(tmp_path / "vectorstores.yaml")

        assert stores == [
            VectorStoreSpec(
                id="chroma",
                name="Chroma",
                compose_path="platform/vectorstores/chroma/docker-compose.yml",
                host="localhost",
                port=8000,
                health_path="/api/v1/heartbeat",
                status="planned",
            )
        ]

    def test_empty_catalog_is_valid(self, tmp_path: Path) -> None:
        write_yaml(tmp_path / "vectorstores.yaml", {"vectorstores": []})
        assert load_vectorstores(tmp_path / "vectorstores.yaml") == []

    def test_loads_the_real_repo_catalog(self) -> None:
        stores = load_vectorstores(REPO_ROOT / "catalog" / "vectorstores.yaml")
        assert stores == []  # nothing shipped yet, per the plan
