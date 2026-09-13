"""Tests for ragscratch.index, written before the implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragscratch.index import CosineSimilarityIndex, DocumentChunk


def chunk(text: str, embedding: list[float], source: str = "doc.txt") -> DocumentChunk:
    return DocumentChunk(text=text, source=source, embedding=embedding)


class TestAddAndLen:
    def test_starts_empty(self) -> None:
        assert len(CosineSimilarityIndex()) == 0

    def test_add_increases_length(self) -> None:
        index = CosineSimilarityIndex()
        index.add([chunk("a", [1.0, 0.0]), chunk("b", [0.0, 1.0])])
        assert len(index) == 2


class TestSearch:
    def test_empty_index_returns_no_results(self) -> None:
        index = CosineSimilarityIndex()
        assert index.search([1.0, 0.0], k=4) == []

    def test_identical_vector_scores_close_to_one(self) -> None:
        index = CosineSimilarityIndex()
        index.add([chunk("a", [1.0, 0.0])])

        (result,) = index.search([1.0, 0.0], k=4)

        assert result.chunk.text == "a"
        assert result.score == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vector_scores_close_to_zero(self) -> None:
        index = CosineSimilarityIndex()
        index.add([chunk("a", [1.0, 0.0])])

        (result,) = index.search([0.0, 1.0], k=4)

        assert result.score == pytest.approx(0.0, abs=1e-6)

    def test_results_are_sorted_most_similar_first(self) -> None:
        index = CosineSimilarityIndex()
        index.add(
            [
                chunk("far", [0.0, 1.0]),
                chunk("near", [0.99, 0.01]),
                chunk("exact", [1.0, 0.0]),
            ]
        )

        results = index.search([1.0, 0.0], k=3)

        assert [r.chunk.text for r in results] == ["exact", "near", "far"]
        assert results[0].score >= results[1].score >= results[2].score

    def test_k_limits_the_number_of_results(self) -> None:
        index = CosineSimilarityIndex()
        index.add([chunk(str(i), [1.0, float(i)]) for i in range(10)])

        results = index.search([1.0, 0.0], k=3)

        assert len(results) == 3

    def test_k_larger_than_the_index_returns_everything(self) -> None:
        index = CosineSimilarityIndex()
        index.add([chunk("a", [1.0, 0.0]), chunk("b", [0.0, 1.0])])

        results = index.search([1.0, 0.0], k=100)

        assert len(results) == 2


class TestSaveAndLoad:
    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        index = CosineSimilarityIndex()
        index.add(
            [chunk("a", [1.0, 0.0], source="doc1.txt"), chunk("b", [0.0, 1.0], source="doc2.txt")]
        )
        index.save(path)

        loaded = CosineSimilarityIndex.load(path)

        assert len(loaded) == 2
        results = loaded.search([1.0, 0.0], k=1)
        assert results[0].chunk.text == "a"
        assert results[0].chunk.source == "doc1.txt"

    def test_loading_a_missing_file_returns_an_empty_index(self, tmp_path: Path) -> None:
        loaded = CosineSimilarityIndex.load(tmp_path / "does-not-exist.json")
        assert len(loaded) == 0
