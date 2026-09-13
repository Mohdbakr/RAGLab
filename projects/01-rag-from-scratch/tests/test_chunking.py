"""Tests for ragscratch.chunking, written before the implementation."""

from __future__ import annotations

import pytest

from ragscratch.chunking import chunk_text


class TestChunkText:
    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("", chunk_size=10, overlap=2) == []

    def test_text_shorter_than_chunk_size_is_a_single_chunk(self) -> None:
        assert chunk_text("one two three", chunk_size=10, overlap=2) == ["one two three"]

    def test_text_exactly_chunk_size_is_a_single_chunk(self) -> None:
        text = " ".join(f"w{i}" for i in range(10))
        assert chunk_text(text, chunk_size=10, overlap=2) == [text]

    def test_splits_longer_text_into_overlapping_word_windows(self) -> None:
        words = [f"w{i}" for i in range(25)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=10, overlap=2)

        assert chunks == [
            " ".join(words[0:10]),
            " ".join(words[8:18]),
            " ".join(words[16:25]),
        ]

    def test_never_produces_an_empty_chunk_and_the_last_one_reaches_the_final_word(self) -> None:
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=10, overlap=2)

        assert all(chunk.strip() for chunk in chunks)
        assert chunks[-1].split()[-1] == words[-1]

    def test_collapses_arbitrary_whitespace_between_words(self) -> None:
        assert chunk_text("one   two\nthree\t\tfour", chunk_size=10, overlap=2) == [
            "one two three four"
        ]

    @pytest.mark.parametrize(
        ("chunk_size", "overlap"),
        [
            (0, 0),
            (-5, 0),
            (10, 10),
            (10, 11),
            (10, -1),
        ],
    )
    def test_rejects_invalid_chunk_size_or_overlap(self, chunk_size: int, overlap: int) -> None:
        with pytest.raises(ValueError, match="chunk_size|overlap"):
            chunk_text("some words here", chunk_size=chunk_size, overlap=overlap)
