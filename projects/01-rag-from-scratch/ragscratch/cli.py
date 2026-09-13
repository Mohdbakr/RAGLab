"""Command-line interface: ingest a file, then ask questions about it."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from ragscratch.index import CosineSimilarityIndex
from ragscratch.logging_setup import get_logger
from ragscratch.rag import ask as ask_question
from ragscratch.rag import ingest_text

app = typer.Typer(help="RAG from scratch: chunk, embed, retrieve, and answer — no framework.")
log = get_logger()

_DEFAULT_INDEX_PATH = Path(".ragscratch_index.json")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to a text file to ingest."),
    index_path: Path = typer.Option(_DEFAULT_INDEX_PATH, help="Where to store the index."),
    chunk_size: int = typer.Option(200, help="Words per chunk."),
    overlap: int = typer.Option(20, help="Words shared between consecutive chunks."),
    embedding_model: str = typer.Option(
        "text-embedding-3-small", help="Any litellm-supported embedding model."
    ),
) -> None:
    """Chunk, embed, and add a text file to the on-disk index."""
    text = path.read_text()
    index = CosineSimilarityIndex.load(index_path)
    added = asyncio.run(
        ingest_text(
            index,
            text,
            source=str(path),
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_model=embedding_model,
        )
    )
    index.save(index_path)
    typer.echo(f"Added {added} chunk(s) from {path} to {index_path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask."),
    index_path: Path = typer.Option(_DEFAULT_INDEX_PATH, help="Index to search."),
    k: int = typer.Option(4, help="Number of chunks to retrieve."),
    embedding_model: str = typer.Option(
        "text-embedding-3-small", help="Any litellm-supported embedding model."
    ),
    chat_model: str = typer.Option("gpt-4o-mini", help="Any litellm-supported chat model."),
) -> None:
    """Answer a question, grounded in whatever's been ingested."""
    index = CosineSimilarityIndex.load(index_path)
    if len(index) == 0:
        log.warning("The index at {} is empty — run `ragscratch ingest` first.", index_path)
        raise typer.Exit(code=1)
    result = asyncio.run(
        ask_question(index, question, k=k, embedding_model=embedding_model, chat_model=chat_model)
    )
    typer.echo(result.answer)
    if result.sources:
        typer.echo("\nSources:")
        for i, scored in enumerate(result.sources, start=1):
            typer.echo(f"  [{i}] {scored.chunk.source} (score={scored.score:.3f})")


if __name__ == "__main__":  # pragma: no cover
    app()
