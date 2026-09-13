"""Command-line interface: ingest a file, then ask questions about it."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from plainrag import __version__
from plainrag.core.config import get_settings
from plainrag.core.exceptions import PlainRAGError
from plainrag.core.logging import get_logger, print_banner
from plainrag.services.index import CosineSimilarityIndex
from plainrag.services.rag import ask as ask_question
from plainrag.services.rag import ingest_text

app = typer.Typer(help="RAG from scratch: chunk, embed, retrieve, and answer — no framework.")

_settings = get_settings()
log = get_logger(level=_settings.log_level, log_file=_settings.log_file)


@app.callback()
def main() -> None:
    """RAG from scratch: chunk, embed, retrieve, and answer — no framework."""
    print_banner(__version__)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to a text file to ingest."),
    index_path: Path = typer.Option(_settings.index_path, help="Where to store the index."),
    chunk_size: int = typer.Option(_settings.chunk_size, help="Words per chunk."),
    overlap: int = typer.Option(
        _settings.chunk_overlap, help="Words shared between consecutive chunks."
    ),
    embedding_model: str = typer.Option(
        _settings.embedding_model, help="Any litellm-supported embedding model."
    ),
) -> None:
    """Chunk, embed, and add a text file to the on-disk index."""
    try:
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
    except (PlainRAGError, OSError) as e:
        log.error(str(e))
        raise typer.Exit(code=1) from e
    typer.echo(f"Added {added} chunk(s) from {path} to {index_path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask."),
    index_path: Path = typer.Option(_settings.index_path, help="Index to search."),
    k: int = typer.Option(_settings.retrieval_k, help="Number of chunks to retrieve."),
    embedding_model: str = typer.Option(
        _settings.embedding_model, help="Any litellm-supported embedding model."
    ),
    chat_model: str = typer.Option(_settings.chat_model, help="Any litellm-supported chat model."),
) -> None:
    """Answer a question, grounded in whatever's been ingested."""
    try:
        index = CosineSimilarityIndex.load(index_path)
        result = asyncio.run(
            ask_question(
                index, question, k=k, embedding_model=embedding_model, chat_model=chat_model
            )
        )
    except PlainRAGError as e:
        log.error(str(e))
        raise typer.Exit(code=1) from e
    typer.echo(result.answer)
    if result.sources:
        typer.echo("\nSources:")
        for i, scored in enumerate(result.sources, start=1):
            typer.echo(f"  [{i}] {scored.chunk.source} (score={scored.score:.3f})")


if __name__ == "__main__":  # pragma: no cover
    app()
