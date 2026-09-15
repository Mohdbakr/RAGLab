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
from plainrag.services.rag import retrieve as retrieve_chunks

app = typer.Typer(help="RAG from scratch: chunk, embed, retrieve, and answer — no framework.")

_settings = get_settings()
log = get_logger(level=_settings.log_level, log_file=_settings.log_file)


def _echo_step(message: str) -> None:
    """Print one pipeline-step progress line, prefixed for scannability.

    Plain ASCII only: Windows consoles default to a legacy codepage
    (e.g. cp1252) that can't encode characters like "·", and this runs
    there.
    """
    typer.echo(f"  -> {message}")


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
                on_step=_echo_step,
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
                index,
                question,
                k=k,
                embedding_model=embedding_model,
                chat_model=chat_model,
                on_step=_echo_step,
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


@app.command()
def demo(
    index_path: Path = typer.Option(_settings.index_path, help="Where to store the index."),
    chunk_size: int = typer.Option(_settings.chunk_size, help="Words per chunk."),
    overlap: int = typer.Option(
        _settings.chunk_overlap, help="Words shared between consecutive chunks."
    ),
    k: int = typer.Option(_settings.retrieval_k, help="Number of chunks to retrieve per search."),
) -> None:
    """Interactively ingest files and search the index — retrieval only, no LLM answers.

    Commands, one per line: ``ingest <path>``, ``search <query>``, ``quit``.
    """
    index = CosineSimilarityIndex.load(index_path)
    typer.echo(f"plainrag demo — {len(index)} chunk(s) loaded from {index_path}")
    typer.echo("Commands: ingest <path> | search <query> | quit\n")

    while True:
        try:
            line = input("plainrag> ").strip()
        except EOFError, KeyboardInterrupt:
            break
        if not line:
            continue

        command, _, rest = line.partition(" ")
        rest = rest.strip()

        if command in ("quit", "exit"):
            break

        elif command == "ingest":
            if not rest:
                typer.echo("Usage: ingest <path>")
                continue
            path = Path(rest)
            try:
                text = path.read_text()
                added = asyncio.run(
                    ingest_text(
                        index,
                        text,
                        source=str(path),
                        chunk_size=chunk_size,
                        overlap=overlap,
                        embedding_model=_settings.embedding_model,
                        on_step=_echo_step,
                    )
                )
                index.save(index_path)
                typer.echo(f"Added {added} chunk(s) from {path}")
            except (PlainRAGError, OSError) as e:
                typer.echo(f"Error: {e}", err=True)

        elif command == "search":
            if not rest:
                typer.echo("Usage: search <query>")
                continue
            try:
                results = asyncio.run(
                    retrieve_chunks(
                        index,
                        rest,
                        k=k,
                        embedding_model=_settings.embedding_model,
                        on_step=_echo_step,
                    )
                )
                for i, scored in enumerate(results, start=1):
                    preview = scored.chunk.text[:150].replace("\n", " ")
                    typer.echo(f"  [{i}] ({scored.score:.3f}) {scored.chunk.source}: {preview}")
            except PlainRAGError as e:
                typer.echo(f"Error: {e}", err=True)

        else:
            typer.echo(f"Unknown command: {command!r}. Try: ingest <path> | search <query> | quit")


if __name__ == "__main__":  # pragma: no cover
    app()
