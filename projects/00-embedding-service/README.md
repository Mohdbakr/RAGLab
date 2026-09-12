# Embedding Service

A standalone, model-swappable embedding microservice. It exists to
demonstrate — and let you benchmark — a real production pattern: running
embedding generation as its own independently-scaled service instead of
embedding in-process inside every RAG backend.

It's a thin HTTP wrapper around `raglab_common.build_embedding_client`, so
it supports exactly the same model strings every other RAGLab project
does: `"local/<model-name>"` (sentence-transformers, no API key needed),
or any `litellm` model string (`"openai/text-embedding-3-small"`,
`"cohere/embed-english-v3.0"`, ...). Built clients are cached per model so
a local model is only loaded into memory once, not on every request.

This project is fully standalone — it doesn't require the unified
launcher or any other project to run or be useful. Any project can *opt in*
to using it by pointing its own `EmbeddingClient` config at
`"http://localhost:9100"` (via `raglab_common.build_embedding_client`,
which routes `http://`/`https://` specs to `HTTPEmbeddingClient`) instead
of embedding in-process — nothing requires it.

## Run it

```bash
docker compose up --build
```

Or without Docker:

```bash
uv sync
uv run fastapi run app/main.py --port 9100
```

## API

**`POST /embed`**

```json
{"texts": ["hello world"], "model": "local/all-mpnet-base-v2"}
```

`model` is optional — omit it to use the service's configured default
(`DEFAULT_EMBEDDING_MODEL`, see `.env.example`).

```json
{"model": "local/all-mpnet-base-v2", "embeddings": [[0.0123, -0.0456, ...]]}
```

**`GET /healthz`** — liveness check.

Every call logs a `BenchmarkEvent` (operation `"embed"`) via
`raglab_common.BenchmarkLogger`, so this service's latency is directly
comparable to in-process embedding in the shared Benchmarks view.
