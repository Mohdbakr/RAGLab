# RAGLab Orchestrator

FastAPI control plane that lets the unified frontend start, stop, check
the health of, and reset any RAGLab backend or vector store on demand, by
driving `docker compose` for whatever project the caller picks. See the
"Orchestrator & frontend" section of the root README for the full design.

## Run locally

```bash
uv sync
uv run fastapi dev app/main.py
```

Requires a reachable Docker daemon to actually start/stop containers; the
catalog/status/lifecycle logic is unit-tested here against a mocked
`docker compose` process, independent of a live daemon.
