# RAGLab Frontend

The one interactive surface for every RAGLab project. Pick a backend from
the dropdown, launch it (the [orchestrator](../orchestrator) spins up its
Docker containers on demand and reports back once it's healthy), pick a
compatible vector store the same way, then try it — chat/upload panels
land per-project as each weekend milestone ships.

## Run locally

```bash
uv sync
uv run streamlit run app.py
```

Requires the orchestrator running (see `../orchestrator/README.md`) and,
by default, reachable at `http://localhost:8100` — override with
`ORCHESTRATOR_BASE_URL` in a `.env` file (see `.env.example`).

All HTTP calls and launch/stop/reset sequencing live in
`raglab_frontend/`, unit-tested independently of Streamlit; `app.py` is
UI wiring only.
