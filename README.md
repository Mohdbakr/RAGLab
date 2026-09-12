# RAGLab

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](./LICENSE)

I built RAGLab to test, compare, and benchmark different
retrieval-augmented-generation (RAG) implementations against each other —
not just to demo that one particular approach works. It's a growing
collection of independent RAG projects, each small enough for me to build
in a weekend, each showing a different retrieval pattern, all launchable
from one interface so I can actually compare them: latency, cost, and
retrieval quality, side by side, logged the same way everywhere.

If you're new here: pick a project from the dropdown in the unified
launcher, click Launch, and try it. Everything below walks you through
that step by step.

## Quickstart (Docker)

This is the standard way I run everything — you only need Docker
installed, nothing else.

**Prerequisites**
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose (bundled
  with Docker Desktop, and with recent Docker Engine on Linux)
- An OpenAI-compatible API key for projects that call an LLM (not needed
  just to launch the interface itself — you'll be asked for it per project)

**1. Clone the repo and start the launcher**

```bash
git clone https://github.com/Mohdbakr/RAGLab.git
cd RAGLab
docker compose up --build
```

This starts two things:
- the **orchestrator** — a small service that starts/stops the actual RAG
  project containers for you, on demand
- the **frontend** — the web page you'll actually use

Give it a minute the first time (it's building images). You'll know it's
ready when you see a line like `You can now view your Streamlit app`.

**2. Open the launcher**

Go to [http://localhost:8501](http://localhost:8501) in your browser.

**3. Pick a project and launch it**

- Choose a backend from the **Backend** dropdown in the sidebar (they're
  ordered from simplest to most advanced — start with the first one if
  you're not sure).
- If it shows a **Vector store** dropdown too, pick one — some projects let
  you swap the vector database and compare them.
- Click **▶ Launch**. The orchestrator spins up that project's containers
  in the background; the sidebar will show 🟡 *starting* and then
  🟢 *healthy* once it's ready. This can take a little while the first
  time a project's images need building.

**4. Try it**

Once the sidebar shows the backend as 🟢 healthy, switch to the **Try it**
tab and start chatting or uploading a document, depending on what that
project does.

**5. Reset or stop when you're done**

- **⟲ Reset** wipes that project's data (uploaded files, vector index,
  memory — whatever it persists) so you can start over with a clean slate.
- **■ Stop** shuts its containers down without deleting anything, so you
  can pick it back up later.

### Platform note

The orchestrator and frontend use `network_mode: host` so they can reach
the project containers they start at `localhost` with no extra network
setup. This works out of the box on Linux. On Docker Desktop (Mac/Windows),
either turn on "host networking" in Docker Desktop's settings, or run the
orchestrator and frontend directly instead of in Docker — see
[Alternative: run without Docker](#alternative-run-without-docker) below;
Docker is still the standard way to run the actual RAG projects either way.

## Alternative: run without Docker

Neither the orchestrator nor the frontend strictly needs to be
containerized — they just need Docker itself (for the orchestrator, so it
can drive `docker compose` for other projects) and Python. If you'd rather
run them directly:

```bash
# terminal 1
cd platform/orchestrator && uv sync && uv run fastapi dev app/main.py --port 8100

# terminal 2
cd platform/frontend && uv sync && uv run streamlit run app.py
```

This needs [uv](https://docs.astral.sh/uv/) installed, and is mainly useful
if you're developing on RAGLab itself or hitting the Docker Desktop
networking caveat above.

## How it's organized

```
RAGLab/
├── catalog/            # backends.yaml + vectorstores.yaml — what the launcher can start
├── platform/
│   ├── common/          # raglab_common: shared benchmarking/logging/reset conventions
│   ├── orchestrator/     # drives docker compose to start/stop/reset things
│   └── frontend/         # the unified launcher UI
└── projects/
    ├── 00-embedding-service/
    ├── 01-rag-from-scratch/
    ├── 02-document-qa-foundation/
    └── ...
```

Every project under `projects/` is also self-contained and can be run on
its own with its own `docker compose up`, without the launcher — the
launcher is a convenience on top, never a requirement.

## The catalog, basic → advanced

| # | Project | Level | What it shows | Status | Tracking |
|---|---|---|---|---|---|
| 01 | RAG From Scratch | basic | Chunking, embeddings, cosine-similarity retrieval — no frameworks | planned | [#2](https://github.com/Mohdbakr/RAGLab/issues/2) |
| 02 | Document QA Foundation | basic | PDF/DOCX/TXT upload + Q&A over Milvus | in progress — I'm reviewing this one myself before further changes | [#3](https://github.com/Mohdbakr/RAGLab/issues/3) |
| 03 | Production RAG Reference | intermediate | Swappable vector-store adapter (Chroma/Qdrant/Weaviate/pgvector), retries, caching | planned | [#4](https://github.com/Mohdbakr/RAGLab/issues/4) |
| 04 | Multi-Document RAG | intermediate | Federated retrieval across named sources, reciprocal rank fusion | planned | [#5](https://github.com/Mohdbakr/RAGLab/issues/5) |
| 05 | Multimodal RAG | intermediate | CLIP text+image retrieval | planned | [#6](https://github.com/Mohdbakr/RAGLab/issues/6) |
| 06 | Agentic RAG | advanced | Hand-rolled ReAct tool-calling loop (framework-free) | planned | [#7](https://github.com/Mohdbakr/RAGLab/issues/7) |
| 07 | Real-Time Assistant | advanced | LangChain LCEL + a live-data tool, streamed responses | planned | [#8](https://github.com/Mohdbakr/RAGLab/issues/8) |
| 08 | AI Research Agent | advanced | arXiv ingestion, map-reduce summarization, structured output | planned | [#9](https://github.com/Mohdbakr/RAGLab/issues/9) |
| 09 | GraphRAG (Classic) | advanced | LLM-built knowledge graph, community summaries, hybrid retrieval | planned | [#10](https://github.com/Mohdbakr/RAGLab/issues/10) |
| 10 | GraphRAG (Graphiti) | advanced | Temporal/episodic knowledge graph via `graphiti-core` + Neo4j | planned | [#11](https://github.com/Mohdbakr/RAGLab/issues/11) |
| 11 | Mem0 Memory Agent | advanced | Persistent long-term memory across sessions via `mem0ai` | planned | [#12](https://github.com/Mohdbakr/RAGLab/issues/12) |
| 12 | LangGraph Production Agent | advanced | Tool-using agent with checkpointed, persisted multi-turn memory | planned | [#13](https://github.com/Mohdbakr/RAGLab/issues/13) |
| 13 | Capstone: Full Production Platform | advanced | Assembles 03 + 10 + 11 + 12 into one system — integration, not new engineering | planned | [#14](https://github.com/Mohdbakr/RAGLab/issues/14) |

The launcher/orchestrator platform itself is tracked in [#1](https://github.com/Mohdbakr/RAGLab/issues/1).

I ordered these basic → advanced on purpose: 01 has zero infra and teaches
the mechanics directly; 02 is the existing full-stack baseline with a real
vector DB; 03 adds the adapter/retry/caching patterns the later projects
build on conceptually; 04–05 add federation and a second modality; 06–08
move into agent reasoning, streaming with live tools, and multi-stage
pipelines; 09–10 are the two graph-based approaches; 11 is memory-centric;
12 is the hardest single system to get right; 13 assembles the pieces into
one production platform.

This table is a living roadmap, not a locked commitment — I revise it as
each build lands and as I learn what's actually worth prioritizing next.

## Supporting services

Not every project is a "RAG pattern to compare" — some are shared
infrastructure other projects can *optionally* call instead of doing
something in-process. These are numbered `00` and don't appear in the
basic→advanced table above, and nothing depends on them being up.

| Project | What it shows | Status | Tracking |
|---|---|---|---|
| [00 — Embedding Service](projects/00-embedding-service) | Standalone, model-swappable embedding microservice — the production pattern of running embeddings as their own scaled service, called over HTTP via `raglab_common.HTTPEmbeddingClient` | shipped | [#15](https://github.com/Mohdbakr/RAGLab/issues/15) |

## Standards every project follows

- **TDD**: tests written before implementation, red confirmed before green.
- **Logging**: `loguru` via `raglab_common.configure_logging`, no stdlib `logging`.
- **Benchmarking**: every retrieval/generation call logs a `BenchmarkEvent`
  (latency, tokens, cost, retrieved-k) via `raglab_common.BenchmarkLogger`
  to a per-project JSONL file the frontend's Benchmarks tab reads.
- **Model-agnostic**: no project calls a vendor SDK directly. LLM calls go
  through `raglab_common.LLMClient` (a `litellm`-backed default), so
  swapping OpenAI for Anthropic, a local Ollama model, or anything else
  litellm supports is a config change (`"openai/gpt-4o-mini"` →
  `"ollama/llama3"`), not a code change. Embeddings go through the
  separate `raglab_common.EmbeddingClient` the same way — including a
  local, offline `sentence-transformers` option (the `local-embeddings`
  extra) and an `HTTPEmbeddingClient` that calls the standalone
  [Embedding Service](projects/00-embedding-service), alongside API-based
  ones — so embedding models (and where they run) can be swapped and
  compared independently of the LLM.
- **Reset**: generic per component — `docker compose down -v && up -d` —
  rather than bespoke per-store wipe logic.
- **Typing/docstrings/SOLID**: full type hints, `Protocol`/ABC at
  integration seams, Google-style docstrings, checked with `ruff` + `mypy --strict`.
- **CLI**: `typer` where a project exposes one (e.g. 01's `python -m ragscratch ask "..."`) —
  it derives the CLI directly from type-hinted function signatures rather
  than a separate `argparse`/`click` parser definition, so the interface
  and its types can't drift apart.

`02-document-qa-foundation` is the one exception to all of the above for
now — untouched until I've finished reviewing it myself.

## Tracking

Each weekend milestone above is tracked as a GitHub issue (linked in the
table) with its own acceptance checklist, closed as it ships.

## Versioning & releases

Tags follow CalVer: `vYYYY.0M.N` — year, zero-padded month, and a release
counter that resets to 0 at the start of each month (e.g. `v2026.09.0`,
then `v2026.09.1` for the next thing that ships the same month,
`v2026.10.0` once October starts). I cut a tag and a
[GitHub Release](https://github.com/Mohdbakr/RAGLab/releases) after each
milestone lands — a new project, or a meaningful platform change — rather
than on a fixed schedule.

I'm using CalVer instead of SemVer on purpose: RAGLab isn't one library
with an API compatibility contract to signal — it's a growing set of
independent projects plus a shared platform, and what's actually useful to
know is *when* something shipped, not whether it counts as a "breaking"
change against some baseline. The month is zero-padded (`2026.09`, not
`2026.9`) so tags still sort correctly as plain strings — bare CalVer
without padding (`2026.9.0`) doesn't sort right once you hit `2026.10.0`.

## License

CC BY-NC 4.0 — free to use for personal use and learning, attribution
required, no commercial use without my permission. See [LICENSE](./LICENSE)
for the full terms and how to reach me about commercial use.
