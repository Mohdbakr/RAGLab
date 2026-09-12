# RAGLab

A monorepo of independent, production-grade RAG reference implementations
— each small enough to build in a weekend, each demonstrating a different
retrieval-augmented-generation pattern, all launchable from one unified
frontend so they can be compared on real benchmarks, not just demoed.

This is a portfolio **and** a benchmarking lab: every project reports
latency/token/cost KPIs in a common format so implementations and vector
stores can be compared side by side, not just shown off individually.

## How it's organized

```
RAGLab/
├── catalog/            # backends.yaml + vectorstores.yaml — the source of truth
│                        # the launcher reads; add a project = add a folder + one entry
├── platform/
│   ├── common/          # raglab_common: shared benchmarking/logging/reset conventions
│   ├── orchestrator/     # FastAPI control plane: start/stop/status/reset via docker compose
│   └── frontend/         # unified Streamlit launcher — the one UI for every project
└── projects/
    ├── 01-rag-from-scratch/
    ├── 02-document-qa-foundation/
    └── ...
```

Each project under `projects/` is self-contained — its own
`pyproject.toml`/`uv.lock`, `Dockerfile`/`docker-compose`, tests, and
README — and runs standalone via its own `docker compose up`. The unified
launcher is a convenience layer on top, never a hard dependency.

## Run the unified launcher

```bash
# terminal 1
cd platform/orchestrator && uv sync && uv run fastapi dev app/main.py --port 8100

# terminal 2
cd platform/frontend && uv sync && uv run streamlit run app.py
```

Pick a backend from the sidebar dropdown, hit **Launch** — the orchestrator
spins up its containers and the sidebar shows starting → healthy. Pick a
compatible vector store the same way when a project supports swapping it.
**Reset** wipes a component's persisted data for a clean demo restart.

## The catalog, basic → advanced

| # | Project | Level | What it shows | Status | Tracking |
|---|---|---|---|---|---|
| 01 | RAG From Scratch | basic | Chunking, embeddings, cosine-similarity retrieval — no frameworks | planned | [#2](https://github.com/Mohdbakr/RAGLab/issues/2) |
| 02 | Document QA Foundation | basic | PDF/DOCX/TXT upload + Q&A over Milvus | in progress — owner reviewing before further changes | [#3](https://github.com/Mohdbakr/RAGLab/issues/3) |
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

Ordering rationale: 01 has zero infra and teaches the mechanics directly;
02 is the existing full-stack baseline with a real vector DB; 03 adds the
adapter/retry/caching patterns other projects build on conceptually;
04–05 add federation and a second modality; 06–08 move into agent
reasoning, streaming with live tools, and multi-stage pipelines; 09–10 are
the two graph-based approaches; 11 is memory-centric; 12 is the hardest
single system to get right; 13 assembles the pieces into one production
platform.

This table is a roadmap, not a locked commitment — it's revised as each
weekend's build lands and as real usage suggests better priorities.

## Standards every project follows

- **TDD**: tests written before implementation, red confirmed before green.
- **Logging**: `loguru` via `raglab_common.configure_logging`, no stdlib `logging`.
- **Benchmarking**: every retrieval/generation call logs a `BenchmarkEvent`
  (latency, tokens, cost, retrieved-k) via `raglab_common.BenchmarkLogger`
  to a per-project JSONL file the frontend's Benchmarks tab reads.
- **Reset**: generic per component — `docker compose down -v && up -d` —
  rather than bespoke per-store wipe logic.
- **Typing/docstrings/SOLID**: full type hints, `Protocol`/ABC at
  integration seams, Google-style docstrings, checked with `ruff` + `mypy --strict`.

`02-document-qa-foundation` is the one exception to all of the above for
now — untouched until its owner finishes reviewing it.

## Tracking

Each weekend milestone above is tracked as a GitHub issue (linked in the
table) with its own acceptance checklist, closed as it ships.
