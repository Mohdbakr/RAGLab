# RAGLab

A benchmarking monorepo for comparing retrieval-augmented generation (RAG)
implementations. Each project under `projects/` is a fully standalone,
self-contained approach — no shared package, no shared assumptions between
them — so they stay directly comparable and independently runnable.

## Projects

- **[`projects/01-rag-from-scratch`](projects/01-rag-from-scratch/README.md)**
  — RAG with no framework and no vector-database service: chunking,
  embeddings, cosine similarity, and prompt assembly written out plainly,
  nothing hidden behind an abstraction. A CLI (`ingest` / `ask` / `demo`)
  ingests a text file and answers questions grounded in it.

More projects will land alongside it over time, each exploring a different
set of tradeoffs (e.g. a framework-backed implementation, an ANN-backed
vector store) for comparison against the others.

## Structure

```
RAGLab/
└── projects/
    └── 01-rag-from-scratch/   # no framework, no vector-DB service
```

## Getting started

Every project is standalone — its own dependencies, its own `.env`, its own
README. Pick one and follow its instructions:

```bash
cd projects/01-rag-from-scratch
cat README.md
```
