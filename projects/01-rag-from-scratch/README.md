# RAG From Scratch

The first project in RAGLab, and the simplest one: retrieval-augmented
generation with no RAG framework and no vector-database service. Just the
actual mechanics — chunking, embeddings, cosine similarity, and prompt
assembly — written out plainly so there's nothing hidden behind an
abstraction.

It's fully standalone: no dependency on anything else in this repo.

## What it does

1. **`ingest`** — read a text file, split it into overlapping word-window
   chunks (`src/plainrag/chunking.py`), embed each chunk
   (`src/plainrag/embeddings.py`, via `litellm` — any embedding provider it
   supports works), and save the result to a small on-disk index
   (`src/plainrag/index.py`: a NumPy array of vectors plus their source
   text, no vector-database server involved).
2. **`ask`** — embed the question the same way, retrieve the top-k most
   similar chunks by cosine similarity, assemble them into a prompt with
   citations, and ask an LLM (`src/plainrag/llm.py`, also via `litellm`) to
   answer strictly from that context.

## Run it

```bash
uv sync
export OPENAI_API_KEY=...   # or point at any litellm-supported provider

uv run plainrag ingest path/to/some.txt
uv run plainrag ask "What does the document say about X?"
```

No Docker, no separate service — the index is just a file on disk
(`.plainrag_index.json` by default).

## Why build this "the hard way"

Every other project in RAGLab uses a framework or a real vector database
for good reasons (they handle edge cases this doesn't). This one exists to
make those mechanics legible before hiding them: what a chunk actually is,
what an embedding actually looks like, and what "retrieval" actually
computes, with nothing abstracted away.

## Design notes / tradeoffs

- **Chunking is word-count-based**, not semantic — simple to reason about
  and test, but it can split a sentence in half. A production chunker
  would respect sentence/paragraph boundaries.
- **The index is a flat NumPy array with brute-force cosine similarity** —
  O(n) per query. Fine for a few thousand chunks; a real vector database
  (see `02-document-qa-foundation` and later projects) uses an ANN index
  instead, which is exactly the tradeoff that makes it a separate concern.
- **No shared package.** This project doesn't depend on anything else in
  the repo — a deliberate choice so it stays trivially runnable and so
  later projects don't inherit assumptions from this one before they've
  earned a shared abstraction.
