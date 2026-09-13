# raglab-common

Tiny shared library used by every RAGLab project so implementations stay
independent while still being comparable:

- `raglab_common.benchmarking` — a `BenchmarkEvent` schema and
  `BenchmarkLogger` so every project reports latency/token/cost KPIs to a
  common JSONL format the platform's Benchmarks dashboard can read.
- `raglab_common.logging_setup` — one-line loguru configuration (console +
  rotating file sink) so no project reaches for stdlib `logging`.
- `raglab_common.resettable` — the `Resettable` protocol stateful
  components implement so the orchestrator's generic reset flow has a
  consistent contract to check for in-process (non-Docker) components.

This package intentionally stays small. It is not a shared application
framework — each project remains standalone-runnable without it beyond
these three concerns.
