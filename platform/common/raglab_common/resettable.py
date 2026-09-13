"""Contract stateful in-process components implement for demo resets.

Docker-backed components (a vector store container, a stateful backend) are
reset generically by the orchestrator via ``docker compose down -v && up
-d`` — no code in this module is involved there. ``Resettable`` exists for
the smaller number of in-process components (e.g. the from-scratch
project's in-memory index) that have no container to recycle.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Resettable(Protocol):
    """An in-process component that can wipe its own state on demand."""

    def reset(self) -> None:
        """Clear all persisted/in-memory state back to a fresh-start baseline.

        Raises:
            Exception: Implementations may raise if the underlying storage
                can't be cleared; callers should treat any exception as a
                failed reset rather than a partial one.
        """
        ...
