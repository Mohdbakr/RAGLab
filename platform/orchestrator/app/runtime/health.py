"""HTTP health probing used to tell "starting" apart from "healthy"."""

from __future__ import annotations

import httpx


class HealthChecker:
    """Polls a URL and reports whether the service behind it is up."""

    def __init__(self, client: httpx.AsyncClient | None = None, timeout: float = 2.0) -> None:
        """Create a health checker.

        Args:
            client: An `httpx.AsyncClient` to reuse. If omitted, a
                short-lived client is created per call.
            timeout: Seconds to wait for a response before treating the
                target as not yet healthy.
        """
        self._client = client
        self._timeout = timeout

    async def is_healthy(self, url: str) -> bool:
        """Check whether ``url`` responds without a server error.

        A container that's running but still starting up typically refuses
        connections or returns nothing at all, while one that's up but
        serving a 404 for this particular path is still "the process is
        alive" — only a connection failure or a 5xx counts as unhealthy.

        Args:
            url: The health-check URL to GET.

        Returns:
            True if the request completes with a status below 500; false
            on any connection error, timeout, or 5xx response.
        """
        client = self._client or httpx.AsyncClient()
        owns_client = self._client is None
        try:
            response = await client.get(url, timeout=self._timeout)
            return response.status_code < 500
        except httpx.HTTPError:
            return False
        finally:
            if owns_client:
                await client.aclose()
