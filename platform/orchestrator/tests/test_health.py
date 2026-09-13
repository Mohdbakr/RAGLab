"""Tests for app.runtime.health, written before the implementation."""

from __future__ import annotations

import httpx
import pytest

from app.runtime.health import HealthChecker


def client_returning(status_code: int) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def client_raising() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestHealthChecker:
    @pytest.mark.asyncio
    async def test_true_on_2xx(self) -> None:
        checker = HealthChecker(client=client_returning(200))
        assert await checker.is_healthy("http://backend/healthz") is True

    @pytest.mark.asyncio
    async def test_true_on_4xx_since_the_server_is_at_least_up(self) -> None:
        checker = HealthChecker(client=client_returning(404))
        assert await checker.is_healthy("http://backend/healthz") is True

    @pytest.mark.asyncio
    async def test_false_on_5xx(self) -> None:
        checker = HealthChecker(client=client_returning(503))
        assert await checker.is_healthy("http://backend/healthz") is False

    @pytest.mark.asyncio
    async def test_false_when_connection_fails(self) -> None:
        checker = HealthChecker(client=client_raising())
        assert await checker.is_healthy("http://backend/healthz") is False
