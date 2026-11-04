from __future__ import annotations

import asyncio

import httpx
import pytest

from aion_u1.teachers.base import TeacherError, TeacherTransientError
from aion_u1.teachers.openrouter import (
    OpenRouterClient, _retry_after_seconds,
)


def _response(status: int = 200, **kwargs) -> httpx.Response:
    if status == 200:
        return httpx.Response(
            status, json={"choices": [{
                "message": {"content": "42"},
                "finish_reason": "stop",
            }]}
        )
    return httpx.Response(status, **kwargs)


async def _generate(handler, *, max_attempts=3):
    client = OpenRouterClient(api_key="test", max_attempts=max_attempts)
    await client.close()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    try:
        return await client.generate(
            "provider/model", [{"role": "user", "content": "42?"}]
        )
    finally:
        await client.close()


def test_openrouter_retries_429_503_then_succeeds(monkeypatch) -> None:
    import aion_u1.teachers.openrouter as module

    sleeps: list[float] = []

    async def fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return _response(429, headers={"Retry-After": "1.5"})
        if len(attempts) == 2:
            return _response(503)
        return _response()

    answer = asyncio.run(_generate(handler))
    assert answer.content == "42"
    assert len(attempts) == 3
    assert sleeps == [1.5, 2.0]


def test_openrouter_stops_immediately_on_401() -> None:
    attempts = []

    def handler(request):
        attempts.append(request)
        return _response(401)

    with pytest.raises(TeacherError, match="Teacher HTTP 401"):
        asyncio.run(_generate(handler))
    assert len(attempts) == 1


def test_openrouter_exhausted_retry_is_transient(monkeypatch) -> None:
    import aion_u1.teachers.openrouter as module

    async def fake_sleep(_seconds: float):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    attempts = []

    def handler(request):
        attempts.append(request)
        return _response(503)

    with pytest.raises(TeacherTransientError, match="Teacher HTTP 503"):
        asyncio.run(_generate(handler))
    assert len(attempts) == 3


def test_openrouter_retries_transport_timeout(monkeypatch) -> None:
    import aion_u1.teachers.openrouter as module

    async def fake_sleep(_seconds: float):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return _response()

    answer = asyncio.run(_generate(handler))
    assert answer.content == "42"
    assert len(attempts) == 2


def test_retry_after_is_capped_and_http_date_ignored() -> None:
    assert _retry_after_seconds("120") == 60.0
    assert _retry_after_seconds("-5") == 0.0
    assert _retry_after_seconds("Wed, 04 Nov 2026 11:00:00 GMT") is None
