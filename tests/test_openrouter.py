from __future__ import annotations

import asyncio

import httpx
import pytest

from aion_u1.teachers.base import TeacherError, TeacherHTTPError, TeacherTransientError
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


async def _generate(handler, *, max_attempts=None):
    options = {} if max_attempts is None else {"max_attempts": max_attempts}
    client = OpenRouterClient(api_key="test", **options)
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

    answer = asyncio.run(_generate(handler, max_attempts=3))
    assert answer.content == "42"
    assert len(attempts) == 3
    assert sleeps == [1.5, 2.0]


@pytest.mark.parametrize("status", [401, 402, 403])
def test_openrouter_stops_immediately_on_auth_or_payment_error(status) -> None:
    attempts = []

    def handler(request):
        attempts.append(request)
        return _response(status, text="private provider response")

    with pytest.raises(TeacherHTTPError, match=f"Teacher HTTP {status}") as error:
        asyncio.run(_generate(handler, max_attempts=3))
    assert error.value.status_code == status
    assert "private provider response" not in str(error.value)
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
        asyncio.run(_generate(handler, max_attempts=3))
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

    answer = asyncio.run(_generate(handler, max_attempts=3))
    assert answer.content == "42"
    assert len(attempts) == 2


def test_retry_after_is_capped_and_http_date_ignored() -> None:
    assert _retry_after_seconds("120") == 60.0
    assert _retry_after_seconds("-5") == 0.0
    assert _retry_after_seconds("Wed, 04 Nov 2026 11:00:00 GMT") is None


@pytest.mark.parametrize("failure", ["503", "timeout"])
def test_default_attempts_make_only_one_request(failure) -> None:
    attempts = []

    def handler(request):
        attempts.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private timeout detail", request=request)
        return _response(503)

    with pytest.raises(TeacherTransientError) as error:
        asyncio.run(_generate(handler))
    assert len(attempts) == 1
    assert "private timeout detail" not in str(error.value)


def test_successful_but_invalid_response_is_never_retried() -> None:
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(200, content=b"invalid JSON")

    with pytest.raises(TeacherError, match="JSON"):
        asyncio.run(_generate(handler, max_attempts=3))
    assert len(attempts) == 1
