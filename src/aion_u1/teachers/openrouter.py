from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from ..models import TeacherResponse
from .base import TeacherError, TeacherTransientError


RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _retry_after_seconds(value: str | None) -> float | None:
    """Interpretiert nur Sekunden, begrenzt auf 60; HTTP-Dates werden ignoriert."""
    if not value:
        return None
    try:
        return max(0.0, min(float(value), 60.0))
    except ValueError:
        return None


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 180.0,
        max_attempts: int = 3,
    ):
        self._api_key = api_key
        self._max_attempts = max(1, max_attempts)
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "AION-U1 Teacher Audit",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.get(
                f"{self._base_url}/models"
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise TeacherError(
                f"Teacher-Katalog konnte nicht geladen werden: {exc}"
            ) from exc

        data = payload.get("data", [])

        if not isinstance(data, list):
            raise TeacherError("Ungültiger Teacher-Katalog")

        return sorted(
            {
                str(item["id"])
                for item in data
                if isinstance(item, dict) and item.get("id")
            }
        )

    async def generate(
        self,
        model_id: str,
        messages: list[dict[str, object]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> TeacherResponse:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        started = time.monotonic()
        last_error: TeacherTransientError | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as exc:
                    raise TeacherError("Ungültige Teacher-JSON-Antwort") from exc
                if not isinstance(data, dict):
                    raise TeacherError("Teacher-Antwort ist kein JSON-Objekt")
                break
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code not in RETRYABLE_STATUSES:
                    raise TeacherError(f"Teacher HTTP {code}") from exc
                last_error = TeacherTransientError(
                    f"Teacher HTTP {code} nach Versuch "
                    f"{attempt}/{self._max_attempts}"
                )
                if attempt == self._max_attempts:
                    raise last_error from exc
                retry_after = _retry_after_seconds(
                    exc.response.headers.get("retry-after")
                )
                await asyncio.sleep(
                    retry_after
                    if retry_after is not None
                    else min(2 ** (attempt - 1), 8.0)
                )
            except httpx.TransportError as exc:
                last_error = TeacherTransientError(
                    "Teacher-Transportfehler nach Versuch "
                    f"{attempt}/{self._max_attempts}"
                )
                if attempt == self._max_attempts:
                    raise last_error from exc
                await asyncio.sleep(min(2 ** (attempt - 1), 8.0))
        else:
            raise last_error or TeacherError("Teacher-Antwort unvollständig")

        choices = data.get("choices", [])

        if not isinstance(choices, list) or not choices:
            raise TeacherError("Teacher-Antwort enthält keine choices")

        choice = choices[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content", "")

        if isinstance(content, list):
            parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)

            content = "\n".join(parts)
        elif not isinstance(content, str):
            content = str(content or "")

        raw_usage = data.get("usage") or {}
        usage: dict[str, int] = {}

        if isinstance(raw_usage, dict):
            for key, value in raw_usage.items():
                if isinstance(value, (int, float)):
                    usage[str(key)] = int(value)

        return TeacherResponse(
            model_id=model_id,
            content=content,
            finish_reason=(
                str(choice.get("finish_reason"))
                if isinstance(choice, dict)
                and choice.get("finish_reason") is not None
                else None
            ),
            usage=usage,
            latency_ms=int((time.monotonic() - started) * 1000),
        )
