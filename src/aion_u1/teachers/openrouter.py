from __future__ import annotations

from typing import Any

import httpx

from ..models import TeacherResponse
from .base import TeacherError


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 180.0,
    ):
        self._api_key = api_key
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

        import time

        started = time.monotonic()

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise TeacherError(
                f"Teacher HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise TeacherError(
                f"Teacher-Anfrage fehlgeschlagen: {exc}"
            ) from exc

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
