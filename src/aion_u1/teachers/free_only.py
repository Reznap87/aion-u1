"""Restricted OpenRouter client for explicitly selected free text models."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from ..models import TeacherResponse
from .base import TeacherError

BASE_URL = "https://openrouter.ai/api/v1"
MODEL_PATTERN = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+:free"


def _zero(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    try:
        number = Decimal(str(value))
        return number.is_finite() and number == 0
    except InvalidOperation:
        return False


def free_model_ids(catalog: Any) -> set[str]:
    if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
        raise TeacherError("Invalid free model catalog")
    result: set[str] = set()
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in catalog["data"]:
        if not isinstance(entry, dict):
            continue
        model = entry.get("id")
        if not isinstance(model, str) or not re.fullmatch(MODEL_PATTERN, model):
            continue
        if model in seen:
            duplicates.add(model)
        seen.add(model)
        pricing = entry.get("pricing")
        if (isinstance(pricing, dict)
                and {"prompt", "completion"} <= pricing.keys()
                and all(_zero(value) for value in pricing.values())):
            result.add(model)
    return result - duplicates


class FreeOnlyClient:
    def __init__(self, api_key: str):
        if not api_key.strip():
            raise ValueError("Free-only API key required")
        self._client = httpx.AsyncClient(
            base_url=BASE_URL + "/", follow_redirects=False, trust_env=False,
            timeout=httpx.Timeout(120, connect=15),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        response = await self._client.get("models")
        response.raise_for_status()
        return sorted(free_model_ids(response.json()))

    async def generate(self, model_id: str, messages: list[dict[str, object]], *,
                       temperature: float = 0.0, max_tokens: int | None = None) -> TeacherResponse:
        if not re.fullmatch(MODEL_PATTERN, model_id):
            raise TeacherError("Explicit free variant required")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
            raise TeacherError("Explicit bounded token budget required")
        if not messages or any(set(message) != {"role", "content"}
                               or message["role"] != "user"
                               or not isinstance(message["content"], str)
                               for message in messages):
            raise TeacherError("Free-only path supports plain user text only")
        # Recheck catalog pricing immediately before each inference request.
        if model_id not in await self.list_models():
            raise TeacherError("Free variant unavailable or pricing not verified")
        payload = {
            "model": model_id, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens,
            "provider": {"allow_fallbacks": False, "require_parameters": True,
                         "max_price": {"prompt": 0, "completion": 0}},
        }
        # Deliberately no retries, model fallback, arbitrary endpoint or budget shrink.
        response = await self._client.post("chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("model") != model_id:
            raise TeacherError("Unexpected response model")
        usage = data.get("usage")
        if not isinstance(usage, dict) or not _zero(usage.get("cost")):
            raise TeacherError("Missing or nonzero reported inference cost")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise TeacherError("Missing completion")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TeacherError("Invalid completion content")
        return TeacherResponse(
            model_id=model_id, content=message["content"],
            finish_reason=choice.get("finish_reason"),
            usage={key: value for key, value in usage.items()
                   if key != "cost" and type(value) is int},
            max_tokens=max_tokens,
        )
