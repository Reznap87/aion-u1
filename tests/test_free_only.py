import asyncio

import httpx
import pytest

from aion_u1.teachers.base import TeacherError
from aion_u1.teachers.free_only import FreeOnlyClient, free_model_ids

MODEL = "example/model:free"
FREE = {"prompt": "0", "completion": "0", "request": "0"}


def test_catalog_requires_explicit_zero_prices():
    assert free_model_ids({"data": [
        {"id": MODEL, "pricing": FREE},
        {"id": "example/paid", "pricing": FREE},
        {"id": "example/missing:free", "pricing": {"prompt": "0"}},
        {"id": "example/fee:free", "pricing": {"prompt": "0", "completion": "0", "request": "0.01"}},
        {"id": "example/invalid:free", "pricing": {"prompt": "NaN", "completion": "0"}},
    ]}) == {MODEL}


def test_duplicate_model_catalog_entry_rejected():
    assert not free_model_ids({"data": [{"id": MODEL, "pricing": FREE}] * 2})


ZERO_KEY = {"limit": 0, "limit_remaining": 0, "is_management_key": False,
            "is_provisioning_key": False, "include_byok_in_limit": True}


def run(handler, model=MODEL, key_info=ZERO_KEY):
    def dispatch(request):
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": key_info})
        return handler(request)
    async def invoke():
        client = FreeOnlyClient("test-key")
        await client.close()
        client._client = httpx.AsyncClient(base_url="https://openrouter.ai/api/v1/",
                                           transport=httpx.MockTransport(dispatch))
        try:
            return await client.generate(model, [{"role": "user", "content": "test"}], max_tokens=32)
        finally:
            await client.close()
    return asyncio.run(invoke())


def test_request_has_zero_price_and_no_fallback():
    import json

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL, "pricing": FREE}]})
        payload = json.loads(request.content)
        assert payload["provider"]["max_price"] == {"prompt": 0, "completion": 0}
        assert payload["provider"]["allow_fallbacks"] is False
        assert "models" not in payload
        assert payload["usage"] == {"include": True}
        return httpx.Response(200, json={"model": MODEL, "usage": {"cost": 0},
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    assert run(handler).content == "ok"


def test_paid_model_never_contacts_provider():
    def handler(request):
        pytest.fail("Unexpected request")
    with pytest.raises(TeacherError):
        run(handler, "example/paid")


def test_unverified_price_never_posts():
    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"data": []})
    with pytest.raises(TeacherError):
        run(handler)


@pytest.mark.parametrize("cost", [None, 0.01])
def test_reject_unverified_reported_cost(cost):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL, "pricing": FREE}]})
        return httpx.Response(200, json={"model": MODEL, "usage": {"cost": cost},
            "choices": [{"message": {"content": "ok"}}]})
    with pytest.raises(TeacherError):
        run(handler)


@pytest.mark.parametrize("key_info", [
    {}, dict(ZERO_KEY, limit=None), dict(ZERO_KEY, limit=1),
    dict(ZERO_KEY, limit_remaining=1), dict(ZERO_KEY, include_byok_in_limit=False),
    dict(ZERO_KEY, is_management_key=True), dict(ZERO_KEY, is_provisioning_key=True),
])
def test_unrestricted_or_ambiguous_key_is_rejected(key_info):
    def handler(request):
        pytest.fail("Catalog or inference request must not be sent")
    with pytest.raises(TeacherError, match="zero limit"):
        run(handler, key_info=key_info)


def test_reported_byok_rejected():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL, "pricing": FREE}]})
        return httpx.Response(200, json={"model": MODEL, "usage": {"cost": 0, "is_byok": True},
            "choices": [{"message": {"content": "ok"}}]})
    with pytest.raises(TeacherError, match="BYOK"):
        run(handler)
