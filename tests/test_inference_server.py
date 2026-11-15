from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import httpx
import pytest
from openai import OpenAI

from aion_u1.inference import Completion, InferenceInputError
from aion_u1.server import InferenceServer, MAX_REQUEST_BYTES

KEY = "synthetic-local-test-key"
PAYLOAD = {
    "model": "aion-u1",
    "messages": [{"role": "user", "content": "synthetic integration test"}],
    "max_tokens": 8,
    "temperature": 0,
}


class SyntheticEngine:
    model_id = "aion-u1"

    def __init__(self):
        self.requests = []
        self.error = None

    def complete(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return Completion("synthetic answer", 4, 2, "stop")


@contextmanager
def running_server(engine):
    with InferenceServer(("127.0.0.1", 0), engine, KEY, max_new_tokens=32) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=3)


@pytest.fixture
def local_client():
    engine = SyntheticEngine()
    with running_server(engine) as url:
        with httpx.Client(
            base_url=url, timeout=3, trust_env=False,
            headers={"Authorization": "Bearer " + KEY},
        ) as client:
            yield client, engine


def test_health_separates_runtime_from_quality_and_production(local_client) -> None:
    client, _ = local_client
    response = client.get("/health", headers={"Authorization": ""})
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready", "model": "aion-u1",
        "quality_verified": False, "production_ready": False,
    }


def test_model_discovery_requires_authentication(local_client) -> None:
    client, _ = local_client
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "aion-u1"
    assert response.headers["Cache-Control"] == "no-store"


def test_real_openai_sdk_can_complete_against_local_transport(local_client) -> None:
    http, engine = local_client
    with OpenAI(
        base_url=str(http.base_url) + "/v1", api_key=KEY,
        max_retries=0, http_client=httpx.Client(trust_env=False), timeout=3,
    ) as client:
        response = client.chat.completions.create(
            model="aion-u1",
            messages=[
                {"role": "system", "content": "Synthetic protocol test."},
                {"role": "user", "content": "Test"},
            ],
            max_tokens=8, temperature=0,
        )
    assert response.choices[0].message.content == "synthetic answer"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 6
    assert [message.role for message in engine.requests[0].messages] == ["system", "user"]


@pytest.mark.parametrize(("changes", "status"), [
    ({"model": "teacher-fallback"}, 404),
    ({"stream": True}, 400),
    ({"stream": "false"}, 400),
    ({"max_tokens": 33}, 400),
    ({"max_tokens": True}, 400),
    ({"max_tokens": 0}, 400),
    ({"temperature": float("nan")}, 400),
    ({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, 400),
    ({"messages": []}, 400),
    ({"tools": []}, 400),
])
def test_unsupported_requests_do_not_reach_model(local_client, changes, status) -> None:
    client, engine = local_client
    response = client.post(
        "/v1/chat/completions", content=json.dumps({**PAYLOAD, **changes}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == status
    assert engine.requests == []


def test_duplicate_json_keys_are_rejected(local_client) -> None:
    client, engine = local_client
    text = json.dumps(PAYLOAD)
    response = client.post(
        "/v1/chat/completions", content='{"model":"other",' + text[1:],
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert engine.requests == []


def test_oversized_request_is_rejected(local_client) -> None:
    client, engine = local_client
    response = client.post(
        "/v1/chat/completions", content=b" " * (MAX_REQUEST_BYTES + 1),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert engine.requests == []


@pytest.mark.parametrize(("error", "status"), [
    (InferenceInputError("private prompt"), 400),
    (RuntimeError("private path and bearer secret"), 503),
])
def test_failures_are_redacted_and_release_the_generation_slot(local_client, error, status) -> None:
    client, engine = local_client
    engine.error = error
    response = client.post("/v1/chat/completions", json=PAYLOAD)
    assert response.status_code == status
    assert "private" not in response.text
    assert "secret" not in response.text
    engine.error = None
    assert client.post("/v1/chat/completions", json=PAYLOAD).status_code == 200


def test_busy_model_keeps_health_available_and_rejects_extra_generation(local_client) -> None:
    client, engine = local_client
    entered = threading.Event()
    release = threading.Event()
    original = engine.complete
    def blocked(request):
        entered.set()
        assert release.wait(timeout=3)
        return original(request)
    engine.complete = blocked
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, "/v1/chat/completions", json=PAYLOAD)
        try:
            assert entered.wait(timeout=2)
            assert client.get("/health").status_code == 200
            busy = client.post("/v1/chat/completions", json=PAYLOAD)
            assert busy.status_code == 503
            assert busy.json()["error"]["code"] == "model_busy"
        finally:
            release.set()
        assert future.result(timeout=3).status_code == 200


def test_errors_and_prompts_do_not_enter_access_logs(local_client, capsys) -> None:
    client, _ = local_client
    assert client.post("/v1/chat/completions", json=PAYLOAD).status_code == 200
    assert capsys.readouterr().err == ""
