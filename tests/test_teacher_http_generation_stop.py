from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from aion_u1.models import TeacherConfig
from aion_u1.teachers.openrouter import OpenRouterClient
from aion_u1.training import generate as generation
from aion_u1.training.checkpoint import CandidateCheckpoint
from aion_u1.training.tasks import TaskBank, TaskSpec, TaskVerifier


@pytest.mark.parametrize("status", [401, 402, 403])
def test_real_http_client_stops_waiting_generation_and_preserves_candidates(
    monkeypatch, tmp_path, status
):
    """Exercise the actual resolver, HTTP client, queue and durable checkpoint."""
    config = TeacherConfig(
        alias="teacher_http", display_name="Synthetic HTTP Teacher",
        provider="openrouter", candidate_model_ids=["synthetic/model"],
    )
    bank = TaskBank(tasks=[
        TaskSpec(
            id=f"task.http.{index}", capability="reasoning.logical",
            prompt=f"Synthetic HTTP {index}", teacher_aliases=[config.alias],
            verifier=TaskVerifier(type="exact_match", values=["OK"]),
        )
        for index in range(5)
    ])
    requests = []
    clients = []
    blocked = True

    def handler(request):
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "synthetic/model"}]})
        assert request.method == "POST"
        assert request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content)
        requests.append(payload)
        if blocked and len(requests) == 2:
            return httpx.Response(status, text="private synthetic provider detail")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "cost": 0.0001},
        })

    # Inject only an offline transport. Do not replace generate() or typed errors.
    real_async_client = httpx.AsyncClient

    def offline_client(*args, **kwargs):
        client = real_async_client(
            *args, **kwargs, transport=httpx.MockTransport(handler)
        )
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", offline_client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-test-only")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://offline.invalid/api/v1")
    assert generation.OpenRouterClient is OpenRouterClient

    output = tmp_path / "candidates.jsonl"

    def run(checkpoint):
        return asyncio.run(generation.generate_candidates(
            [config], bank, root=tmp_path, concurrency=1, checkpoint=checkpoint
        ))

    with CandidateCheckpoint(output) as checkpoint:
        rows, outcomes, resolutions = run(checkpoint)
    assert len(requests) == 2, "waiting requests must stop after the actual HTTP error"
    assert len(rows) == 1
    assert resolutions[0]["status"] == "verified"
    errors = [item for item in outcomes if not item.get("skipped")]
    assert len(errors) == 1 and errors[0]["http_status"] == status
    assert errors[0]["error_code"] == (
        "payment_required" if status == 402 else "teacher_request_failed"
    )
    assert "private synthetic provider detail" not in json.dumps(outcomes)
    assert sum(bool(item.get("skipped")) for item in outcomes) == 3
    before = output.read_bytes()
    saved = [json.loads(line) for line in before.decode("utf-8").splitlines()]
    assert len(saved) == 1 and saved[0]["response"] == "OK"
    assert saved[0]["cost_usd"] == 0.0001
    assert "cost" not in saved[0]["usage"]
    assert saved[0]["provenance"]["training_approved"] is False
    assert all(client.is_closed for client in clients)
    assert not checkpoint.lock_path.exists()

    blocked = False
    with CandidateCheckpoint(output, resume=True) as checkpoint:
        rows, outcomes, _ = run(checkpoint)
    assert len(rows) == 5 and not outcomes
    assert len(requests) == 6, "resume must request only the four missing candidates"
    assert output.read_bytes().startswith(before)
    assert all(client.is_closed for client in clients)
    assert not checkpoint.lock_path.exists()
