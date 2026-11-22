from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from aion_u1.models import TeacherConfig, TeacherResponse
from aion_u1.teachers.base import TeacherHTTPError
from aion_u1.teachers.openrouter import OpenRouterClient
from aion_u1.training import generate as gen
from aion_u1.training.checkpoint import CandidateCheckpoint
from aion_u1.training.tasks import TaskBank, TaskSpec, TaskVerifier


@pytest.fixture
def setup_generation(monkeypatch, tmp_path):
    config = TeacherConfig(
        alias="teacher_a", display_name="Synthetic Teacher", provider="openrouter",
        candidate_model_ids=["synthetic/model"],
    )
    bank = TaskBank(tasks=[
        TaskSpec(id=f"task.{i}", capability="reasoning.logical", prompt=f"Synthetic {i}",
                 teacher_aliases=["teacher_a"],
                 verifier=TaskVerifier(type="exact_match", values=["OK"]))
        for i in range(5)
    ])
    class Client:
        def __init__(self, **kwargs):
            self.calls = []
            self.closed = False
            self.failure_at = None
            self.before_call = None
        async def list_models(self):
            return ["synthetic/model"]
        async def generate(self, model_id, messages, **kwargs):
            self.calls.append(messages[0]["content"])
            if self.before_call:
                self.before_call()
            await asyncio.sleep(0)
            if self.failure_at == len(self.calls):
                raise TeacherHTTPError(402)
            return TeacherResponse(model_id=model_id, content="OK", cost_usd=0.0001)
        async def close(self):
            self.closed = True
    client = Client()
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-test-only")
    monkeypatch.setattr(gen, "OpenRouterClient", lambda **kwargs: client)
    monkeypatch.setattr(gen, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(gen, "load_dotenv", lambda: None)
    monkeypatch.setattr(gen, "load_task_bank", lambda path: bank)
    monkeypatch.setattr(gen, "load_teacher_configs", lambda root, path=None: [config])
    return client, config, bank, tmp_path


def run(config, bank, root, checkpoint, concurrency=1):
    return asyncio.run(gen.generate_candidates(
        [config], bank, root=root, concurrency=concurrency, checkpoint=checkpoint
    ))


def test_402_stops_waiting_jobs_and_preserves_successes(setup_generation):
    client, config, bank, root = setup_generation
    client.failure_at = 3
    output = root / "candidates.jsonl"
    with CandidateCheckpoint(output) as checkpoint:
        rows, errors, _, _ = run(config, bank, root, checkpoint)
    assert len(client.calls) == 3
    assert len(rows) == 2
    assert sum(e.get("http_status") == 402 for e in errors) == 1
    assert sum(bool(e.get("skipped")) for e in errors) == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert client.closed
    assert not Path(str(output) + ".lock").exists()


def test_resume_only_generates_missing_jobs(setup_generation):
    client, config, bank, root = setup_generation
    output = root / "candidates.jsonl"
    client.failure_at = 3
    with CandidateCheckpoint(output) as checkpoint:
        run(config, bank, root, checkpoint)
    original = output.read_bytes()
    client.calls.clear()
    client.failure_at = None
    with CandidateCheckpoint(output, resume=True) as checkpoint:
        rows, errors, _, _ = run(config, bank, root, checkpoint)
    assert len(rows) == 5 and not errors
    assert len(client.calls) == 3
    assert output.read_bytes().startswith(original)
    assert all(row["provenance"]["training_approved"] is False for row in rows)
    client.calls.clear()
    with CandidateCheckpoint(output, resume=True) as checkpoint:
        rows, errors, _, _ = run(config, bank, root, checkpoint)
    assert len(rows) == 5 and not client.calls and not errors


def test_completed_response_is_checkpointed_while_another_call_runs(setup_generation):
    client, config, bank, root = setup_generation
    output = root / "candidates.jsonl"
    original_generate = client.generate
    async def slow_generate(*args, **kwargs):
        if client.calls:
            await asyncio.sleep(0.01)
            assert output.read_text(encoding="utf-8").strip()
        return await original_generate(*args, **kwargs)
    client.generate = slow_generate
    with CandidateCheckpoint(output) as checkpoint:
        rows, errors, _, _ = run(config, bank, root, checkpoint)
    assert len(rows) == 5 and not errors


@pytest.mark.parametrize("mutation", ["prompt", "temperature", "model", "duplicate", "cost"])
def test_mismatched_resume_never_calls_teacher(setup_generation, mutation):
    client, config, bank, root = setup_generation
    output = root / "candidates.jsonl"
    with CandidateCheckpoint(output) as checkpoint:
        run(config, bank, root, checkpoint)
    if mutation == "prompt":
        bank.tasks[0].prompt = "Changed"
    elif mutation == "temperature":
        bank.tasks[0].temperature = 0.99
    elif mutation == "model":
        config.candidate_model_ids = ["other/model"]
    else:
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        if mutation == "duplicate":
            rows.append(rows[0])
        else:
            rows[0]["cost_usd"] = -1
        output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = output.read_bytes()
    client.calls.clear()
    with CandidateCheckpoint(output, resume=True) as checkpoint, pytest.raises(ValueError):
        run(config, bank, root, checkpoint)
    assert not client.calls
    assert output.read_bytes() == before


def test_old_candidates_without_fingerprint_are_preserved(setup_generation):
    client, config, bank, root = setup_generation
    output = root / "candidates.jsonl"
    with CandidateCheckpoint(output) as checkpoint:
        run(config, bank, root, checkpoint)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row.pop("request_fingerprint")
        row.pop("cost_usd")
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = output.read_bytes()
    client.calls.clear()
    with CandidateCheckpoint(output, resume=True) as checkpoint:
        run(config, bank, root, checkpoint)
    assert not client.calls and output.read_bytes() == before


def test_existing_output_is_never_overwritten_and_writer_is_exclusive(tmp_path):
    path = tmp_path / "candidates.jsonl"
    with CandidateCheckpoint(path):
        with pytest.raises(ValueError, match="gesperrt"):
            with CandidateCheckpoint(path, resume=True):
                pass
    with pytest.raises(FileExistsError):
        with CandidateCheckpoint(path):
            pass
    assert not Path(str(path) + ".lock").exists()


def test_torn_checkpoint_is_preserved(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_bytes(b'{"incomplete":')
    with pytest.raises(ValueError, match="unvollständigen"):
        with CandidateCheckpoint(path, resume=True):
            pass
    assert path.read_bytes() == b'{"incomplete":'


def test_partial_cli_exit_and_cost_report(setup_generation, capsys):
    client, _, _, root = setup_generation
    client.failure_at = 3
    assert gen.main(["--output", "candidates.jsonl", "--report", "report.json",
                     "--concurrency", "1"]) == 2
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert (report["candidates"], report["errors"], report["skipped_candidates"]) == (2, 1, 2)
    assert report["skipped"] == {"unbekannte_aliase": [], "tasks_ohne_lehrer": []}
    assert report["known_cost_usd"] == 0.0002
    assert report["stop_reason"] == "payment_required"
    client.failure_at = None
    assert gen.main(["--output", "candidates.jsonl", "--report", "report.json",
                     "--concurrency", "1", "--resume"]) == 0
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert (report["candidates"], report["reused_candidates"], report["new_candidates"]) == (5, 2, 3)
    assert report["known_cost_usd"] == 0.0005


def test_provider_error_does_not_echo_response_or_credentials():
    async def check():
        async def handler(request):
            return httpx.Response(402, json={"error": "sk-or-v1-SENSITIVE secret-header"})
        client = OpenRouterClient(api_key="synthetic-key")
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(TeacherHTTPError) as error:
                await client.generate("synthetic/model", [])
            assert error.value.status_code == 402
            assert str(error.value) == "Teacher HTTP 402"
        finally:
            await client.close()
    asyncio.run(check())


def test_provider_fractional_cost_is_preserved():
    async def check():
        async def handler(request):
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 7, "cost": 0.0001},
            })
        client = OpenRouterClient(api_key="synthetic-key")
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            response = await client.generate("synthetic/model", [])
            assert response.cost_usd == 0.0001
            assert "cost" not in response.usage
            assert response.usage["prompt_tokens"] == 15
        finally:
            await client.close()
    asyncio.run(check())


def test_402_allows_only_already_inflight_requests(setup_generation):
    client, config, bank, root = setup_generation
    async def generate(model_id, messages, **kwargs):
        client.calls.append(messages[0]["content"])
        number = len(client.calls)
        await asyncio.sleep(0)
        if number == 1:
            raise TeacherHTTPError(402)
        return TeacherResponse(model_id=model_id, content="OK")
    client.generate = generate
    with CandidateCheckpoint(root / "candidates.jsonl") as checkpoint:
        rows, outcomes, _, _ = run(config, bank, root, checkpoint, concurrency=4)
    assert len(client.calls) == 4
    assert len(rows) == 3
    assert sum(bool(error.get("skipped")) for error in outcomes) == 1


def test_56_of_84_resume_makes_exactly_28_requests(setup_generation):
    client, config, bank, root = setup_generation
    bank.tasks = [
        bank.tasks[0].model_copy(update={"id": f"task.{i}", "prompt": f"Synthetic {i}"})
        for i in range(84)
    ]
    output = root / "candidates.jsonl"
    client.failure_at = 57
    with CandidateCheckpoint(output) as checkpoint:
        rows, _, _, _ = run(config, bank, root, checkpoint)
    assert len(rows) == 56
    before = output.read_bytes()
    client.calls.clear()
    client.failure_at = None
    with CandidateCheckpoint(output, resume=True) as checkpoint:
        rows, errors, _, _ = run(config, bank, root, checkpoint)
    assert len(rows) == 84 and not errors
    assert len(client.calls) == 28
    assert output.read_bytes().startswith(before)


def test_cli_preserves_custom_teachers_token_budget_and_skipped_report(
    setup_generation, monkeypatch, capsys
):
    client, config, bank, root = setup_generation
    bank.tasks = bank.tasks[:1]
    bank.tasks[0].teacher_aliases.append("missing_teacher")
    loaded_paths = []

    def load_configs(project_root, path=None):
        assert project_root == root
        loaded_paths.append(path)
        return [config]

    original_generate = client.generate

    async def generate(model_id, messages, **kwargs):
        assert kwargs["max_tokens"] == 37
        return await original_generate(model_id, messages, **kwargs)

    monkeypatch.setattr(gen, "load_teacher_configs", load_configs)
    client.generate = generate
    assert gen.main([
        "--output", "candidates.jsonl", "--report", "report.json",
        "--teachers", "custom-teachers.yaml", "--max-tokens", "37",
    ]) == 2
    assert loaded_paths == [root / "custom-teachers.yaml"]
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["skipped"] == {
        "unbekannte_aliase": ["missing_teacher"], "tasks_ohne_lehrer": [],
    }
    assert report["skipped_candidates"] == 0
    assert report["expected_candidates"] == 2
    assert report["candidates"] == 1 and report["status"] == "partial"
    assert "missing_teacher" in capsys.readouterr().err


@pytest.mark.parametrize("max_tokens", ["0", "-1", "32769"])
def test_cli_rejects_invalid_token_budgets_before_teacher_calls(
    setup_generation, max_tokens
):
    client, _, _, root = setup_generation
    assert gen.main([
        "--output", "candidates.jsonl", "--report", "report.json",
        "--max-tokens", max_tokens,
    ]) == 1
    assert not client.calls
    assert not (root / "candidates.jsonl").exists()
