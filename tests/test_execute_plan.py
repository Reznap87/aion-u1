import asyncio
import json

import pytest

from aion_u1.models import TeacherResponse
from aion_u1.training.execute_plan import prepare_jobs, execute_jobs
from aion_u1.training.tasks import TaskBank, TaskSpec, TaskVerifier


def bank(prompt="synthetic question"):
    return TaskBank(tasks=[TaskSpec(
        id="task.one", capability="code.synthesis", prompt=prompt,
        teacher_aliases=["allowed"], samples_per_teacher=2,
        verifier=TaskVerifier(type="exact_match", values=["ok"]),
        max_tokens=32,
    )])


def proposal():
    return {"selected_tasks": [{"task_id": "task.one", "teacher_aliases": ["allowed"],
                               "samples_per_teacher": 2}],
            "planned_candidates": 2, "max_candidates": 2}


def test_mapping_and_budget_preflight():
    jobs = prepare_jobs(proposal(), bank(), {"allowed": "example/model:free"})
    assert len(jobs) == 2
    assert jobs[0][1].model_id == "example/model:free"
    with pytest.raises(ValueError, match="free model mapping"):
        prepare_jobs(proposal(), bank(), {"allowed": "example/paid"})


def test_privacy_preflight():
    with pytest.raises(ValueError, match="privacy"):
        prepare_jobs(proposal(), bank("contact test@example.com"), {"allowed": "example/model:free"})


def test_candidate_cap_rechecked():
    plan = proposal()
    plan["max_candidates"] = 1
    with pytest.raises(ValueError, match="count mismatch"):
        prepare_jobs(plan, bank(), {"allowed": "example/model:free"})


def test_execution_preserves_unapproved_state(tmp_path):
    class Client:
        async def generate(self, model_id, messages, **kwargs):
            return TeacherResponse(model_id=model_id, content="ok", max_tokens=32)

    jobs = prepare_jobs(proposal(), bank(), {"allowed": "example/model:free"})
    output = tmp_path / "candidates.jsonl"
    assert asyncio.run(execute_jobs(Client(), jobs, tmp_path, output)) == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(not row["provenance"]["training_approved"] for row in rows)
    assert all(row["provenance"]["license"] == "unreviewed" for row in rows)
    assert all(row["verification"]["status"] == "pending" for row in rows)


def test_execution_stops_after_first_provider_error(tmp_path):
    class Client:
        calls = 0

        async def generate(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("unavailable")

    jobs = prepare_jobs(proposal(), bank(), {"allowed": "example/model:free"})
    client = Client()
    output = tmp_path / "candidates.jsonl"
    with pytest.raises(RuntimeError, match="generation stopped"):
        asyncio.run(execute_jobs(client, jobs, tmp_path, output))
    assert client.calls == 1
    assert output.read_text() == ""
