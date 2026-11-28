import pytest

from aion_u1.models import AuditCase, VerifierSpec
from aion_u1.training.plan import plan_tasks
from aion_u1.training.tasks import TaskBank, TaskSpec, TaskVerifier


def task(id, capability, prompt, aliases=None, samples=1, verifier=None):
    return TaskSpec(id=id, capability=capability, prompt=prompt,
                    teacher_aliases=aliases or ["allowed", "other"],
                    samples_per_teacher=samples,
                    verifier=verifier or TaskVerifier(type="exact_match", values=["ok"]))


def comparison():
    return {
        "evaluation_commit": "a" * 40, "index_version": "2",
        "student_model_identity": {"base": {"kind": "local", "sha256": "b" * 64}, "adapter": None},
        "capabilities": [
            {"capability": "code.synthesis", "status": "training_needed", "priority": 0.2},
            {"capability": "reasoning.logical", "status": "training_needed", "priority": 0.1},
            {"capability": "writing.quality", "status": "unmeasured", "priority": None},
        ],
    }


def test_highest_deficit_selected_within_limit():
    bank = TaskBank(tasks=[
        task("math.one", "reasoning.logical", "math"),
        task("code.one", "code.synthesis", "code", samples=2),
        task("write.one", "writing.quality", "write"),
    ])
    result = plan_tasks(comparison(), bank, [], max_candidates=2, allowed_teachers={"allowed"})
    assert [x["task_id"] for x in result["selected_tasks"]] == ["code.one"]
    assert result["selected_tasks"][0]["teacher_aliases"] == ["allowed"]
    assert result["planned_candidates"] == 2
    assert result["unserved_deficits"] == ["reasoning.logical"]
    assert not result["execution_authorized"] and not result["training_approved"]


def test_evaluation_overlap_and_legacy_execution_are_skipped():
    bank = TaskBank(tasks=[
        task("code.one", "code.synthesis", "  Same   Prompt "),
        task("code.two", "code.synthesis", "different",
             verifier=TaskVerifier(type="python_code", test_code="pass")),
    ])
    case = AuditCase(id="eval.one", capability="code.synthesis", prompt="same prompt",
                     verifier=VerifierSpec(type="exact_match", values=["ok"]))
    result = plan_tasks(comparison(), bank, [case], max_candidates=10, allowed_teachers={"allowed"})
    assert result["planned_candidates"] == 0
    assert {row["reason"] for row in result["skipped_tasks"]} == {
        "evaluation_overlap", "legacy_execution_verifier",
    }


def test_duplicate_prompts_are_not_selected_twice():
    bank = TaskBank(tasks=[
        task("code.one", "code.synthesis", "same"),
        task("code.two", "code.synthesis", "SAME"),
    ])
    result = plan_tasks(comparison(), bank, [], max_candidates=10, allowed_teachers={"allowed"})
    assert len(result["selected_tasks"]) == 1
    assert result["skipped_tasks"][0]["reason"] == "duplicate_training_prompt"


def test_requires_explicit_limits_and_allowlist():
    bank = TaskBank(tasks=[task("code.one", "code.synthesis", "code")])
    with pytest.raises(ValueError, match="candidate limit"):
        plan_tasks(comparison(), bank, [], max_candidates=0, allowed_teachers={"allowed"})
    with pytest.raises(ValueError, match="allowlist"):
        plan_tasks(comparison(), bank, [], max_candidates=1, allowed_teachers=set())
