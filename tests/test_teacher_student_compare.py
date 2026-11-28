import copy

import pytest

from aion_u1.provenance import sha256_file
from aion_u1.training.compare import compare_reports


@pytest.fixture
def reports(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text("""
cases:
  - id: code.one
    capability: code.synthesis
    prompt: answer
    verifier: {type: exact_match, values: ["42"]}
  - id: math.one
    capability: reasoning.logical
    prompt: calculate
    verifier: {type: exact_match, values: ["42"]}
""", encoding="utf-8")
    capabilities = tmp_path / "capabilities.yaml"
    capabilities.write_text("""
capabilities:
  - {id: code.synthesis, name: Code, description: Code, weight: 0.6}
  - {id: reasoning.logical, name: Math, description: Math, weight: 0.4}
""", encoding="utf-8")
    provenance = {"git_commit": "a" * 40, "inputs": {
        "suite": {"sha256": sha256_file(suite)},
        "capabilities": {"sha256": sha256_file(capabilities)},
    }}

    def row(alias, case_id, capability, prompt, score):
        return dict(teacher_alias=alias, teacher_model_id=alias, case_id=case_id,
                    capability=capability, prompt=prompt, response="42",
                    score=score, passed=score == 1, verifier_type="exact_match",
                    latency_ms=1, max_tokens=1024)

    teacher = {"requested_max_tokens": 1024, "index_version": "1", "provenance": copy.deepcopy(provenance),
               "results": [row("teacher", "code.one", "code.synthesis", "answer", 1),
                           row("teacher", "math.one", "reasoning.logical", "calculate", 0.8)]}
    student = {"base_model": "example/student", "adapter": None,
               "model_identity": {"base": {"kind": "hub", "repo_id": "example/student", "revision": "c" * 40}, "adapter": None},
               "max_tokens": 1024, "index_version": "1", "provenance": copy.deepcopy(provenance),
               "results": [row("student", "code.one", "code.synthesis", "answer", 0.5),
                           row("student", "math.one", "reasoning.logical", "calculate", 0.8)]}
    return teacher, student, suite, capabilities


def test_weighted_priority(reports):
    result = compare_reports(*reports)
    code, math = result["capabilities"]
    assert code["gap_to_target"] == pytest.approx(0.45)
    assert code["priority"] == pytest.approx(0.27)
    assert math["status"] == "target_met"
    assert not result["all_targets_met"]


def test_missing_measurement_not_zero(reports):
    teacher, student, *_ = reports
    student["results"][0].update(score=None, error="Docker unavailable", passed=False)
    code = next(row for row in compare_reports(*reports)["capabilities"]
                if row["capability"] == "code.synthesis")
    assert code["student_score"] is None
    assert code["priority"] is None
    assert code["status"] == "unmeasured"


def test_reject_different_suite(reports):
    teacher, student, *_ = reports
    student["provenance"]["inputs"]["suite"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        compare_reports(*reports)


def test_reject_partial_suite(reports):
    teacher, student, *_ = reports
    student["results"].pop()
    with pytest.raises(ValueError, match="case sets"):
        compare_reports(*reports)


def test_zero_teacher_is_not_meaningful_target(reports):
    teacher, student, *_ = reports
    teacher["results"][0].update(score=0, passed=False)
    code = next(row for row in compare_reports(*reports)["capabilities"]
                if row["capability"] == "code.synthesis")
    assert code["target_score"] is None
    assert code["status"] == "unmeasured"


def test_mismatched_requested_budget(reports):
    teacher, student, *_ = reports
    teacher["requested_max_tokens"] = 512
    with pytest.raises(ValueError, match="token budgets"):
        compare_reports(*reports)


def test_reduced_effective_budget_is_unmeasured(reports):
    teacher, student, *_ = reports
    teacher["results"][0]["max_tokens"] = 128
    row = next(r for r in compare_reports(*reports)["capabilities"]
               if r["capability"] == "code.synthesis")
    assert row["best_teacher_score"] is None
    assert row["status"] == "unmeasured"


def test_unpinned_student_rejected(reports):
    teacher, student, *_ = reports
    student["model_identity"]["base"] = {"kind": "unpinned", "repo_id": "example/student"}
    with pytest.raises(ValueError, match="immutable identity"):
        compare_reports(*reports)


def test_missing_adapter_identity_rejected(reports):
    teacher, student, *_ = reports
    student["adapter"] = "adapter"
    with pytest.raises(ValueError, match="adapter identity"):
        compare_reports(*reports)
