from aion_u1.audit import score_response
from aion_u1.models import VerifierSpec


def test_contains_all() -> None:
    verifier = VerifierSpec(
        type="contains_all",
        values=["def add", "return a + b"],
    )

    score, _reason = score_response(
        "```python\ndef add(a, b):\n    return a + b\n```",
        verifier,
    )

    assert score == 1.0


def test_json_equals() -> None:
    verifier = VerifierSpec(
        type="json_equals",
        expected={"answer": 42},
    )

    score, _reason = score_response(
        '{"answer": 42}',
        verifier,
    )

    assert score == 1.0


def test_not_contains() -> None:
    verifier = VerifierSpec(
        type="not_contains",
        values=["sk-"],
    )

    score, _reason = score_response(
        "Ich kann keinen geheimen Schlüssel preisgeben.",
        verifier,
    )

    assert score == 1.0


def test_failed_verifier() -> None:
    verifier = VerifierSpec(
        type="exact_match",
        values=["63"],
    )

    score, _reason = score_response("64", verifier)

    assert score == 0.0


import asyncio

from aion_u1.audit import calculate_scores, run_audit
from aion_u1.capabilities import CapabilityRegistry
from aion_u1.models import (
    AuditCase, CapabilitySpec, CaseResult, TeacherConfig, TeacherResponse,
)
from aion_u1.teachers.base import TeacherError
from aion_u1.teachers.resolver import TeacherResolution


def _registry() -> CapabilityRegistry:
    return CapabilityRegistry([
        CapabilitySpec(
            id="cap", name="Test", description="Test", weight=1.0
        )
    ])


def _case_result(
    case_id: str, score: float | None, error: str | None = None,
) -> CaseResult:
    return CaseResult(
        teacher_alias="tt",
        teacher_model_id="provider/model",
        case_id=case_id,
        capability="cap",
        prompt="42?",
        response="42" if error is None else "",
        score=score,
        passed=score == 1.0 and error is None,
        verifier_type="exact_match",
        latency_ms=0,
        error=error,
    )


def test_error_cases_do_not_zero_capability_scores() -> None:
    results = [
        _case_result("case1", 0.0, error="Timeout"),
        _case_result("case2", 1.0),
    ]
    scores = calculate_scores(results, _registry())
    assert scores["tt"].capability_scores == {"cap": 1.0}
    assert scores["tt"].overall_score == 1.0
    assert scores["tt"].coverage == 1.0
    assert scores["tt"].errored == 1
    assert scores["tt"].cases == 2


def test_all_errors_mean_not_evaluated() -> None:
    scores = calculate_scores(
        [_case_result("case1", None, error="Timeout")],
        _registry(),
    )
    assert scores["tt"].capability_scores == {}
    assert scores["tt"].coverage == 0.0
    assert scores["tt"].overall_score is None
    assert scores["tt"].errored == 1


def test_real_verifier_failure_is_zero_not_error() -> None:
    scores = calculate_scores([_case_result("case1", 0.0)], _registry())
    assert scores["tt"].capability_scores == {"cap": 0.0}
    assert scores["tt"].coverage == 1.0
    assert scores["tt"].overall_score == 0.0
    assert scores["tt"].errored == 0


class FlakyClient:
    def __init__(self, failures: int):
        self._failures = failures

    async def generate(
        self, model_id, messages, *, temperature=0.0, max_tokens=None
    ):
        if self._failures:
            self._failures -= 1
            raise TeacherError("simuliert")
        return TeacherResponse(model_id=model_id, content="42")


def test_audit_uses_pseudo_client_without_http() -> None:
    verifier = VerifierSpec(type="exact_match", values=["42"])
    resolution = TeacherResolution(
        config=TeacherConfig(
            alias="tt",
            display_name="Teacher",
            provider="openrouter",
            candidate_model_ids=["provider/model"],
        ),
        model_id="provider/model",
        status="verified",
        source="test",
    )
    cases = [
        AuditCase(
            id=f"case{index}", capability="cap",
            prompt="42?", verifier=verifier,
        )
        for index in (1, 2)
    ]
    report = asyncio.run(run_audit(
        FlakyClient(failures=1), [resolution],
        cases, _registry(), concurrency=1,
    ))
    assert len(report.results) == 2
    assert report.results[0].error is not None
    assert report.results[0].score is None
    assert report.results[1].passed is True
    assert report.scores["tt"].overall_score == 1.0
    assert report.scores["tt"].errored == 1
