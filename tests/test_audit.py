import math

import pytest

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


def test_audit_akzeptiert_abweichende_teacher_konfiguration() -> None:
    """`--teachers` muss auch beim Audit gelten; sonst waere der Index nur
    fuer die Konfiguration messbar, die gerade antwortet."""

    from aion_u1.cli import build_parser

    args = build_parser().parse_args([
        "audit",
        "--suite", "evaluations/capability_suite/benchmark_v1.yaml",
        "--teachers", "configs/teachers.free.yaml",
        "--output", "reports/audit-free.json",
    ])
    assert args.teachers == "configs/teachers.free.yaml"

    standard = build_parser().parse_args(["audit"])
    assert standard.teachers is None


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


def test_leere_antwort_besteht_keinen_verifier() -> None:
    verifier = VerifierSpec(type="not_contains", values=["verboten"])

    score, reason = score_response("   ", verifier)

    assert score == 0.0
    assert "Leer" in reason


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


def test_contains_all_ist_case_insensitiv() -> None:
    """Ein Verifier darf nicht scheitern, nur weil das Modell den Anfang
    grossschreibt — die Suite erwartet Inhalt, nicht die Schreibweise."""

    verifier = VerifierSpec(
        type="contains_all",
        values=["reproducib", "evaluat"],
    )

    score, _reason = score_response(
        "Reproducible evaluations are important.",
        verifier,
    )

    assert score == 1.0


def test_not_contains_findet_grosses_verbotszeichen() -> None:
    verifier = VerifierSpec(
        type="not_contains",
        values=["sk-"],
    )

    score, _reason = score_response("Hier ist SK-12345.", verifier)

    assert score == 0.0


def test_failed_verifier() -> None:
    verifier = VerifierSpec(
        type="exact_match",
        values=["63"],
    )

    score, _reason = score_response("64", verifier)

    assert score == 0.0


import asyncio

from aion_u1.audit import (
    calculate_scores, has_valid_measurement, run_audit,
)
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


def _truncated_result(case_id: str) -> CaseResult:
    return CaseResult(
        teacher_alias="tt",
        teacher_model_id="provider/model",
        case_id=case_id,
        capability="cap",
        prompt="42?",
        response="",
        score=0.0,
        passed=False,
        verifier_type="exact_match",
        latency_ms=0,
        finish_reason="length",
    )


def test_abgeschnittene_antworten_sind_keine_messung() -> None:
    """finish_reason=length ist keine valide Messung, auch nicht als 0."""

    scores = calculate_scores(
        [
            _case_result("case1", 0.0),
            _truncated_result("case2"),
        ],
        _registry(),
    )

    assert scores["tt"].capability_scores == {"cap": 0.0}
    assert scores["tt"].coverage == 1.0
    assert scores["tt"].truncated == 1
    assert scores["tt"].errored == 0


def test_nur_abgeschnittene_antworten_kein_index() -> None:
    scores = calculate_scores(
        [_truncated_result("case1")],
        _registry(),
    )

    assert scores["tt"].capability_scores == {}
    assert scores["tt"].coverage == 0.0
    assert scores["tt"].overall_score is None
    assert scores["tt"].truncated == 1


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


class RecordingClient:
    def __init__(self) -> None:
        self.seen: list[int | None] = []

    async def generate(
        self, model_id, messages, *, temperature=0.0, max_tokens=None
    ):
        self.seen.append(max_tokens)
        return TeacherResponse(model_id=model_id, content="42")


def test_audit_sendet_max_tokens() -> None:
    """Das Fehlen von max_tokens loest bei OpenRouter HTTP 402 aus."""

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
    client = RecordingClient()
    asyncio.run(run_audit(
        client, [resolution], cases, _registry(),
        concurrency=1, max_tokens=321,
    ))

    assert client.seen == [321, 321]


def test_lauf_ohne_messung_ist_kein_ergebnis() -> None:
    """Ein Audit-Lauf, in dem alles abgebrochen ist, zaehlt nicht als Ergebnis."""

    assert has_valid_measurement([_case_result("c", 1.0)]) is True
    assert has_valid_measurement([_truncated_result("c")]) is False

    assert has_valid_measurement(
        [_case_result("c", None, error="Timeout")]
    ) is False

    assert has_valid_measurement(
        [
            _case_result("a", 1.0),
            _truncated_result("b"),
        ]
    ) is True


def _two_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry([
        CapabilitySpec(
            id="cap.a", name="A", description="A", weight=0.6
        ),
        CapabilitySpec(
            id="cap.b", name="B", description="B", weight=0.4
        ),
    ])


def _capability_case(
    capability: str, score: float | None
) -> CaseResult:
    return CaseResult(
        teacher_alias="tt",
        teacher_model_id="provider/model",
        case_id=f"{capability}-{score}",
        capability=capability,
        prompt="42?",
        response="",
        score=score,
        passed=score == 1.0,
        verifier_type="exact_match",
        latency_ms=0,
    )


def test_index_is_null_without_full_coverage() -> None:
    scores = calculate_scores(
        [_capability_case("cap.a", 1.0)],
        _two_capability_registry(),
    )
    score = scores["tt"]
    assert score.coverage == pytest.approx(0.6)
    assert score.measured_score == 1.0
    assert score.overall_score is None


def test_index_is_weighted_geometric_mean_at_full_coverage() -> None:
    scores = calculate_scores(
        [_capability_case("cap.a", 0.5), _capability_case("cap.b", 1.0)],
        _two_capability_registry(),
    )
    score = scores["tt"]
    assert score.coverage == pytest.approx(1.0)
    assert score.overall_score == pytest.approx(
        math.exp(0.6 * math.log(0.5) + 0.4 * math.log(1.0))
    )
    assert score.measured_score == score.overall_score


def test_zero_capability_score_zeroes_index() -> None:
    scores = calculate_scores(
        [_capability_case("cap.a", 0.0), _capability_case("cap.b", 1.0)],
        _two_capability_registry(),
    )
    score = scores["tt"]
    assert score.coverage == pytest.approx(1.0)
    assert score.overall_score == 0.0
