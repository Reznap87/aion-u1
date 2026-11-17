from __future__ import annotations

from datetime import datetime, timezone

from aion_u1.dataset import build_dataset
from aion_u1.models import AuditReport, CaseResult
from aion_u1.reasoning import strip_reasoning


def test_strip_reasoning_entfernt_block() -> None:
    cleaned, stripped = strip_reasoning(
        "<think>Schritt fuer Schritt: 2 + 2 = 4.</think>"
        " Also ist die Antwort 42."
    )

    assert stripped is True
    assert "Schritt fuer Schritt" not in cleaned
    assert cleaned == "Also ist die Antwort 42."


def test_strip_reasoning_lasst_normale_antwort() -> None:
    text = "```python\ndef add(a, b):\n    return a + b\n```"

    assert strip_reasoning(text) == (text, False)


def test_strip_reasoning_lasst_code_tags() -> None:
    """`<analysis>` o. ae. kommen in Code vor und sind kein Reasoning."""

    text = "<analysis>f(3) = 9</analysis> ist das Ergebnis."

    assert strip_reasoning(text) == (text, False)


def _result(case_id: str, response: str) -> CaseResult:
    return CaseResult(
        teacher_alias="tt",
        teacher_model_id="provider/model",
        case_id=case_id,
        capability="cap",
        prompt="42?",
        response=response,
        score=1.0,
        passed=True,
        verifier_type="exact_match",
        latency_ms=1,
    )


def _report(*results: CaseResult) -> AuditReport:
    return AuditReport(
        generated_at=datetime.now(timezone.utc),
        suite="smoke",
        suite_path="smoke.yaml",
        teacher_resolutions=[],
        results=list(results),
        scores={},
    )


def test_dataset_entfernt_reasoning_aus_dem_label() -> None:
    candidates, pending = build_dataset(
        _report(_result("c1", "<think>Denke nach.</think> 42"))
    )

    assert candidates[0]["reasoning_stripped"] is True
    assert candidates[0]["response"] == "42"
    assert pending[0]["messages"][-1]["content"] == "42"


def test_dataset_markiert_externe_antworten_als_untrusted() -> None:
    _, pending = build_dataset(_report(_result("c1", "42")))

    provenance = pending[0]["provenance"]
    assert provenance["external_untrusted"] is True
    assert provenance["teacher_alias"] == "tt"


def test_dataset_id_bleibt_bei_gleichem_label_stabil() -> None:
    first = build_dataset(_report(_result("c1", "42")))
    second = build_dataset(_report(_result("c1", "42")))

    assert first[0][0]["id"] == second[0][0]["id"]


def test_dataset_id_dedupliziert_gleiches_label() -> None:
    """Reasoning gehoert nicht zum Label: gleiches Label, gleiche ID."""

    ohne = build_dataset(_report(_result("c1", "42")))
    mit = build_dataset(_report(_result("c1", "<think>Denke.</think> 42")))

    assert ohne[0][0]["id"] == mit[0][0]["id"]
    assert mit[0][0]["reasoning_stripped"] is True
