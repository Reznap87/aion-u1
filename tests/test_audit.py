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
