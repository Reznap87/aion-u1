import pytest

from aion_u1.verifiers import VerificationResult, extract_json, score_text


def test_error_has_no_score():
    result = VerificationResult(None, False, "infrastructure", "python_code", True)
    assert result.score is None
    with pytest.raises(ValueError):
        VerificationResult(0.0, False, "infrastructure", "python_code", True)


def test_fenced_json():
    assert extract_json('Result:\n```json\n{"answer": 42}\n```') == {"answer": 42}


@pytest.mark.parametrize("text", ['{"a": 1} {"b": 2}', '{"a": 1, "a": 2}', '{"a": NaN}'])
def test_ambiguous_or_invalid_json(text):
    with pytest.raises(ValueError):
        extract_json(text)


def test_partial_scoring_is_opt_in():
    args = dict(kind="contains_all", values=["a", "b", "c"])
    assert score_text("a b", **args).score == 0.0
    result = score_text("a b", partial=True, **args)
    assert result.score == pytest.approx(2 / 3)
    assert not result.passed


def test_case_sensitivity():
    assert score_text("ABC", kind="contains_all", values=["abc"]).passed
    assert not score_text("ABC", kind="contains_all", values=["abc"], case_insensitive=False).passed


def test_empty_conditions_rejected():
    with pytest.raises(ValueError):
        score_text("anything", kind="contains_all", values=[])


def test_zero_threshold_rejected():
    with pytest.raises(ValueError):
        score_text("wrong", kind="contains_all", values=["right"], pass_threshold=0)


def test_forbidden_condition_cannot_be_compensated():
    assert not score_text("secret", kind="not_contains", values=["secret"], pass_threshold=0.1).passed
