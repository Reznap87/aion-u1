import pytest

from aion_u1.audit import verify_response as audit_verify
from aion_u1.models import VerifierSpec
from aion_u1.training.tasks import TaskVerifier, verify_response as training_verify


def test_audit_partial_opt_in():
    result = audit_verify("a b", VerifierSpec(
        type="contains_all", values=["a", "b", "c"],
        engine="v2", partial=True, pass_threshold=0.6,
    ))
    assert result.score == pytest.approx(2 / 3)
    assert result.passed


@pytest.mark.parametrize("engine", ["legacy", "v2"])
def test_training_contains_case_sensitive(engine):
    result = training_verify("ABC", TaskVerifier(
        type="contains_all", values=["abc"], engine=engine,
    ))
    assert not result.passed


@pytest.mark.parametrize("engine", ["legacy", "v2"])
def test_training_regex_case_insensitive(engine):
    result = training_verify("ABC", TaskVerifier(
        type="regex", values=["abc"], engine=engine,
    ))
    assert result.passed


def test_training_v2_rejects_ambiguous_json():
    result = training_verify('{"a": 1, "a": 2}', TaskVerifier(
        type="json_equals", expected={"a": 2}, engine="v2",
    ))
    assert not result.passed
    assert not result.error


def test_legacy_audit_remains_default():
    assert VerifierSpec(type="contains_all", values=["a"]).engine == "legacy"
