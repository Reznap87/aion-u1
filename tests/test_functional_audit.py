from pathlib import Path

import pytest
from pydantic import ValidationError

from aion_u1 import audit
from aion_u1.cli import load_cases
from aion_u1.models import VerifierSpec
from aion_u1.verifiers import VerificationResult


def spec(**overrides):
    data = dict(type="python_function", engine="v2", function="answer",
                function_cases=[{"args": [41], "expected": 42}])
    data.update(overrides)
    return VerifierSpec(**data)


def test_functional_audit_dispatch(monkeypatch):
    image = "python@sha256:" + "a" * 64
    monkeypatch.setenv("AION_VERIFIER_DOCKER_IMAGE", image)

    def verify(response, **kwargs):
        assert response == "def answer(x): return x + 1"
        assert kwargs["docker_image"] == image
        assert kwargs["cases"] == [{"args": [41], "expected": 42}]
        return VerificationResult(1.0, True, "pass", "python_function")

    monkeypatch.setattr(audit, "verify_function", verify)
    assert audit.verify_response("def answer(x): return x + 1", spec()).passed


def test_missing_image_fails_closed(monkeypatch):
    monkeypatch.delenv("AION_VERIFIER_DOCKER_IMAGE", raising=False)
    result = audit.verify_response("pass", spec())
    assert result.error and result.score is None


@pytest.mark.parametrize("overrides", [
    {"engine": "legacy"}, {"function_cases": []}, {"pass_threshold": 0.5},
])
def test_functional_spec_validation(overrides):
    with pytest.raises(ValidationError):
        spec(**overrides)


def test_hard_v2_contains_executable_cases():
    root = Path(__file__).resolve().parents[1]
    cases = load_cases(root / "evaluations/capability_suite/benchmark_hard_v2.yaml")
    functional = [case for case in cases if case.verifier.type == "python_function"]
    assert len(functional) == 2
    assert all(case.verifier.engine == "v2" for case in functional)
