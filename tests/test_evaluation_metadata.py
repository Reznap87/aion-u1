from aion_u1.audit import evaluation_metadata
from aion_u1.models import AuditCase, VerifierSpec


def case(verifier):
    return AuditCase(id="test.case", capability="code.synthesis",
                     prompt="Test", verifier=verifier)


def test_legacy_version():
    assert evaluation_metadata([case(VerifierSpec(type="exact_match", values=["42"]))]) == {
        "index_version": "1", "verifier_image": None,
    }


def test_v2_version():
    verifier = VerifierSpec(type="contains_all", values=["a", "b"],
                            engine="v2", partial=True)
    assert evaluation_metadata([case(verifier)])["index_version"] == "2"


def test_functional_image_provenance(monkeypatch):
    verifier = VerifierSpec(type="python_function", engine="v2", function="answer",
                            function_cases=[{"args": [1], "expected": 2}])
    image = "python@sha256:" + "a" * 64
    monkeypatch.setenv("AION_VERIFIER_DOCKER_IMAGE", image)
    assert evaluation_metadata([case(verifier)]) == {
        "index_version": "2", "verifier_image": image,
    }
    monkeypatch.setenv("AION_VERIFIER_DOCKER_IMAGE", "invalid")
    assert evaluation_metadata([case(verifier)])["verifier_image"] is None
