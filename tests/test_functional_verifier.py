import json
import subprocess

import pytest

from aion_u1 import functional_verifier as fv

IMAGE = "python@sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def docker_available(monkeypatch):
    monkeypatch.setattr(fv.shutil, "which", lambda _: "/usr/bin/docker")


def verify(**kwargs):
    return fv.verify_function(
        "def answer(x): return x + 1",
        function="answer",
        cases=[{"args": [41], "expected": 42}],
        docker_image=kwargs.pop("docker_image", IMAGE),
        **kwargs,
    )


def test_host_compares_actual_output(monkeypatch):
    def execute(workdir, image, payload, timeout):
        assert image == IMAGE
        assert json.loads(payload) == {"function": "answer", "args": [41], "kwargs": {}}
        assert sorted(p.name for p in workdir.iterdir()) == ["runner.py", "solution.py"]
        return 0, b"42"
    monkeypatch.setattr(fv, "_execute", execute)
    assert verify().passed


@pytest.mark.parametrize("output", [b"", b"OK", b"0", b"true", b"42 42"])
def test_exit_zero_alone_is_not_success(monkeypatch, output):
    monkeypatch.setattr(fv, "_execute", lambda *args: (0, output))
    result = verify()
    assert result.score == 0.0 and not result.passed and not result.error


def test_mutable_image_rejected_before_execution():
    result = verify(docker_image="python:3.12-slim")
    assert result.error and result.score is None


def test_docker_error_is_unscored(monkeypatch):
    monkeypatch.setattr(fv, "_execute", lambda *args: (125, b""))
    assert verify().score is None


def test_timeout_is_unscored(monkeypatch):
    def execute(*args):
        raise subprocess.TimeoutExpired("docker", 10)
    monkeypatch.setattr(fv, "_execute", execute)
    assert verify().error and verify().score is None


def test_multiple_code_fences_rejected():
    with pytest.raises(ValueError):
        fv.extract_source("```python\npass\n```\n```python\npass\n```")
