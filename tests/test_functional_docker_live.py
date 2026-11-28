"""Opt-in live Docker integration tests using only synthetic source code."""
import os

import pytest

from aion_u1.functional_verifier import verify_function

IMAGE = os.environ.get("AION_VERIFIER_DOCKER_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="Digest-pinned test image not configured")


def verify(source, expected=42, timeout=15):
    assert IMAGE
    return verify_function(
        source, function="answer",
        cases=[{"args": [41], "expected": expected}],
        docker_image=IMAGE, timeout_seconds=timeout,
    )


def test_real_function_passes():
    result = verify("def answer(x): return x + 1")
    assert result.passed and result.score == 1.0, result.reason


def test_real_wrong_result_fails():
    result = verify("def answer(x): return x - 1")
    assert not result.passed and result.score == 0.0, result.reason


def test_real_early_success_exit_fails():
    result = verify("import sys\nsys.exit(0)")
    assert not result.passed and result.score == 0.0, result.reason


def test_real_unprivileged_user():
    result = verify("import os\ndef answer(x): return os.getuid()", 65534)
    assert result.passed, result.reason


def test_real_excess_output_fails():
    result = verify("def answer(x):\n    print('a' * 100000)\n    return x + 1")
    assert not result.passed and result.score == 0.0, result.reason


def test_real_timeout_is_unscored():
    result = verify("def answer(x):\n    while True: pass", timeout=5)
    assert result.error and result.score is None, result.reason
