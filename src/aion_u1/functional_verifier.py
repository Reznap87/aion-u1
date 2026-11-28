"""Functional verification using JSON inputs and host-side expected outputs.

Only the candidate, function name and current input enter the container.
Expected outputs and the scoring logic remain outside it.
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .verifiers import VerificationResult, extract_json

MAX_OUTPUT = 64 * 1024
MAX_SOURCE = 128 * 1024
MAX_CASES = 32
METHOD = "python_function:docker:v2"

_RUNNER = """import json
import sys
import solution

payload = json.load(sys.stdin)
result = getattr(solution, payload["function"])(*payload["args"], **payload["kwargs"])
print(json.dumps(result, allow_nan=False))
"""


def _error(reason: str) -> VerificationResult:
    return VerificationResult(None, False, reason, METHOD, True)


def _failed(reason: str) -> VerificationResult:
    return VerificationResult(0.0, False, reason, METHOD)


def extract_source(response: str) -> str:
    text = response.strip()
    if "```" not in text:
        return text
    pattern = r"^```(?:python|py)?[ \t]*\r?\n(.*?)^```[ \t]*$"
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if len(matches) != 1 or text.count("```") != 2:
        raise ValueError("Expected one Python code block")
    return matches[0].strip()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _execute(
    workdir: Path, image: str, payload: bytes, timeout: float
) -> tuple[int, bytes]:
    name = "aion-function-" + uuid.uuid4().hex
    command = [
        "docker", "run", "--rm", "--name", name, "--pull", "never",
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "65534:65534",
        "--pids-limit", "64", "--memory", "256m", "--memory-swap", "256m",
        "--cpus", "1", "--log-driver", "none",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m",
        "--mount", f"type=bind,source={workdir.resolve().as_posix()},target=/workspace,readonly",
        "--workdir", "/workspace", "--entrypoint", "python", "-i", image,
        "-B", "/workspace/runner.py",
    ]
    deadline = time.monotonic() + timeout
    # A file-backed stdin avoids blocking on a child that never reads its input.
    with tempfile.TemporaryFile() as input_file:
        input_file.write(payload)
        input_file.seek(0)
        process = subprocess.Popen(command, stdin=input_file, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
        output: queue.Queue[bytes | BaseException] = queue.Queue(maxsize=1)

        def read_output() -> None:
            try:
                assert process.stdout is not None
                output.put(process.stdout.read(MAX_OUTPUT + 1))
            except BaseException as exc:
                output.put(exc)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        completed = False
        try:
            try:
                data = output.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise subprocess.TimeoutExpired(command, timeout) from exc
            if isinstance(data, BaseException):
                raise OSError("Could not read container output") from data
            if len(data) > MAX_OUTPUT:
                raise ValueError("Container output limit exceeded")
            code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
            completed = True
            return code, data
        finally:
            if not completed:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    subprocess.run(["docker", "rm", "-f", name], timeout=10,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            reader.join(timeout=1)
            if not reader.is_alive() and process.stdout is not None:
                process.stdout.close()


def verify_function(
    response: str, *, function: str, cases: list[dict[str, Any]],
    docker_image: str, timeout_seconds: float = 10.0,
) -> VerificationResult:
    """Run each input in a fresh container, compare outputs in trusted host code."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", function):
        return _error("Invalid function name")
    if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", docker_image):
        return _error("A digest-pinned Docker image is required")
    if not 0 < timeout_seconds <= 120 or not 1 <= len(cases) <= MAX_CASES:
        return _error("Invalid timeout or case count")
    prepared: list[tuple[bytes, str]] = []
    try:
        for case in cases:
            if set(case) - {"args", "kwargs", "expected"} or "expected" not in case:
                raise ValueError("Invalid test case")
            args, kwargs = case.get("args", []), case.get("kwargs", {})
            if not isinstance(args, list) or not isinstance(kwargs, dict):
                raise ValueError("Invalid arguments")
            payload = _canonical({"function": function, "args": args, "kwargs": kwargs}).encode()
            expected = _canonical(case["expected"])
            if len(payload) > MAX_OUTPUT or len(expected.encode()) > MAX_OUTPUT:
                raise ValueError("Oversized test case")
            prepared.append((payload, expected))
    except (TypeError, ValueError):
        return _error("Invalid JSON test specification")
    if shutil.which("docker") is None:
        return _error("Docker unavailable")
    try:
        source = extract_source(response)
        if not source or len(source.encode()) > MAX_SOURCE:
            return _failed("Invalid candidate source")
    except (ValueError, UnicodeError):
        return _failed("Invalid candidate source")

    try:
        with tempfile.TemporaryDirectory(prefix="aion-function-") as tmp:
            workdir = Path(tmp)
            workdir.chmod(0o755)
            for filename, content in (("solution.py", source), ("runner.py", _RUNNER)):
                path = workdir / filename
                path.write_text(content, encoding="utf-8")
                path.chmod(0o644)
            for payload, expected in prepared:
                try:
                    code, output = _execute(workdir, docker_image, payload, timeout_seconds)
                except ValueError:
                    return _failed("Candidate output limit exceeded")
                if code in {125, 126, 127}:
                    return _error("Docker execution unavailable")
                if code != 0:
                    return _failed("Candidate execution failed")
                try:
                    actual = _canonical(extract_json(output.decode("utf-8")))
                except (ValueError, UnicodeError, RecursionError):
                    return _failed("Invalid candidate output")
                if actual != expected:
                    return _failed("Functional assertion failed")
    except subprocess.TimeoutExpired:
        return _error("Container execution timed out")
    except OSError:
        return _error("Container execution unavailable")
    return VerificationResult(1.0, True, "All functional assertions passed", METHOD)
