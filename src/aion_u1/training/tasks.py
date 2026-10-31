from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskVerifier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "contains_all",
        "contains_any",
        "not_contains",
        "regex",
        "exact_match",
        "json_equals",
        "python_code",
    ]
    values: list[str] = Field(default_factory=list)
    expected: Any = None
    test_code: str | None = None
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    docker_image: str = "python:3.12-slim"
    pull_image: bool = False

    @model_validator(mode="after")
    def validate_arguments(self) -> "TaskVerifier":
        if self.type in {
            "contains_all",
            "contains_any",
            "not_contains",
            "regex",
            "exact_match",
        } and not self.values:
            raise ValueError(
                f"Verifier {self.type} benötigt mindestens einen Wert."
            )

        if (
            self.type == "json_equals"
            and "expected" not in self.model_fields_set
        ):
            raise ValueError("json_equals benötigt expected.")

        if self.type == "python_code" and not self.test_code:
            raise ValueError("python_code benötigt test_code.")

        return self


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,95}$")
    capability: str
    prompt: str = Field(min_length=1)
    modality: Literal["text", "image"] = "text"
    image_path: str | None = None
    image_mime_type: str = "image/png"
    verifier: TaskVerifier
    teacher_aliases: list[str] = Field(min_length=1)
    samples_per_teacher: int = Field(default=1, ge=1, le=8)
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int | None = Field(default=900, gt=0, le=32_768)


class TaskBank(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    tasks: list[TaskSpec] = Field(min_length=1)

    def task_map(self) -> dict[str, TaskSpec]:
        result: dict[str, TaskSpec] = {}

        for task in self.tasks:
            if task.id in result:
                raise ValueError(f"Doppelte Task-ID: {task.id}")
            result[task.id] = task

        return result


@dataclass(frozen=True)
class VerificationResult:
    score: float
    passed: bool
    reason: str
    method: str
    error: bool = False
def find_project_root() -> Path:
    current = Path.cwd().resolve()

    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "aion_u1").is_dir()
        ):
            return candidate

    return Path(__file__).resolve().parents[3]


def load_task_bank(path: Path) -> TaskBank:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Task-Bank muss ein YAML-Mapping sein.")

    return TaskBank.model_validate(raw)


def build_task_input(
    task: TaskSpec,
    root: Path | None = None,
) -> str | list[dict[str, Any]]:
    root = root or find_project_root()

    if task.modality == "text":
        return task.prompt

    if not task.image_path:
        raise ValueError(f"Task {task.id} benötigt image_path.")

    image_path = root / task.image_path

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Bild für Task {task.id} fehlt: {image_path}"
        )

    encoded = base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")

    data_url = (
        f"data:{task.image_mime_type};base64,{encoded}"
    )

    return [
        {"type": "text", "text": task.prompt},
        {
            "type": "image_url",
            "image_url": {"url": data_url},
        },
    ]


def _extract_json(text: str) -> Any:
    stripped = text.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    starts = [
        position
        for position in (
            stripped.find("{"),
            stripped.find("["),
        )
        if position >= 0
    ]

    if not starts:
        raise ValueError("Kein JSON gefunden.")

    start = min(starts)
    opener = stripped[start]
    closer = "}" if opener == "{" else "]"
    end = stripped.rfind(closer)

    if end < start:
        raise ValueError("Unvollständiges JSON.")

    return json.loads(stripped[start : end + 1])


def extract_python_source(response: str) -> str:
    text = response.strip()

    fenced = re.search(
        r"```(?:python|py)?\s*\n(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:
        return fenced.group(1).strip()

    return text
def build_docker_test_command(
    workdir: Path,
    verifier: TaskVerifier,
) -> list[str]:
    mount = (
        f"type=bind,source={workdir.resolve().as_posix()},"
        "target=/workspace,readonly"
    )

    return [
        "docker",
        "run",
        "--rm",
        "--pull",
        "always" if verifier.pull_image else "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "256m",
        "--cpus",
        "1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        mount,
        "--workdir",
        "/workspace",
        verifier.docker_image,
        "python",
        "test_solution.py",
    ]


def run_python_tests(
    response: str,
    verifier: TaskVerifier,
) -> VerificationResult:
    if not verifier.test_code:
        return VerificationResult(
            score=0.0,
            passed=False,
            reason="Kein Testcode vorhanden.",
            method="python_code",
            error=True,
        )

    if shutil.which("docker") is None:
        return VerificationResult(
            score=0.0,
            passed=False,
            reason=(
                "Docker ist nicht verfügbar. Der Code wird nicht "
                "unsicher auf dem Host ausgeführt."
            ),
            method="python_code:docker",
            error=True,
        )

    source = extract_python_source(response)

    try:
        with tempfile.TemporaryDirectory(
            prefix="aion-u1-verifier-"
        ) as temporary_directory:
            workdir = Path(temporary_directory)

            (workdir / "solution.py").write_text(
                source,
                encoding="utf-8",
            )
            (workdir / "test_solution.py").write_text(
                verifier.test_code,
                encoding="utf-8",
            )

            command = build_docker_test_command(
                workdir,
                verifier,
            )

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=verifier.timeout_seconds,
                check=False,
            )

            if completed.returncode == 0:
                return VerificationResult(
                    score=1.0,
                    passed=True,
                    reason="Alle Python-Tests bestanden.",
                    method="python_code:docker",
                )

            details = (
                completed.stdout + "\n" + completed.stderr
            ).strip()

            return VerificationResult(
                score=0.0,
                passed=False,
                reason=(
                    "Python-Tests fehlgeschlagen: "
                    + details[-1200:]
                ),
                method="python_code:docker",
            )

    except subprocess.TimeoutExpired:
        return VerificationResult(
            score=0.0,
            passed=False,
            reason=(
                "Docker-Test hat das Zeitlimit überschritten: "
                f"{verifier.timeout_seconds}s"
            ),
            method="python_code:docker",
            error=True,
        )

    except OSError as exc:
        return VerificationResult(
            score=0.0,
            passed=False,
            reason=f"Docker konnte nicht gestartet werden: {exc}",
            method="python_code:docker",
            error=True,
        )
def verify_response(
    response: str,
    verifier: TaskVerifier,
) -> VerificationResult:
    text = response.strip()

    if verifier.type == "python_code":
        return run_python_tests(text, verifier)

    if verifier.type == "contains_all":
        missing = [
            value
            for value in verifier.values
            if value not in text
        ]

        return VerificationResult(
            score=0.0 if missing else 1.0,
            passed=not missing,
            reason=(
                f"Fehlende Werte: {missing}"
                if missing
                else "Alle Werte gefunden."
            ),
            method=verifier.type,
        )

    if verifier.type == "contains_any":
        matched = [
            value
            for value in verifier.values
            if value in text
        ]

        return VerificationResult(
            score=1.0 if matched else 0.0,
            passed=bool(matched),
            reason=(
                f"Gefundene Werte: {matched}"
                if matched
                else "Keiner der erlaubten Werte gefunden."
            ),
            method=verifier.type,
        )

    if verifier.type == "not_contains":
        forbidden = [
            value
            for value in verifier.values
            if value in text
        ]

        return VerificationResult(
            score=0.0 if forbidden else 1.0,
            passed=not forbidden,
            reason=(
                f"Verbotene Werte gefunden: {forbidden}"
                if forbidden
                else "Keine verbotenen Werte gefunden."
            ),
            method=verifier.type,
        )

    if verifier.type == "exact_match":
        expected = verifier.values[0].strip()

        return VerificationResult(
            score=1.0 if text == expected else 0.0,
            passed=text == expected,
            reason=(
                "Exakter Match."
                if text == expected
                else f"Erwartet wurde {expected!r}."
            ),
            method=verifier.type,
        )

    if verifier.type == "regex":
        matched = next(
            (
                pattern
                for pattern in verifier.values
                if re.search(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                )
            ),
            None,
        )

        return VerificationResult(
            score=1.0 if matched else 0.0,
            passed=matched is not None,
            reason=(
                f"Regex gefunden: {matched}"
                if matched
                else "Kein Regex gefunden."
            ),
            method=verifier.type,
        )

    if verifier.type == "json_equals":
        try:
            actual = _extract_json(text)
        except (ValueError, json.JSONDecodeError) as exc:
            return VerificationResult(
                score=0.0,
                passed=False,
                reason=f"JSON ungültig: {exc}",
                method=verifier.type,
            )

        passed = actual == verifier.expected

        return VerificationResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=(
                "JSON entspricht Erwartung."
                if passed
                else (
                    f"Verwartet={verifier.expected!r}, "
                    f"actual={actual!r}"
                )
            ),
            method=verifier.type,
        )

    return VerificationResult(
        score=0.0,
        passed=False,
        reason=f"Unbekannter Verifier: {verifier.type}",
        method=verifier.type,
        error=True,
    )