from pathlib import Path

from aion_u1.training.tasks import (
    TaskBank,
    TaskVerifier,
    build_docker_test_command,
    find_project_root,
    load_task_bank,
    verify_response,
)


def test_pilot_task_bank_is_valid() -> None:
    root = find_project_root()

    bank = load_task_bank(
        root
        / "training"
        / "configs"
        / "tasks"
        / "pilot.yaml"
    )

    assert len(bank.tasks) == 12
    assert len(bank.task_map()) == len(bank.tasks)

    capabilities = {
        task.capability
        for task in bank.tasks
    }

    assert "code.synthesis" in capabilities
    assert "reasoning.logical" in capabilities
    assert "long_context.retention" in capabilities
    assert "agent.tool_use" in capabilities
    assert "research.verification" in capabilities
    assert "safety.injection_resistance" in capabilities


def test_json_verifier() -> None:
    verifier = TaskVerifier(
        type="json_equals",
        expected={"answer": 42},
    )

    result = verify_response(
        '{"answer": 42}',
        verifier,
    )

    assert result.passed is True
    assert result.score == 1.0


def test_contains_any_verifier() -> None:
    verifier = TaskVerifier(
        type="contains_any",
        values=[
            "ACCESS DENIED",
            "Zugriff verweigert",
        ],
    )

    result = verify_response(
        "ACCESS DENIED",
        verifier,
    )

    assert result.passed is True


def test_docker_command_has_security_boundaries(
    tmp_path: Path,
) -> None:
    verifier = TaskVerifier(
        type="python_code",
        test_code="import unittest",
    )

    command = build_docker_test_command(
        tmp_path,
        verifier,
    )

    assert "--network" in command
    assert (
        command[
            command.index("--network") + 1
        ]
        == "none"
    )

    assert "--read-only" in command
    assert "--cap-drop" in command
    assert "--security-opt" in command
    assert "no-new-privileges" in command
    assert "--pids-limit" in command