import subprocess

from aion_u1.training import tasks


def verifier():
    return tasks.TaskVerifier(type="python_code", test_code="import unittest")


def test_no_docker_is_error(monkeypatch):
    monkeypatch.setattr(tasks.shutil, "which", lambda _: None)
    result = tasks.run_python_tests("pass", verifier())
    assert result.error and not result.passed


def test_docker_output_discarded_and_infra_error(monkeypatch):
    monkeypatch.setattr(tasks.shutil, "which", lambda _: "/usr/bin/docker")

    def run(command, **kwargs):
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert command[command.index("--user") + 1] == "65534:65534"
        assert command[command.index("--log-driver") + 1] == "none"
        return subprocess.CompletedProcess(command, 125)

    monkeypatch.setattr(tasks.subprocess, "run", run)
    result = tasks.run_python_tests("pass", verifier())
    assert result.error and not result.passed


def test_timeout_removes_named_container(monkeypatch):
    monkeypatch.setattr(tasks.shutil, "which", lambda _: "/usr/bin/docker")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tasks.subprocess, "run", run)
    result = tasks.run_python_tests("pass", verifier())
    assert result.error and not result.passed
    name = commands[0][commands[0].index("--name") + 1]
    assert commands[1] == ["docker", "rm", "-f", name]
