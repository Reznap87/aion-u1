import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def measurement():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("measure_synthetic_loading", scripts / "measure_synthetic_loading.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    return module


def test_process_tree_includes_child_memory(measurement):

    class Process:
        def __init__(self, rss, private, children=()):
            self.info = SimpleNamespace(rss=rss, private=private)
            self.members = children

        def memory_info(self):
            return self.info

        def children(self, recursive):
            assert recursive is True
            return self.members

    assert measurement.process_tree_memory(Process(4, 5, [Process(100, 150)])) == (104, 155)


def test_cleanup_after_process_lookup_failure(measurement, monkeypatch, tmp_path):
    events = []
    child = SimpleNamespace(pid=123, poll=lambda: None,
        kill=lambda: events.append("kill"), wait=lambda timeout: events.append(("wait", timeout)))
    monkeypatch.setattr(measurement.subprocess, "Popen", lambda *args, **kwargs: child)

    def lookup(pid):
        raise measurement.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(measurement.psutil, "Process", lookup)
    with pytest.raises(measurement.psutil.NoSuchProcess):
        measurement.measure(tmp_path, "float32")
    assert events[0] == "kill"
    assert events[1][0] == "wait" and 0 <= events[1][1] <= 10


def test_cleanup_keeps_observed_children_after_launcher_exit(measurement, monkeypatch):
    events = []

    def disappeared(*args, **kwargs):
        raise measurement.psutil.NoSuchProcess(123)

    gone_member = SimpleNamespace(pid=124, kill=disappeared)
    live_member = SimpleNamespace(pid=125, kill=lambda: events.append("kill-descendant"))
    launcher = SimpleNamespace(children=disappeared)
    child = SimpleNamespace(poll=lambda: 0,
        kill=lambda: pytest.fail("launcher already exited"),
        wait=lambda timeout: events.append(("wait-launcher", timeout)))

    def wait_procs(members, timeout):
        assert {member.pid for member in members} == {124, 125}
        assert 0 <= timeout <= 10
        events.append("wait-descendants")
        return members, []

    monkeypatch.setattr(measurement.psutil, "wait_procs", wait_procs)
    measurement.stop_child(child, launcher, {124: gone_member, 125: live_member})
    assert events[0] == "kill-descendant"
    assert events[1][0] == "wait-launcher"
    assert events[2] == "wait-descendants"


def test_cleanup_reports_descendant_that_did_not_stop(measurement, monkeypatch):
    member = SimpleNamespace(pid=125, kill=lambda: None)
    child = SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0)
    monkeypatch.setattr(measurement.psutil, "wait_procs", lambda *args, **kwargs: ([], [member]))
    with pytest.raises(RuntimeError, match="child processes did not stop"):
        measurement.stop_child(child, None, {125: member})


def test_output_pipe_timeout_still_cleans_up(measurement, monkeypatch, tmp_path):
    events = []
    member = SimpleNamespace(pid=125, kill=lambda: events.append("kill-descendant"))
    process = SimpleNamespace(children=lambda recursive: [member],
        memory_info=lambda: SimpleNamespace(rss=10, private=20))
    member.memory_info = process.memory_info

    def communicate(timeout):
        assert timeout == 10
        raise measurement.subprocess.TimeoutExpired("synthetic-child", timeout)

    child = SimpleNamespace(pid=123, stdin=io.StringIO(), stdout=io.StringIO("READY\n"),
        poll=lambda: 0, wait=lambda timeout: 0, communicate=communicate)
    monkeypatch.setattr(measurement.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(measurement.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(measurement.psutil, "wait_procs", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(measurement, "reserve_ok", lambda minimum: True)
    with pytest.raises(measurement.subprocess.TimeoutExpired):
        measurement.measure(tmp_path, "float32")
    assert events == ["kill-descendant"]
