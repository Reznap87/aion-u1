from concurrent.futures import ThreadPoolExecutor

import pytest

from aion_u1.training.run_ledger import RunLedger, plan_id


PLAN = {"planned_candidates": 2, "input_hashes": {"tasks": "a" * 64}}


@pytest.mark.parametrize("state", ["reserved", "completed", "failed"])
def test_persistent_replay_denied(tmp_path, state):
    path = tmp_path / "runs.sqlite"
    ledger = RunLedger(path)
    identifier = ledger.reserve(PLAN)
    if state != "reserved":
        ledger.finish(identifier, success=state == "completed")
    with pytest.raises(RuntimeError, match="replay denied"):
        RunLedger(path).reserve(PLAN)


def test_concurrent_reservation(tmp_path):
    path = tmp_path / "runs.sqlite"
    RunLedger(path)

    def reserve():
        try:
            RunLedger(path).reserve(PLAN)
            return True
        except RuntimeError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(lambda _: reserve(), range(4))) == 1


def test_stable_identifier():
    assert plan_id(PLAN) == plan_id(dict(reversed(list(PLAN.items()))))
    assert plan_id(PLAN) != plan_id(dict(PLAN, planned_candidates=3))


def test_finalize_twice_rejected(tmp_path):
    ledger = RunLedger(tmp_path / "runs.sqlite")
    identifier = ledger.reserve(PLAN)
    ledger.finish(identifier, success=True)
    with pytest.raises(RuntimeError, match="finalized"):
        ledger.finish(identifier, success=False)
