import hashlib
from pathlib import Path

from aion_u1.training.approve import main as approve_main
from aion_u1.training.data import read_jsonl, write_jsonl
from aion_u1.training.ranking import rank_candidates
from aion_u1.training.tasks import TaskBank


def _bank() -> TaskBank:
    return TaskBank.model_validate(
        {
            "version": 1,
            "tasks": [
                {
                    "id": "task.ok",
                    "capability": "code.synthesis",
                    "prompt": "Antworte mit ok.",
                    "verifier": {
                        "type": "contains_all",
                        "values": ["ok"],
                    },
                    "teacher_aliases": ["t"],
                },
                {
                    "id": "task.fail",
                    "capability": "reasoning.logical",
                    "prompt": "Antworte mit ok.",
                    "verifier": {
                        "type": "contains_all",
                        "values": ["ok"],
                    },
                    "teacher_aliases": ["t"],
                },
            ],
        }
    )


def _row(
    task_id: str,
    alias: str,
    response: str,
    *,
    latency_ms: int = 10,
) -> dict:
    digest = hashlib.sha256(
        f"{task_id}|{alias}|{response}".encode("utf-8")
    ).hexdigest()[:24]

    return {
        "id": f"aion-candidate-{digest}",
        "task_id": task_id,
        "prompt": "Antworte mit ok.",
        "response": response,
        "teacher_alias": alias,
        "latency_ms": latency_ms,
        "capabilities": ["code.synthesis"],
        "provenance": {
            "teacher_alias": alias,
            "teacher_model_id": f"provider/{alias}",
        },
    }


def _approve(
    source: Path,
    target: Path,
    *extra: str,
) -> int:
    return approve_main(
        [
            "--input",
            str(source),
            "--output",
            str(target),
            "--license",
            "reviewed-test-license",
            "--operator",
            "test-reviewer",
            *extra,
        ]
    )


def test_best_contains_only_verified_candidates() -> None:
    rows = [
        _row("task.ok", "a", "ok"),
        _row("task.ok", "b", "bad"),
        _row("task.fail", "a", "bad"),
        _row("task.fail", "b", "bad"),
    ]

    _dedup, best, _preferences, report = rank_candidates(
        rows,
        _bank(),
    )

    assert [row["task_id"] for row in best] == ["task.ok"]
    assert all(
        row["verification"]["status"] == "verified"
        for row in best
    )
    assert report["best_candidates"] == 1
    assert report["tasks_without_verified_candidate"] == 1
    assert report[
        "tasks_without_verified_candidate_ids"
    ] == ["task.fail"]


def test_approve_accepts_best_candidates(tmp_path: Path) -> None:
    rows = [_row("task.ok", "a", "ok")]
    _dedup, best, _preferences, _report = rank_candidates(
        rows,
        _bank(),
    )

    source = tmp_path / "best_candidates.jsonl"
    target = tmp_path / "verified" / "sft_canonical.jsonl"
    write_jsonl(best, source)

    assert _approve(source, target, "--confirm-all") == 0

    approved = read_jsonl(target)

    assert len(approved) == 1
    assert approved[0]["verification"]["status"] == "verified"
    assert (
        approved[0]["provenance"]["training_approved"]
        is True
    )


def test_approve_refuses_rejected_rows(tmp_path: Path) -> None:
    rows = [_row("task.ok", "a", "bad")]
    dedup, _best, _preferences, _report = rank_candidates(
        rows,
        _bank(),
    )

    assert dedup[0]["verification"]["status"] == "rejected"

    source = tmp_path / "scored_candidates.jsonl"
    target = tmp_path / "verified" / "sft_canonical.jsonl"
    write_jsonl(dedup, source)

    assert _approve(source, target, "--confirm-all") == 1
    assert not target.exists()
