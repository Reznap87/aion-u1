from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aion_u1.cli import build_parser, command_dataset
from aion_u1.dataset import build_dataset, write_jsonl
from aion_u1.models import AuditReport, CaseResult
from aion_u1.training.approve import main as approve_main
from aion_u1.training.data import (
    TrainingDataError, read_jsonl, select_canonical,
)


def _result(
    case_id: str, *, passed: bool, error: str | None = None
) -> CaseResult:
    return CaseResult(
        teacher_alias="tt",
        teacher_model_id="provider/model",
        case_id=case_id,
        capability="cap",
        prompt="42?",
        response="42" if passed else "",
        score=1.0 if passed else None,
        passed=passed,
        verifier_type="exact_match",
        latency_ms=1,
        error=error,
    )


def _report() -> AuditReport:
    return AuditReport(
        generated_at=datetime.now(timezone.utc),
        suite="smoke",
        suite_path="smoke.yaml",
        teacher_resolutions=[],
        results=[
            _result("passed1", passed=True),
            _result("passed2", passed=True),
            _result("failed", passed=False, error="Timeout"),
        ],
        scores={},
    )


def test_smoke_export_only_creates_pending_rows() -> None:
    candidates, pending = build_dataset(_report())
    assert len(candidates) == 3
    assert len(pending) == 2
    assert {
        row["verification"]["status"] for row in pending
    } == {"pending_review"}
    assert all(
        row["provenance"].get("training_approved") is not True
        for row in pending
    )
    assert select_canonical(pending, require_approved=True) == []


def test_cli_dataset_writes_pending_not_verified(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        _report().model_dump_json(), encoding="utf-8"
    )
    args = build_parser().parse_args([
        "dataset",
        "--report", str(report),
        "--output", str(tmp_path / "candidates.jsonl"),
        "--pending-output", str(tmp_path / "generated" / "pending.jsonl"),
    ])
    assert command_dataset(args) == 0
    pending = read_jsonl(tmp_path / "generated" / "pending.jsonl")
    assert len(pending) == 2
    assert not (tmp_path / "verified").exists()


def test_approve_promotes_only_selected_rows(tmp_path: Path) -> None:
    _, rows = build_dataset(_report())
    source = tmp_path / "generated" / "pending.jsonl"
    approved_path = tmp_path / "verified" / "approved.jsonl"
    ids_file = tmp_path / "approved_ids.txt"
    write_jsonl(rows, source)
    ids_file.write_text(rows[0]["id"] + "\n", encoding="utf-8")

    assert approve_main([
        "--input", str(source),
        "--output", str(approved_path),
        "--license", "reviewed-test-license",
        "--operator", "test-reviewer",
        "--ids-file", str(ids_file),
    ]) == 0

    approved = read_jsonl(approved_path)
    assert len(approved) == 1
    assert approved[0]["id"] == rows[0]["id"]
    assert approved[0]["verification"]["status"] == "verified"
    assert approved[0]["provenance"]["training_approved"] is True
    approval = approved[0]["provenance"]["approval"]
    assert approval["operator"] == "test-reviewer"
    assert approval["previous_verification_status"] == "pending_review"
    assert approval["approved_at"]
    assert len(select_canonical(approved, require_approved=True)) == 1

    original = read_jsonl(source)
    assert all(
        row["verification"]["status"] == "pending_review"
        for row in original
    )


def test_approve_rejects_unknown_ids_without_export(tmp_path: Path) -> None:
    _, rows = build_dataset(_report())
    source = tmp_path / "pending.jsonl"
    target = tmp_path / "verified" / "approved.jsonl"
    ids_file = tmp_path / "approved_ids.txt"
    write_jsonl(rows, source)
    ids_file.write_text("nonexistent-id\n", encoding="utf-8")

    with pytest.raises(TrainingDataError, match="Unbekannte IDs"):
        approve_main([
            "--input", str(source),
            "--output", str(target),
            "--license", "reviewed-test-license",
            "--operator", "test-reviewer",
            "--ids-file", str(ids_file),
        ])
    assert not target.exists()


def test_approve_rejects_unverified_source(tmp_path: Path) -> None:
    _, rows = build_dataset(_report())
    rows[0]["verification"]["status"] = "unverified"
    source = tmp_path / "pending.jsonl"
    target = tmp_path / "verified" / "approved.jsonl"
    ids_file = tmp_path / "approved_ids.txt"
    write_jsonl(rows, source)
    ids_file.write_text(rows[0]["id"] + "\n", encoding="utf-8")
    with pytest.raises(TrainingDataError, match="nicht für eine Freigabe"):
        approve_main([
            "--input", str(source),
            "--output", str(target),
            "--license", "reviewed-test-license",
            "--operator", "test-reviewer",
            "--ids-file", str(ids_file),
        ])
    assert not target.exists()
