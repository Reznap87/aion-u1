from __future__ import annotations

import json
from pathlib import Path

import pytest

from aion_u1.cli import build_parser, command_dataset
from aion_u1.models import AuditReport, CaseResult
from aion_u1.privacy import (
    KIND_PII, KIND_SECRET, mask, scan_row, scan_text, summarize,
)
from aion_u1.training.approve import main as approve_main
from aion_u1.training.data import write_jsonl


def test_mask_verkuertzt_treffer() -> None:
    assert mask("abcd") == "****"
    assert mask("sk-or-v1-1234567890abcdef") == "sk-or-…cdef"
    assert "1234567890" not in mask("sk-or-v1-1234567890abcdef")


def test_scan_text_erkennt_secrets() -> None:
    text = (
        "key=sk-or-v1-0123456789abcdef0123456789abcdef "
        "ghp_01234567890123456789012345678901234 "
        "Bearer abcdefghijklmnopqrstuvwxyz123456"
    )

    labels = {f.label for f in scan_text(text)}

    assert "openrouter_key" in labels
    assert "github_token" in labels
    assert "bearer_token" in labels


def test_scan_text_erkennt_pii() -> None:
    findings = scan_text("Kontakt: max@example.com, IBAN DE89370400440532013000")

    assert {f.kind for f in findings} == {KIND_PII}
    assert {f.label for f in findings} == {"email", "iban"}


def test_scan_text_normale_prosa_ohne_fund() -> None:
    assert scan_text("Bitte schreibe eine Funktion add(a, b).") == []
    assert scan_text('{"name": "AION", "version": 1}') == []


def test_scan_row_nennt_das_feld() -> None:
    findings = scan_row(
        {"prompt": "sk-or-v1-0123456789abcdef0123456789abcdef", "response": ""},
    )

    assert len(findings) == 1
    assert findings[0].field == "prompt"


def test_summarize_zaehlt_kinds() -> None:
    counts = summarize(scan_text("a@example.com sk-or-v1-0123456789abcdef"))

    assert counts == {KIND_SECRET: 1, KIND_PII: 1}


def _row(row_id: str, response: str) -> dict:
    return {
        "id": row_id,
        "task_id": "code.add",
        "group_id": "code.add",
        "prompt": "Schreibe add(a, b).",
        "response": response,
        "capabilities": ["code.synthesis"],
        "teacher_alias": "tt",
        "provenance": {"license": "unreviewed", "training_approved": False},
        "verification": {"status": "pending_review", "score": 1.0},
    }


def _approve_args(source: Path, target: Path, *extra: str) -> list[str]:
    return [
        "--input", str(source),
        "--output", str(target),
        "--license", "reviewed-test-license",
        "--operator", "test-reviewer",
        "--confirm-all",
        *extra,
    ]


def test_approve_sperrt_secrets(tmp_path: Path, capsys) -> None:
    source = tmp_path / "pending.jsonl"
    target = tmp_path / "verified" / "approved.jsonl"
    write_jsonl(
        [
            _row("r1", "def add(a, b):\n    return a + b"),
            _row("r2", "Der Key lautet sk-or-v1-0123456789abcdef"),
        ],
        source,
    )

    assert approve_main(_approve_args(source, target)) == 1

    err = capsys.readouterr().err
    assert "unvertraegliche Inhalte" in err
    assert "sk-or-…cdef" in err
    assert "0123456789abcdef" not in err
    assert not target.exists()


def test_approve_zaehlt_pii_und_sperrt_auf_wunsch(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    source = tmp_path / "pending.jsonl"
    target = tmp_path / "verified" / "approved.jsonl"
    write_jsonl(
        [_row("r1", "Schreib eine Mail an max@example.com")],
        source,
    )

    assert approve_main(_approve_args(source, target)) == 0
    assert json.loads(capsys.readouterr().out)["privacy"] == {
        KIND_SECRET: 0, KIND_PII: 1,
    }

    capsys.readouterr()
    target.unlink()
    assert approve_main(_approve_args(source, target, "--block-pii")) == 1
    assert "unvertraegliche Inhalte" in capsys.readouterr().err
    assert not target.exists()


def test_dataset_zehlt_privacy(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    report = AuditReport(
        generated_at="2026-11-17T00:00:00+00:00",
        suite="smoke",
        suite_path="smoke.yaml",
        teacher_resolutions=[],
        results=[
            CaseResult(
                teacher_alias="tt",
                teacher_model_id="provider/model",
                case_id="c1",
                capability="cap",
                prompt="p",
                response="sk-or-v1-0123456789abcdef",
                score=1.0,
                passed=True,
                verifier_type="exact_match",
                latency_ms=1,
            )
        ],
        scores={},
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")

    args = build_parser().parse_args([
        "dataset",
        "--report", str(report_path),
        "--output", str(tmp_path / "candidates.jsonl"),
        "--pending-output", str(tmp_path / "pending.jsonl"),
    ])
    assert command_dataset(args) == 0

    out = json.loads(capsys.readouterr().out)
    assert out["privacy"] == {KIND_SECRET: 1, KIND_PII: 0}
