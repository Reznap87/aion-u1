import importlib.util
import json
import subprocess
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evaluations/comparison_v2_independent"
spec = importlib.util.spec_from_file_location("comparison_score_answers", FOLDER / "score_answers.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def cases():
    return [json.loads(line) for line in (FOLDER / "holdout.jsonl").read_text(encoding="utf-8").splitlines()]


def write_answers(path, answers):
    path.write_text("".join(json.dumps(row) + "\n" for row in answers), encoding="utf-8")


def test_cli_scores_complete_synthetic_answers_and_preserves_report(tmp_path):
    records = [{"id": c["id"], "response": json.dumps({"groesser": c["evaluation"]["left"]})} for c in cases()]
    answers, report = tmp_path / "answers.jsonl", tmp_path / "report.json"
    write_answers(answers, records)
    result = subprocess.run([sys.executable, str(FOLDER / "score_answers.py"), "--answers", str(answers), "--output", str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert (data["passed"], data["cases"]) == (12, 18)
    assert data["model_execution_verified"] is data["model_identity_verified"] is False
    with pytest.raises(FileExistsError): evaluation.evaluate(answers, report)
    assert json.loads(report.read_text(encoding="utf-8")) == data


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "duplicate_key"])
def test_invalid_response_ids_fail_closed_without_report(tmp_path, problem):
    records = [{"id": c["id"], "response": "{}"} for c in cases()]
    answers, report = tmp_path / "answers.jsonl", tmp_path / "report.json"
    if problem == "missing": records.pop()
    if problem == "duplicate": records[-1] = records[0]
    if problem == "unknown": records[-1]["id"] = "unknown"
    write_answers(answers, records)
    if problem == "duplicate_key":
        with answers.open("a", encoding="utf-8") as out: out.write('{"id":"x","id":"y","response":"{}"}\n')
    with pytest.raises(ValueError): evaluation.evaluate(answers, report)
    assert not report.exists()


def test_changed_eval_data_is_rejected(tmp_path, monkeypatch):
    folder = tmp_path / "eval"
    folder.mkdir()
    (folder / "holdout.jsonl").write_bytes((FOLDER / "holdout.jsonl").read_bytes() + b" ")
    monkeypatch.setattr(evaluation, "DIRECTORY", folder)
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluation.evaluate(tmp_path / "missing", tmp_path / "report")
