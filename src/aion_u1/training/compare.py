"""Offline comparison of teacher and student evaluations."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..capabilities import CapabilityRegistry
from ..cli import load_cases
from ..models import AuditCase, CaseResult
from ..provenance import sha256_file


def _provenance(report: dict[str, Any], key: str) -> str:
    provenance = report.get("provenance") or {}
    if key == "git_commit":
        value = provenance.get(key)
        pattern = r"[0-9a-f]{40}"
    else:
        value = provenance.get("inputs", {}).get(key, {}).get("sha256")
        pattern = r"[0-9a-f]{64}"
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"Missing or invalid provenance: {key}")
    return value


def _groups(report: dict[str, Any], cases: dict[str, AuditCase]):
    groups: dict[str, dict[str, CaseResult]] = defaultdict(dict)
    for raw in report["results"]:
        row = CaseResult.model_validate(raw)
        case = cases.get(row.case_id)
        if case is None or (row.capability, row.prompt, row.verifier_type) != (
            case.capability, case.prompt, case.verifier.type
        ):
            raise ValueError("Report case does not match suite")
        group = groups[row.teacher_alias]
        if row.case_id in group:
            raise ValueError("Duplicate report case")
        group[row.case_id] = row
    for group in groups.values():
        if len({row.teacher_model_id for row in group.values()}) != 1:
            raise ValueError("Mixed model identities within one alias")
    return groups


def _score(rows: dict[str, CaseResult], ids: list[str], token_budget: int) -> float | None:
    values = []
    if not ids:
        return None
    for case_id in ids:
        row = rows.get(case_id)
        if row is None or row.error is not None or row.finish_reason == "length" or row.score is None:
            return None
        if row.max_tokens != token_budget:
            return None
        values.append(row.score)
    return sum(values) / len(values)


def compare_reports(
    teacher: dict[str, Any], student: dict[str, Any],
    suite_path: Path, capability_path: Path,
) -> dict[str, Any]:
    """Require matching evaluation provenance; never replace missing evidence with zero."""
    cases = load_cases(suite_path)
    by_id = {case.id: case for case in cases}
    if not cases or len(by_id) != len(cases):
        raise ValueError("Empty suite or duplicate case identifiers")
    registry = CapabilityRegistry.load(capability_path)
    for case in cases:
        registry.get(case.capability)
    version = "2" if any(case.verifier.engine == "v2" for case in cases) else "1"
    for report in (teacher, student):
        if report.get("index_version", "1") != version:
            raise ValueError("Index version mismatch")
        for key, path in (("suite", suite_path), ("capabilities", capability_path)):
            if _provenance(report, key) != sha256_file(path):
                raise ValueError(f"Input hash mismatch: {key}")
    commit = _provenance(teacher, "git_commit")
    if commit != _provenance(student, "git_commit"):
        raise ValueError("Evaluation code revision mismatch")
    image = teacher.get("verifier_image")
    if image != student.get("verifier_image"):
        raise ValueError("Verifier image mismatch")
    if any(case.verifier.type == "python_function" for case in cases):
        if not isinstance(image, str) or not re.fullmatch(r"[A-Za-z0-9_./:-]+@sha256:[a-f0-9]{64}", image):
            raise ValueError("Functional verifier image provenance required")
    teachers = _groups(teacher, by_id)
    students = _groups(student, by_id)
    if not teachers or len(students) != 1:
        raise ValueError("Expected at least one teacher and exactly one student")
    student_alias, student_rows = next(iter(students.items()))
    if set(student_rows) != set(by_id) or any(set(rows) != set(by_id) for rows in teachers.values()):
        raise ValueError("Report case sets must equal the entire suite")
    token_budget = student.get("max_tokens")
    if type(token_budget) is not int or token_budget <= 0:
        raise ValueError("Student token budget required")
    if teacher.get("requested_max_tokens") != token_budget:
        raise ValueError("Requested token budgets differ or are missing")
    output = []
    for spec in registry.list():
        ids = [case.id for case in cases if case.capability == spec.id]
        scores = {alias: _score(rows, ids, token_budget) for alias, rows in teachers.items()}
        measured = [(score, alias) for alias, score in scores.items() if score is not None]
        best, best_alias = max(measured) if measured else (None, None)
        current = _score(student_rows, ids, token_budget)
        target = 0.95 * best if best is not None and best > 0 else None
        deficit = max(0.0, target - current) if target is not None and current is not None else None
        output.append({
            "capability": spec.id,
            "weight": spec.weight,
            "case_count": len(ids),
            "teacher_scores": scores,
            "best_teacher": best_alias,
            "best_teacher_score": best,
            "student_score": current,
            "target_score": target,
            "gap_to_target": deficit,
            "priority": deficit * spec.weight if deficit is not None else None,
            "status": "unmeasured" if deficit is None else "target_met" if deficit == 0 else "training_needed",
        })
    output.sort(key=lambda row: (row["priority"] is None, -(row["priority"] or 0), row["capability"]))
    return {
        "schema_version": 1,
        "index_version": version,
        "evaluation_commit": commit,
        "suite_sha256": sha256_file(suite_path),
        "capabilities_sha256": sha256_file(capability_path),
        "verifier_image": image,
        "student_alias": student_alias,
        "student_base_model": student.get("base_model"),
        "student_adapter": student.get("adapter"),
        "target_ratio": 0.95,
        "max_tokens": token_budget,
        "student_model_identity": student.get("model_identity"),
        "capabilities": output,
        "all_targets_met": all(row["status"] == "target_met" for row in output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "student", "suite", "capabilities", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    report = compare_reports(
        json.loads(args.teacher.read_text(encoding="utf-8")),
        json.loads(args.student.read_text(encoding="utf-8")),
        args.suite, args.capabilities,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
