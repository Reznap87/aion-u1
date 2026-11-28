"""Build a deterministic, offline proposal for the next data-generation cycle."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ..cli import load_cases
from ..models import AuditCase
from ..provenance import sha256_file
from .compare import compare_reports
from .tasks import TaskBank, load_task_bank


def _prompt_key(prompt: str) -> str:
    return " ".join(prompt.split()).casefold()


def plan_tasks(
    comparison: dict[str, Any], bank: TaskBank, evaluation_cases: list[AuditCase],
    *, max_candidates: int, allowed_teachers: set[str],
) -> dict[str, Any]:
    """Select whole tasks without spending money, invoking teachers or approving data."""
    if type(max_candidates) is not int or max_candidates <= 0:
        raise ValueError("An explicit positive candidate limit is required")
    if not allowed_teachers or any(not alias.strip() for alias in allowed_teachers):
        raise ValueError("An explicit teacher allowlist is required")
    bank.task_map()
    priorities: dict[str, float] = {}
    for row in comparison["capabilities"]:
        if row["status"] != "training_needed":
            continue
        priority = row["priority"]
        if not isinstance(priority, (int, float)) or not math.isfinite(priority) or priority <= 0:
            raise ValueError("Invalid measured training priority")
        priorities[row["capability"]] = priority
    eval_ids = {case.id for case in evaluation_cases}
    eval_prompts = {_prompt_key(case.prompt) for case in evaluation_cases}
    selected = []
    skipped = []
    seen_prompts = set()
    remaining = max_candidates
    ordered = sorted(bank.tasks, key=lambda task: (-priorities.get(task.capability, 0), task.id))
    for task in ordered:
        aliases = sorted(set(task.teacher_aliases) & allowed_teachers)
        prompt_key = _prompt_key(task.prompt)
        cost = len(aliases) * task.samples_per_teacher
        if task.id in eval_ids or prompt_key in eval_prompts:
            reason = "evaluation_overlap"
        elif task.capability not in priorities:
            reason = "no_measured_deficit"
        elif task.verifier.type == "python_code":
            reason = "legacy_execution_verifier"
        elif not aliases:
            reason = "no_allowed_teacher"
        elif prompt_key in seen_prompts:
            reason = "duplicate_training_prompt"
        elif cost > remaining:
            reason = "candidate_limit"
        else:
            selected.append({
                "task_id": task.id, "capability": task.capability,
                "priority": priorities[task.capability],
                "teacher_aliases": aliases,
                "samples_per_teacher": task.samples_per_teacher,
                "planned_candidates": cost,
            })
            seen_prompts.add(prompt_key)
            remaining -= cost
            continue
        skipped.append({"task_id": task.id, "reason": reason})
    covered = {row["capability"] for row in selected}
    return {
        "schema_version": 1, "kind": "offline_training_proposal",
        "evaluation_commit": comparison["evaluation_commit"],
        "index_version": comparison["index_version"],
        "student_model_identity": comparison["student_model_identity"],
        "max_candidates": max_candidates, "planned_candidates": max_candidates - remaining,
        "allowed_teachers": sorted(allowed_teachers),
        "selected_tasks": selected, "skipped_tasks": skipped,
        "unserved_deficits": sorted(set(priorities) - covered),
        "execution_authorized": False, "training_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "student", "suite", "capabilities", "tasks", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--allow-teacher", action="append", required=True)
    args = parser.parse_args(argv)
    paths = {name: getattr(args, name) for name in ("teacher", "student", "suite", "capabilities", "tasks")}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if any(value is None for value in hashes.values()):
        raise ValueError("All planning inputs must be readable")
    comparison = compare_reports(
        json.loads(args.teacher.read_text(encoding="utf-8")),
        json.loads(args.student.read_text(encoding="utf-8")),
        args.suite, args.capabilities,
    )
    result = plan_tasks(comparison, load_task_bank(args.tasks), load_cases(args.suite),
                        max_candidates=args.max_candidates, allowed_teachers=set(args.allow_teacher))
    if hashes != {name: sha256_file(path) for name, path in paths.items()}:
        raise RuntimeError("Planning inputs changed during execution")
    result["input_hashes"] = hashes
    if args.output.resolve() in {path.resolve() for path in paths.values()}:
        raise ValueError("Output must not overwrite an input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
