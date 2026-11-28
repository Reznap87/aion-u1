"""Explicitly authorized execution of a revalidated offline training proposal."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from ..cli import load_cases
from ..models import TeacherConfig
from ..privacy import scan_text
from ..provenance import sha256_file
from ..teachers.free_only import FreeOnlyClient, MODEL_PATTERN
from ..teachers.resolver import TeacherResolution
from .compare import compare_reports
from .generate import generate_one
from .plan import plan_tasks
from .run_ledger import RunLedger
from .tasks import TaskBank, TaskSpec, load_task_bank


def prepare_jobs(
    proposal: dict[str, Any], bank: TaskBank, models: dict[str, str],
) -> list[tuple[TaskSpec, TeacherResolution, int]]:
    """Preflight every job before any provider request."""
    tasks = bank.task_map()
    jobs = []
    for entry in proposal["selected_tasks"]:
        task = tasks[entry["task_id"]]
        if task.modality != "text" or scan_text(task.prompt):
            raise ValueError("Only privacy-screened text tasks are supported")
        if task.max_tokens is None:
            raise ValueError("Explicit token budget required")
        for alias in entry["teacher_aliases"]:
            model = models.get(alias, "")
            if not re.fullmatch(MODEL_PATTERN, model):
                raise ValueError("Selected teacher requires an explicit free model mapping")
            config = TeacherConfig(alias=alias, display_name=alias, provider="openrouter",
                                   candidate_model_ids=[model], license="unreviewed")
            resolution = TeacherResolution(config, model, "configured", "explicit_free_mapping")
            for index in range(entry["samples_per_teacher"]):
                jobs.append((task, resolution, index))
    if len(jobs) != proposal["planned_candidates"] or len(jobs) > proposal["max_candidates"]:
        raise ValueError("Candidate count mismatch")
    return jobs


async def execute_jobs(client: FreeOnlyClient, jobs, root: Path, output: Path) -> int:
    """Serial execution: stop on the first error, preserve only unapproved candidates."""
    count = 0
    semaphore = asyncio.Semaphore(1)
    with output.open("x", encoding="utf-8") as stream:
        for task, resolution, index in jobs:
            row, error = await generate_one(client, resolution, task, index, semaphore, root)
            if error is not None or row is None:
                raise RuntimeError("Free-only generation stopped; provider or generation error")
            if not isinstance(row.get("response"), str) or scan_text(row["response"]):
                raise RuntimeError("Free-only generation stopped; privacy check failed")
            row["provenance"]["training_approved"] = False
            row["provenance"]["license"] = "unreviewed"
            row["provenance"]["admission"] = "free_only"
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "teacher", "student", "suite", "capabilities", "tasks", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="alias=provider/model:free")
    parser.add_argument("--ledger", type=Path, help="Persistent SQLite ledger shared across all workers")
    parser.add_argument("--execute-free", action="store_true", help="Explicitly authorize free-only inference")
    args = parser.parse_args(argv)
    paths = {name: getattr(args, name) for name in ("teacher", "student", "suite", "capabilities", "tasks")}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    proposal = json.loads(args.plan.read_text(encoding="utf-8"))
    if any(value is None for value in hashes.values()) or proposal.get("input_hashes") != hashes:
        raise ValueError("Planning input hashes differ")
    comparison = compare_reports(json.loads(args.teacher.read_text(encoding="utf-8")),
                                 json.loads(args.student.read_text(encoding="utf-8")),
                                 args.suite, args.capabilities)
    bank = load_task_bank(args.tasks)
    expected = plan_tasks(comparison, bank, load_cases(args.suite),
                          max_candidates=proposal["max_candidates"],
                          allowed_teachers=set(proposal["allowed_teachers"]))
    expected["input_hashes"] = hashes
    if proposal != expected or hashes != {name: sha256_file(path) for name, path in paths.items()}:
        raise ValueError("Proposal failed deterministic revalidation")
    models = {}
    for entry in args.model:
        alias, sep, model = entry.partition("=")
        if not sep or alias in models:
            raise ValueError("Invalid or duplicate model mapping")
        models[alias] = model
    jobs = prepare_jobs(expected, bank, models)
    if not args.execute_free:
        print(json.dumps({"planned_requests": len(jobs), "executed": 0, "training_approved": False}))
        return 0
    if args.ledger is None:
        raise ValueError("Persistent execution ledger required")
    if args.ledger.resolve() in {args.output.resolve(), args.plan.resolve(), *(path.resolve() for path in paths.values())}:
        raise ValueError("Ledger must be separate from inputs and output")
    if not jobs:
        raise ValueError("No eligible jobs")
    key = os.getenv("AION_OPENROUTER_FREE_API_KEY", "")
    if not key:
        raise ValueError("Dedicated free-only API key missing")
    if args.output.exists():
        raise ValueError("Output already exists; refusing overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    async def run() -> int:
        client = FreeOnlyClient(key)
        try:
            await client.require_zero_limit()
            ledger = RunLedger(args.ledger)
            identifier = ledger.reserve(expected)
            try:
                count = await execute_jobs(client, jobs, args.tasks.parent, args.output)
            except BaseException:
                ledger.finish(identifier, success=False)
                raise
            ledger.finish(identifier, success=True)
            return count
        finally:
            await client.close()

    print(json.dumps({"generated": asyncio.run(run()), "training_approved": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
