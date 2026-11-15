from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from ..models import TeacherConfig
from ..teachers.base import TeacherError, TeacherHTTPError
from ..teachers.openrouter import OpenRouterClient
from ..teachers.resolver import (
    TeacherResolution,
    resolve_teachers,
)
from .checkpoint import CandidateCheckpoint
from .tasks import (
    TaskBank,
    TaskSpec,
    build_task_input,
    find_project_root,
    load_task_bank,
)


def load_teacher_configs(
    root: Path,
) -> list[TeacherConfig]:
    raw = yaml.safe_load(
        (root / "configs" / "teachers.yaml").read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(raw, dict):
        raise ValueError(
            "teachers.yaml muss ein YAML-Mapping sein."
        )

    entries = raw.get("teachers")

    if not isinstance(entries, list):
        raise ValueError(
            "teachers muss eine Liste sein."
        )

    return [
        TeacherConfig.model_validate(entry)
        for entry in entries
    ]


def resolution_to_dict(
    resolution: TeacherResolution,
) -> dict[str, Any]:
    return {
        "alias": resolution.config.alias,
        "display_name": resolution.config.display_name,
        "provider": resolution.config.provider,
        "model_id": resolution.model_id,
        "status": resolution.status,
        "source": resolution.source,
        "suggestions": list(resolution.suggestions),
    }


def candidate_id(
    task_id: str,
    teacher_alias: str,
    model_id: str,
    sample_index: int,
    response: str,
) -> str:
    digest = hashlib.sha256(
        (
            f"{task_id}|{teacher_alias}|{model_id}|"
            f"{sample_index}|{response}"
        ).encode("utf-8")
    ).hexdigest()

    return f"aion-candidate-{digest[:24]}"


def request_fingerprint(task: TaskSpec, model_id: str, sample: int, content: Any) -> str:
    payload = {"task": task.model_dump(mode="json"), "model": model_id,
               "sample": sample, "input": content}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def resumed_keys(rows, bank: TaskBank, root: Path):
    tasks = bank.task_map()
    seen = set()
    for row in rows:
        cost = row.get("cost_usd")
        if cost is not None and (
            type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
        ):
            raise ValueError("Checkpoint enthält ungültige Kostenangaben.")
        task = tasks.get(row.get("task_id"))
        alias = row.get("teacher_alias")
        model = row.get("teacher_model_id")
        sample = row.get("teacher_sample_index")
        if (
            task is None or alias not in task.teacher_aliases
            or not isinstance(model, str) or not model
            or type(sample) is not int or not 0 <= sample < task.samples_per_teacher
            or not isinstance(row.get("response"), str)
        ):
            raise ValueError("Checkpoint passt nicht zur Aufgabenbank.")
        content = build_task_input(task, root)
        if (
            row.get("prompt") != task.prompt or row.get("input_content") != content
            or row.get("capabilities") != [task.capability]
            or row.get("id") != candidate_id(task.id, alias, model, sample, row["response"])
            or (row.get("request_fingerprint") is not None and
                row["request_fingerprint"] != request_fingerprint(task, model, sample, content))
        ):
            raise ValueError("Checkpoint-Inhalt oder Anfragekonfiguration wurde verändert.")
        key = (task.id, alias, model, sample)
        if key in seen:
            raise ValueError("Checkpoint enthält doppelte Aufträge.")
        seen.add(key)
    return seen


async def generate_one(
    client: OpenRouterClient,
    resolution: TeacherResolution,
    task: TaskSpec,
    sample_index: int,
    semaphore: asyncio.Semaphore,
    root: Path,
    stop: asyncio.Event | None = None,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    async with semaphore:
        if stop is not None and stop.is_set():
            return None, {
                "task_id": task.id, "teacher_alias": resolution.config.alias,
                "teacher_model_id": resolution.model_id, "sample_index": sample_index,
                "skipped": True, "error_code": "run_stopped",
            }
        try:
            content = build_task_input(task, root)

            response = await client.generate(
                resolution.model_id or "",
                [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                temperature=task.temperature,
                max_tokens=task.max_tokens,
            )

            generated_at = datetime.now(
                timezone.utc
            ).isoformat()

            row = {
                "id": candidate_id(
                    task.id,
                    resolution.config.alias,
                    response.model_id,
                    sample_index,
                    response.content,
                ),
                "task_id": task.id,
                "group_id": task.id,
                "case_id": task.id,
                "prompt": task.prompt,
                "input_content": content,
                "contains_media": task.modality == "image",
                "response": response.content,
                "capabilities": [task.capability],
                "teacher_alias": resolution.config.alias,
                "teacher_model_id": response.model_id,
                "teacher_sample_index": sample_index,
                "latency_ms": response.latency_ms,
                "usage": response.usage,
                "cost_usd": response.cost_usd,
                "request_fingerprint": request_fingerprint(
                    task, resolution.model_id or "", sample_index, content
                ),
                "finish_reason": response.finish_reason,
                "provenance": {
                    "source": "teacher_generated",
                    "license": "unreviewed",
                    "training_approved": False,
                    "teacher_alias": resolution.config.alias,
                    "teacher_model_id": response.model_id,
                    "provider": resolution.config.provider,
                    "generated_at": generated_at,
                    "contains_media": task.modality == "image",
                },
                "verification": {
                    "status": "pending",
                    "score": 0.0,
                    "method": task.verifier.type,
                },
            }

            return row, None

        except Exception as exc:
            status = exc.status_code if isinstance(exc, TeacherHTTPError) else None
            if status in {401, 402, 403} and stop is not None:
                stop.set()
            error = {
                "task_id": task.id,
                "teacher_alias": resolution.config.alias,
                "teacher_model_id": resolution.model_id,
                "sample_index": sample_index,
                "error": f"Teacher HTTP {status}" if status else "Teacher request failed",
                "http_status": status,
                "error_code": "payment_required" if status == 402 else "teacher_request_failed",
            }

            return None, error


async def generate_candidates(
    configs: list[TeacherConfig],
    bank: TaskBank,
    *,
    root: Path,
    concurrency: int,
    checkpoint: CandidateCheckpoint | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not configs:
        raise ValueError(
            "Keine Teacher sind konfiguriert."
        )
    if not 1 <= concurrency <= 32:
        raise ValueError("concurrency muss zwischen 1 und 32 liegen.")
    if len({config.alias for config in configs}) != len(configs):
        raise ValueError("Doppelte Teacher-Aliase.")
    existing = list(checkpoint.rows) if checkpoint else []
    seen = resumed_keys(existing, bank, root)
    for task in bank.tasks:
        if len(set(task.teacher_aliases)) != len(task.teacher_aliases):
            raise ValueError("Doppelte Teacher-Aliase in einer Aufgabe.")
        build_task_input(task, root)

    api_key = os.getenv(
        configs[0].api_key_env,
        "",
    ).strip()

    if not api_key:
        raise ValueError(
            f"API-Key {configs[0].api_key_env} fehlt."
        )

    base_url = os.getenv(
        configs[0].base_url_env,
        configs[0].base_url_default,
    )

    client = OpenRouterClient(
        api_key=api_key,
        base_url=base_url,
    )

    try:
        resolutions = await resolve_teachers(
            client,
            configs,
        )

        resolution_by_alias = {
            resolution.config.alias: resolution
            for resolution in resolutions
        }
        for task_id, alias, model_id, sample in seen:
            resolution = resolution_by_alias.get(alias)
            if (
                resolution is None or resolution.status != "verified"
                or resolution.model_id != model_id
            ):
                raise ValueError("Checkpoint-Modell stimmt nicht mit dem verifizierten Teacher überein.")

        semaphore = asyncio.Semaphore(
            max(1, concurrency)
        )
        stop = asyncio.Event()
        errors: list[dict[str, Any]] = []

        jobs: list[
            asyncio.Task[
                tuple[
                    dict[str, Any] | None,
                    dict[str, Any] | None,
                ]
            ]
        ] = []

        for task in bank.tasks:
            for alias in task.teacher_aliases:
                resolution = resolution_by_alias.get(alias)

                if (
                    resolution is None
                    or resolution.status != "verified"
                    or not resolution.model_id
                ):
                    errors.append({
                        "task_id": task.id, "teacher_alias": alias,
                        "error_code": "teacher_unverified",
                        "error": "Teacher is not verified in the current catalog",
                    })
                    continue

                for sample_index in range(
                    task.samples_per_teacher
                ):
                    if (task.id, alias, resolution.model_id, sample_index) in seen:
                        continue
                    jobs.append(
                        asyncio.create_task(
                            generate_one(
                                client,
                                resolution,
                                task,
                                sample_index,
                                semaphore,
                                root,
                                stop,
                            )
                        )
                    )

        rows: list[dict[str, Any]] = existing
        try:
            for outcome in asyncio.as_completed(jobs):
                row, error = await outcome
                if row is not None:
                    if checkpoint:
                        checkpoint.append(row)
                    rows.append(row)
                if error is not None:
                    errors.append(error)
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)

        resolution_report = [
            resolution_to_dict(resolution)
            for resolution in resolutions
        ]

        return rows, errors, resolution_report

    finally:
        await client.close()


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aion-u1-generate-candidates",
        description=(
            "Erzeugt Multi-Teacher-Kandidaten für AION-U1."
        ),
    )

    parser.add_argument(
        "--tasks",
        default=(
            "training/configs/tasks/pilot.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "data/generated/teacher_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        default="reports/data-generation.json",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Vorhandene Kandidaten erhalten und nur fehlende Antworten erzeugen",
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    try:
        root = find_project_root()

        tasks_path = Path(args.tasks)
        output_path = Path(args.output)
        report_path = Path(args.report)

        if not tasks_path.is_absolute():
            tasks_path = root / tasks_path

        if not output_path.is_absolute():
            output_path = root / output_path

        if not report_path.is_absolute():
            report_path = root / report_path

        bank = load_task_bank(tasks_path)
        configs = load_teacher_configs(root)

        if report_path.resolve() in {
            output_path.resolve(), output_path.with_name(output_path.name + ".lock").resolve()
        }:
            raise ValueError("Report und Checkpoint müssen verschiedene Dateien sein.")
        if report_path.exists() and not args.resume:
            raise ValueError("Report existiert bereits. Neuen Pfad oder --resume verwenden.")
        with CandidateCheckpoint(output_path, resume=args.resume) as checkpoint:
            reused = len(checkpoint.rows)
            legacy = sum("request_fingerprint" not in row for row in checkpoint.rows)
            rows, outcomes, resolutions = asyncio.run(
                generate_candidates(
                    configs, bank, root=root, concurrency=args.concurrency,
                    checkpoint=checkpoint,
                )
            )
            errors = [error for error in outcomes if not error.get("skipped")]
            skipped = sum(bool(error.get("skipped")) for error in outcomes)
            expected = sum(len(task.teacher_aliases) * task.samples_per_teacher for task in bank.tasks)
            complete = not errors and not skipped and len(rows) == expected
            costs = [row.get("cost_usd") for row in rows]

            report = {
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "tasks": len(bank.tasks),
                "candidates": len(rows),
                "errors": len(errors),
                "status": "completed" if complete else "partial",
                "reused_candidates": reused,
                "new_candidates": len(rows) - reused,
                "legacy_reused_candidates": legacy,
                "skipped": skipped,
                "expected_candidates": expected,
                "known_cost_usd": float(sum(
                    (Decimal(str(cost)) for cost in costs if cost is not None), Decimal(0)
                )),
                "unknown_cost_candidates": sum(cost is None for cost in costs),
                "stop_reason": next(
                    (error["error_code"] for error in errors
                     if error.get("http_status") in {401, 402, 403}), None
                ),
                "teacher_resolutions": resolutions,
                "output": str(output_path),
                "error_details": errors,
            }

            write_json(report_path, report)

            print(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )

            return 0 if complete else (2 if rows else 1)

    except (
        ValueError,
        OSError,
        json.JSONDecodeError,
        TeacherError,
    ) as exc:
        print(
            f"Fehler: {exc}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("Abgebrochen; gespeicherte Kandidaten bleiben erhalten.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
