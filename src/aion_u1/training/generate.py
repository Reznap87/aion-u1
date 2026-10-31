from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from ..models import TeacherConfig
from ..teachers.openrouter import OpenRouterClient
from ..teachers.resolver import (
    TeacherResolution,
    resolve_teachers,
)
from .data import write_jsonl
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


async def generate_one(
    client: OpenRouterClient,
    resolution: TeacherResolution,
    task: TaskSpec,
    sample_index: int,
    semaphore: asyncio.Semaphore,
    root: Path,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    async with semaphore:
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
            error = {
                "task_id": task.id,
                "teacher_alias": resolution.config.alias,
                "teacher_model_id": resolution.model_id,
                "sample_index": sample_index,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                )[:1000],
            }

            return None, error


async def generate_candidates(
    configs: list[TeacherConfig],
    bank: TaskBank,
    *,
    root: Path,
    concurrency: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not configs:
        raise ValueError(
            "Keine Teacher sind konfiguriert."
        )

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

        semaphore = asyncio.Semaphore(
            max(1, concurrency)
        )

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
                    continue

                for sample_index in range(
                    task.samples_per_teacher
                ):
                    jobs.append(
                        asyncio.create_task(
                            generate_one(
                                client,
                                resolution,
                                task,
                                sample_index,
                                semaphore,
                                root,
                            )
                        )
                    )

        outcomes = (
            await asyncio.gather(*jobs)
            if jobs
            else []
        )

        rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for row, error in outcomes:
            if row is not None:
                rows.append(row)

            if error is not None:
                errors.append(error)

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

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


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

        rows, errors, resolutions = asyncio.run(
            generate_candidates(
                configs,
                bank,
                root=root,
                concurrency=args.concurrency,
            )
        )

        write_jsonl(rows, output_path)

        report = {
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "tasks": len(bank.tasks),
            "candidates": len(rows),
            "errors": len(errors),
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

        return 0 if rows else 1

    except (
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"Fehler: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())