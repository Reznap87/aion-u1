from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import read_jsonl, write_jsonl
from .ranking import rank_candidates
from .tasks import (
    find_project_root,
    load_task_bank,
)


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
        prog="aion-u1-rank-candidates",
        description=(
            "Verifiziert, dedupliziert und rankt "
            "Multi-Teacher-Kandidaten."
        ),
    )

    parser.add_argument(
        "--tasks",
        default=(
            "training/configs/tasks/pilot.yaml"
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--output",
        default=(
            "data/generated/scored_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--best-output",
        default=(
            "data/generated/best_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--preferences-output",
        default=(
            "data/generated/preference_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        default="reports/candidate-ranking.json",
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    root = find_project_root()

    try:
        tasks_path = Path(args.tasks)
        input_path = Path(args.input)
        output_path = Path(args.output)
        best_path = Path(args.best_output)
        preferences_path = Path(
            args.preferences_output
        )
        report_path = Path(args.report)

        resolved: list[Path] = []

        for path in (
            tasks_path,
            input_path,
            output_path,
            best_path,
            preferences_path,
            report_path,
        ):
            resolved.append(
                path
                if path.is_absolute()
                else root / path
            )

        (
            tasks_path,
            input_path,
            output_path,
            best_path,
            preferences_path,
            report_path,
        ) = resolved

        bank = load_task_bank(tasks_path)
        rows = read_jsonl(input_path)

        scored, best, preferences, report = rank_candidates(
            rows,
            bank,
        )

        write_jsonl(scored, output_path)
        write_jsonl(best, best_path)
        write_jsonl(preferences, preferences_path)

        report["generated_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        report["input"] = str(input_path)
        report["output"] = str(output_path)
        report["best_output"] = str(best_path)
        report["preferences_output"] = str(
            preferences_path
        )

        write_json(report_path, report)

        print(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            )
        )

        return 0

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