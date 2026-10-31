from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (
    TrainingDataError,
    read_jsonl,
    select_canonical,
    split_rows,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aion-u1-prepare-sft",
        description=(
            "Wählt eine kanonische Antwort pro "
            "Aufgabengruppe und erstellt optional einen "
            "leakfreien Train/Eval-Split."
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--train-output",
    )
    parser.add_argument(
        "--eval-output",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--require-approved",
        action="store_true",
    )
    parser.add_argument(
        "--teacher-priority",
        default="",
        help=(
            "Kommagetrennte Teacher-Aliase in "
            "Präferenzreihenfolge."
        ),
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    if bool(args.train_output) != bool(
        args.eval_output
    ):
        raise TrainingDataError(
            "train-output und eval-output müssen "
            "gemeinsam angegeben werden."
        )

    raw_rows = read_jsonl(
        Path(args.input)
    )

    priority = tuple(
        alias.strip()
        for alias in args.teacher_priority.split(",")
        if alias.strip()
    )

    selected = select_canonical(
        raw_rows,
        require_approved=args.require_approved,
        min_score=args.min_score,
        teacher_priority=priority,
    )

    if not selected:
        raise TrainingDataError(
            "Keine kanonischen Trainingsdaten nach "
            "den Filtern gefunden."
        )

    output = Path(args.output)
    write_jsonl(selected, output)

    report: dict[str, object] = {
        "input": str(args.input),
        "output": str(output),
        "raw_rows": len(raw_rows),
        "selected_rows": len(selected),
        "approved_rows": sum(
            row.get("provenance", {}).get(
                "training_approved"
            ) is True
            for row in selected
        ),
    }

    if args.train_output and args.eval_output:
        train_rows, eval_rows, split_stats = split_rows(
            selected,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
        )

        train_output = Path(args.train_output)
        eval_output = Path(args.eval_output)

        write_jsonl(train_rows, train_output)
        write_jsonl(eval_rows, eval_output)

        report["train_output"] = str(train_output)
        report["eval_output"] = str(eval_output)
        report["split"] = split_stats

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
