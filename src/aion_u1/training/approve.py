from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import (
    TrainingDataError,
    read_jsonl,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aion-u1-approve-sft",
        description=(
            "Zeichnet eine manuelle Trainingsfreigabe auf. "
            "Dieses Werkzeug ersetzt keine rechtliche Prüfung."
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
        "--license",
        required=True,
    )
    parser.add_argument(
        "--operator",
        required=True,
    )
    parser.add_argument(
        "--ids-file",
    )
    parser.add_argument(
        "--confirm-all",
        action="store_true",
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    if (
        not args.ids_file
        and not args.confirm_all
    ):
        raise TrainingDataError(
            "Es fehlt --ids-file oder --confirm-all."
        )

    rows = read_jsonl(Path(args.input))
    ids: set[str] | None = None

    if args.ids_file:
        ids = {
            line.strip()
            for line in Path(
                args.ids_file
            ).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
            and not line.lstrip().startswith("#")
        }

    approved_at = datetime.now(
        timezone.utc
    ).isoformat()

    approved_ids: list[str] = []

    for row in rows:
        row_id = str(row.get("id", ""))

        should_approve = (
            args.confirm_all
            or (
                ids is not None
                and row_id in ids
            )
        )

        if not should_approve:
            continue

        provenance = dict(
            row.get("provenance") or {}
        )

        provenance["license"] = args.license
        provenance["training_approved"] = True
        provenance["approval"] = {
            "operator": args.operator,
            "approved_at": approved_at,
            "method": (
                "manual-confirm-all"
                if args.confirm_all
                else "manual-id-list"
            ),
        }

        row["provenance"] = provenance
        approved_ids.append(row_id)

    output = Path(args.output)
    write_jsonl(rows, output)

    report = {
        "input": str(args.input),
        "output": str(output),
        "approved_count": len(approved_ids),
        "total_count": len(rows),
        "license": args.license,
        "operator": args.operator,
        "approved_at": approved_at,
    }

    if ids is not None:
        report["unmatched_ids"] = sorted(
            ids - set(approved_ids)
        )

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
