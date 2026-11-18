from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .audit import has_valid_measurement, run_audit
from .capabilities import CapabilityRegistry
from .dataset import build_dataset, write_jsonl
from .models import AuditCase, AuditReport, TeacherConfig
from .privacy import KIND_SECRET, scan_row, scan_text, summarize
from .provenance import collect as collect_provenance
from .teachers.base import TeacherError
from .teachers.openrouter import DEFAULT_MAX_TOKENS, OpenRouterClient
from .teachers.resolver import TeacherResolution, resolve_teachers


def repository_root() -> Path:
    configured = os.getenv("AION_U1_ROOT", "").strip()

    if configured:
        return Path(configured).expanduser().resolve()

    current = Path.cwd().resolve()

    if (current / "configs").is_dir():
        return current

    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError(f"YAML muss ein Mapping sein: {path}")

    return raw


def load_teachers(path: Path) -> list[TeacherConfig]:
    raw = load_yaml(path)
    entries = raw.get("teachers")

    if not isinstance(entries, list):
        raise ValueError("teachers muss eine Liste sein")

    return [TeacherConfig.model_validate(entry) for entry in entries]


def load_cases(path: Path) -> list[AuditCase]:
    raw = load_yaml(path)
    entries = raw.get("cases")

    if not isinstance(entries, list):
        raise ValueError("cases muss eine Liste sein")

    return [AuditCase.model_validate(entry) for entry in entries]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolution_payload(
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


async def resolve_all(
    configs: list[TeacherConfig],
) -> tuple[
    OpenRouterClient,
    list[TeacherResolution],
]:
    if not configs:
        raise ValueError("Keine Teacher konfiguriert")

    api_key = os.getenv(configs[0].api_key_env, "").strip()

    if not api_key:
        raise ValueError(
            f"API-Key {configs[0].api_key_env} fehlt"
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
        resolutions = await resolve_teachers(client, configs)
    except Exception:
        await client.close()
        raise

    return client, resolutions


def command_capabilities(args: argparse.Namespace) -> int:
    root = repository_root()
    registry = CapabilityRegistry.load(
        root / "configs" / "capabilities.yaml"
    )

    if args.json:
        payload = {
            "total_weight": registry.total_weight,
            "capabilities": [
                spec.model_dump()
                for spec in registry.list()
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for spec in registry.list():
        parents = ", ".join(spec.parents) or "-"
        print(
            f"{spec.id:36} weight={spec.weight:.2f} "
            f"parents={parents}"
        )

    print(f"\nGesamtgewicht: {registry.total_weight:.3f}")
    return 0


def command_verify_teachers(args: argparse.Namespace) -> int:
    root = repository_root()
    configs = load_teachers(root / "configs" / "teachers.yaml")

    async def execute() -> int:
        client, resolutions = await resolve_all(configs)

        try:
            payload = {
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "teachers": [
                    resolution_payload(resolution)
                    for resolution in resolutions
                ],
            }

            if args.output:
                write_json(Path(args.output), payload)

            print(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        finally:
            await client.close()

    return asyncio.run(execute())


def command_audit(args: argparse.Namespace) -> int:
    root = repository_root()
    teacher_path = root / "configs" / "teachers.yaml"
    suite_path = Path(args.suite)

    if not suite_path.is_absolute():
        suite_path = root / suite_path

    configs = load_teachers(teacher_path)
    cases = load_cases(suite_path)
    registry = CapabilityRegistry.load(
        root / "configs" / "capabilities.yaml"
    )

    for case in cases:
        registry.get(case.capability)

    async def execute() -> int:
        client, resolutions = await resolve_all(configs)

        try:
            report = await run_audit(
                client,
                resolutions,
                cases,
                registry,
                concurrency=args.concurrency,
                root=root,
                max_tokens=args.max_tokens,
            )
        finally:
            await client.close()

        report.suite = load_yaml(suite_path).get(
            "suite",
            suite_path.stem,
        )
        report.suite_path = str(suite_path)

        report.provenance = collect_provenance(
            root,
            {
                "suite": suite_path,
                "teachers": teacher_path,
                "capabilities": (
                    root / "configs" / "capabilities.yaml"
                ),
            },
            command=sys.argv[1:],
        )

        privacy_counts = summarize(
            finding
            for result in report.results
            for finding in scan_text(
                result.response or "", field="response"
            )
        )

        output_path = Path(args.output)
        write_json(
            output_path,
            report.model_dump(mode="json"),
        )

        print(
            json.dumps(
                {
                    "report": str(output_path),
                    "scores": {
                        alias: score.model_dump()
                        for alias, score in report.scores.items()
                    },
                    # AGENTS.md: keine Secrets in Logs — der Report ist
                    # eine Datei und wird hier mitgescannt.
                    "privacy": privacy_counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        if privacy_counts[KIND_SECRET]:
            print(
                "Warnung: Teacher-Antworten im Report enthalten "
                f"{privacy_counts[KIND_SECRET]} Secret(s). Der Report "
                "bleibt als Messung unveraendert; betroffene Zeilen nicht "
                "freigeben.",
                file=sys.stderr,
            )

        if not has_valid_measurement(report.results):
            print(
                "Fehler: Keine verwertbare Messung — alle "
                f"{len(report.results)} Fälle sind abgebrochen oder "
                "abgeschnitten. Der Report ist kein Ergebnis; "
                "Kontostand und --max-tokens prüfen.",
                file=sys.stderr,
            )
            return 1

        return 0

    return asyncio.run(execute())


def command_dataset(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    report = AuditReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )

    candidates, pending = build_dataset(report)

    output_path = Path(args.output)
    pending_path = Path(args.pending_output)

    write_jsonl(candidates, output_path)
    write_jsonl(pending, pending_path)

    # AGENTS.md: keine Secrets oder personenbezogenen Daten in Datensaetzen.
    # Hier nur zaehlen und anzeigen; die harte Sperre liegt in approve-sft.
    privacy_counts = summarize(
        finding
        for row in list(candidates) + list(pending)
        for finding in scan_row(row)
    )

    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "pending_review_examples": len(pending),
                "candidate_output": str(output_path),
                "pending_output": str(pending_path),
                "privacy": privacy_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aion-u1",
        description=(
            "AION-U1 Teacher-Verifikation, Capability-Audit "
            "und Trainingsdatenexport"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    capabilities = subparsers.add_parser(
        "capabilities",
        help="Capability-Spezifikationen anzeigen",
    )
    capabilities.add_argument(
        "--json",
        action="store_true",
    )
    capabilities.set_defaults(
        func=command_capabilities,
    )

    verify = subparsers.add_parser(
        "verify-teachers",
        help="Teacher live gegen den Provider-Katalog prüfen",
    )
    verify.add_argument(
        "--output",
        default="reports/teachers.json",
    )
    verify.set_defaults(
        func=command_verify_teachers,
    )

    audit = subparsers.add_parser(
        "audit",
        help="Capability-Audit ausführen",
    )
    audit.add_argument(
        "--suite",
        default="evaluations/capability_suite/smoke.yaml",
    )
    audit.add_argument(
        "--output",
        default="reports/audit-smoke.json",
    )
    audit.add_argument(
        "--concurrency",
        type=int,
        default=4,
    )
    audit.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    audit.set_defaults(
        func=command_audit,
    )

    dataset = subparsers.add_parser(
        "dataset",
        help="Auditresultate in Pilot-Trainingsdaten umwandeln",
    )
    dataset.add_argument(
        "--report",
        required=True,
    )
    dataset.add_argument(
        "--output",
        default="data/generated/teacher_candidates.jsonl",
    )
    dataset.add_argument(
        "--pending-output",
        default="data/generated/smoke_sft_pending.jsonl",
    )
    dataset.set_defaults(
        func=command_dataset,
    )

    return parser


def main() -> int:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    try:
        return int(args.func(args))
    except (ValueError, TeacherError, FileNotFoundError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Abgebrochen.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
