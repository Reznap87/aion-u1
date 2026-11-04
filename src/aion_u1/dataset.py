from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import AuditReport


def build_dataset(
    report: AuditReport,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exportiert Smoke-Erfolge zur manuellen Prüfung, nicht zum Training."""
    candidates: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for result in report.results:
        digest = hashlib.sha256(
            (
                f"{result.teacher_alias}|"
                f"{result.case_id}|"
                f"{result.response}"
            ).encode("utf-8")
        ).hexdigest()

        candidate_id = f"aion-{digest[:24]}"

        candidate = {
            "id": candidate_id,
            "case_id": result.case_id,
            "capability": result.capability,
            "prompt": result.prompt,
            "response": result.response,
            "teacher_alias": result.teacher_alias,
            "teacher_model_id": result.teacher_model_id,
            "score": result.score,
            "passed": result.passed,
            "verifier_type": result.verifier_type,
            "latency_ms": result.latency_ms,
            "usage": result.usage,
            "error": result.error,
        }
        candidates.append(candidate)

        if result.passed and result.error is None:
            pending.append(
                {
                    "id": candidate_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Du bist AION-U1. Antworte korrekt, "
                                "präzise und vollständig."
                            ),
                        },
                        {
                            "role": "user",
                            "content": result.prompt,
                        },
                        {
                            "role": "assistant",
                            "content": result.response,
                        },
                    ],
                    "capabilities": [result.capability],
                    "provenance": {
                        "teacher_alias": result.teacher_alias,
                        "teacher_model_id": result.teacher_model_id,
                    },
                    "verification": {
                        "status": "pending_review",
                        "score": result.score,
                        "method": (
                            f"smoke:{result.verifier_type}"
                        ),
                    },
                }
            )

    return candidates, pending


def write_jsonl(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
