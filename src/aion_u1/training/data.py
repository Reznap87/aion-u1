from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class TrainingDataError(ValueError):
    """The training data is malformed or violates a hard constraint."""


VALID_ROLES = {"system", "user", "assistant", "tool"}
UNREVIEWED_LICENSES = {
    "",
    "unreviewed",
    "pending",
    "unknown",
    "license-unknown",
}


def _digest(value: Any, length: int = 64) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()[:length]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(
                    f"Ungültiges JSON in {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise TrainingDataError(
                    f"Zeile {line_number} in {path} ist kein JSON-Objekt."
                )

            rows.append(row)

    return rows


def write_jsonl(
    rows: Iterable[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def _normalize_messages(
    row: dict[str, Any],
) -> list[dict[str, str]]:
    raw_messages = row.get("messages")

    if raw_messages is None:
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()

        if not prompt or not response:
            raise TrainingDataError(
                "Ohne messages müssen prompt und response "
                "vorhanden sein."
            )

        raw_messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

    if not isinstance(raw_messages, list) or not raw_messages:
        raise TrainingDataError(
            "messages muss eine nicht leere Liste sein."
        )

    messages: list[dict[str, str]] = []

    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise TrainingDataError(
                f"messages[{index}] muss ein JSON-Objekt sein."
            )

        role = str(message.get("role", "")).strip()
        content = message.get("content", "")

        if role not in VALID_ROLES:
            raise TrainingDataError(
                f"Ungültige Nachrichtenrolle: {role!r}"
            )

        if not isinstance(content, str):
            content = json.dumps(
                content,
                ensure_ascii=False,
            )

        if role == "assistant" and not content.strip():
            raise TrainingDataError(
                "Eine leere Assistant-Antwort ist kein "
                "gültiges SFT-Ziel."
            )

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    if not any(
        message["role"] == "user"
        for message in messages
    ):
        raise TrainingDataError(
            "SFT-Daten benötigen mindestens eine User-Nachricht."
        )

    if messages[-1]["role"] != "assistant":
        raise TrainingDataError(
            "Die letzte Nachricht muss eine Assistant-Antwort sein."
        )

    return messages


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class CanonicalExample:
    source_id: str
    group_id: str
    case_id: str | None
    messages: list[dict[str, str]]
    capabilities: tuple[str, ...]
    provenance: dict[str, Any]
    verification: dict[str, Any]
    score: float


def canonicalize(
    row: dict[str, Any],
    index: int = 0,
) -> CanonicalExample:
    messages = _normalize_messages(row)

    provenance = dict(
        row.get("provenance") or {}
    )
    verification = dict(
        row.get("verification") or {}
    )

    teacher_alias = str(
        provenance.get("teacher_alias")
        or row.get("teacher_alias")
        or "unknown"
    )

    teacher_model_id = str(
        provenance.get("teacher_model_id")
        or row.get("teacher_model_id")
        or "unknown"
    )

    license_name = str(
        provenance.get("license")
        or "unreviewed"
    ).strip()

    training_approved = (
        provenance.get("training_approved") is True
        and license_name.lower()
        not in UNREVIEWED_LICENSES
    )

    provenance.setdefault(
        "source",
        "teacher_generated",
    )
    provenance["license"] = license_name
    provenance["training_approved"] = training_approved
    provenance["teacher_alias"] = teacher_alias
    provenance["teacher_model_id"] = teacher_model_id

    status = str(
        verification.get("status")
        or (
            "verified"
            if row.get("passed", True)
            else "unverified"
        )
    )

    method = str(
        verification.get("method")
        or row.get("verifier_type")
        or "unknown"
    )

    score = _score(
        verification.get(
            "score",
            row.get("score", 0.0),
        )
    )

    verification = {
        "status": status,
        "score": score,
        "method": method,
    }

    existing_group = row.get("group_id")
    case_id = row.get("case_id")

    if existing_group:
        group_id = str(existing_group)
    elif case_id:
        group_id = f"case:{case_id}"
    else:
        group_id = f"prompt:{_digest(messages[:-1])}"

    capabilities = tuple(
        sorted(
            {
                str(capability)
                for capability in row.get(
                    "capabilities",
                    [],
                )
                if str(capability)
            }
        )
    )

    if not capabilities:
        raise TrainingDataError(
            "Jede Trainingsprobe benötigt mindestens eine Capability."
        )

    source_id = str(
        row.get("id")
        or f"source-{index}"
    )

    return CanonicalExample(
        source_id=source_id,
        group_id=group_id,
        case_id=(
            str(case_id)
            if case_id
            else None
        ),
        messages=messages,
        capabilities=capabilities,
        provenance=provenance,
        verification=verification,
        score=score,
    )


def is_training_approved(
    row: dict[str, Any],
) -> bool:
    provenance = row.get("provenance") or {}

    license_name = str(
        provenance.get("license")
        or "unreviewed"
    ).strip()

    return (
        provenance.get("training_approved") is True
        and license_name.lower()
        not in UNREVIEWED_LICENSES
    )


def select_canonical(
    rows: Iterable[dict[str, Any]],
    *,
    require_approved: bool = False,
    min_score: float = 0.0,
    teacher_priority: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    groups: dict[
        str,
        list[CanonicalExample],
    ] = {}

    for index, row in enumerate(rows):
        example = canonicalize(row, index)
        groups.setdefault(
            example.group_id,
            [],
        ).append(example)

    priority = {
        alias: position
        for position, alias in enumerate(
            teacher_priority
        )
    }

    selected_rows: list[dict[str, Any]] = []

    for group_id, examples in sorted(groups.items()):
        eligible = [
            example
            for example in examples
            if example.verification["status"] == "verified"
            and example.score >= min_score
        ]

        if not eligible:
            continue

        approved = [
            example
            for example in eligible
            if example.provenance.get(
                "training_approved"
            ) is True
            and str(
                example.provenance.get(
                    "license",
                    "",
                )
            ).lower() not in UNREVIEWED_LICENSES
        ]

        if require_approved:
            if not approved:
                continue

            eligible = approved
        elif approved:
            eligible = approved

        selected = min(
            eligible,
            key=lambda example: (
                -example.score,
                priority.get(
                    str(
                        example.provenance.get(
                            "teacher_alias"
                        )
                    ),
                    len(priority) + 1,
                ),
                str(
                    example.provenance.get(
                        "teacher_alias"
                    )
                ),
                example.source_id,
            ),
        )

        provenance = dict(
            selected.provenance
        )

        provenance["selected_teacher_alias"] = (
            provenance.get("teacher_alias")
        )
        provenance["selected_teacher_model_id"] = (
            provenance.get("teacher_model_id")
        )
        provenance["selection_score"] = selected.score
        provenance["candidate_teacher_aliases"] = sorted(
            {
                str(
                    example.provenance.get(
                        "teacher_alias"
                    )
                )
                for example in examples
            }
        )
        provenance["candidate_count"] = len(
            examples
        )

        example_id = (
            "aion-sft-"
            f"{_digest([group_id, selected.messages[-1]])}"
        )

        selected_rows.append(
            {
                "id": example_id,
                "group_id": group_id,
                "case_id": selected.case_id,
                "messages": selected.messages,
                "capabilities": list(
                    selected.capabilities
                ),
                "provenance": provenance,
                "verification": selected.verification,
                "source_id": selected.source_id,
            }
        )

    return selected_rows


def split_rows(
    rows: list[dict[str, Any]],
    *,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not 0.0 <= eval_fraction < 1.0:
        raise TrainingDataError(
            "eval_fraction muss zwischen 0.0 "
            "und 1.0 liegen."
        )

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for row in rows:
        group_id = str(
            row.get("group_id")
            or _digest(row.get("messages", []))
        )

        groups.setdefault(
            group_id,
            [],
        ).append(row)

    ordered_groups = sorted(
        groups,
        key=lambda group_id: _digest(
            [seed, group_id]
        ),
    )

    eval_count = int(
        round(
            len(ordered_groups)
            * eval_fraction
        )
    )

    if (
        eval_fraction > 0
        and len(ordered_groups) >= 2
    ):
        eval_count = max(1, eval_count)

    eval_count = min(
        eval_count,
        max(0, len(ordered_groups) - 1),
    )

    eval_groups = set(
        ordered_groups[:eval_count]
    )

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for group_id, group_rows in groups.items():
        if group_id in eval_groups:
            eval_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)

    train_rows.sort(
        key=lambda row: str(row.get("id"))
    )
    eval_rows.sort(
        key=lambda row: str(row.get("id"))
    )

    train_groups = {
        str(row.get("group_id"))
        for row in train_rows
    }

    eval_groups_in_rows = {
        str(row.get("group_id"))
        for row in eval_rows
    }

    if train_groups & eval_groups_in_rows:
        raise TrainingDataError(
            "Train- und Eval-Gruppen "
            "überschneiden sich."
        )

    stats = {
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_groups": len(train_groups),
        "eval_groups": len(eval_groups_in_rows),
        "group_overlap": 0,
    }

    return train_rows, eval_rows, stats
