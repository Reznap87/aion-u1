from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any

from .tasks import TaskBank, verify_response


def normalize_response(response: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        response.strip().casefold(),
    )


def rank_candidates(
    rows: list[dict[str, Any]],
    bank: TaskBank,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    task_map = bank.task_map()
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        task_id = str(
            row.get("task_id", "")
        )

        if task_id not in task_map:
            raise ValueError(
                f"Unbekannte Task-ID im Kandidaten: {task_id}"
            )

        task = task_map[task_id]
        response = str(
            row.get("response", "")
        )

        result = verify_response(
            response,
            task.verifier,
        )

        updated = copy.deepcopy(row)

        updated["verification"] = {
            "status": (
                "verified"
                if result.passed
                else (
                    "error"
                    if result.error
                    else "rejected"
                )
            ),
            "score": result.score,
            "method": result.method,
        }

        updated["verification_reason"] = result.reason
        grouped[task_id].append(updated)

    deduplicated: list[
        dict[str, Any]
    ] = []
    best: list[dict[str, Any]] = []
    preferences: list[
        dict[str, Any]
    ] = []

    for task_id in sorted(grouped):
        candidates = grouped[task_id]

        buckets: dict[
            str,
            list[dict[str, Any]],
        ] = defaultdict(list)

        for candidate in candidates:
            key = normalize_response(
                str(
                    candidate.get(
                        "response",
                        "",
                    )
                )
            )
            buckets[key].append(candidate)

        task_candidates: list[
            dict[str, Any]
        ] = []

        for key in sorted(buckets):
            duplicates = buckets[key]

            representative = min(
                duplicates,
                key=lambda item: (
                    -float(
                        item.get(
                            "verification",
                            {},
                        ).get(
                            "score",
                            0.0,
                        )
                    ),
                    int(
                        item.get(
                            "latency_ms",
                            0,
                        )
                    ),
                    str(
                        item.get(
                            "teacher_alias",
                            "",
                        )
                    ),
                ),
            )

            representative = copy.deepcopy(
                representative
            )

            provenance = dict(
                representative.get(
                    "provenance",
                )
                or {}
            )

            provenance[
                "duplicate_teacher_aliases"
            ] = sorted(
                {
                    str(
                        item.get(
                            "teacher_alias",
                            "",
                        )
                    )
                    for item in duplicates
                }
            )

            provenance["duplicate_count"] = len(
                duplicates
            )

            representative["provenance"] = provenance
            task_candidates.append(
                representative
            )

        task_candidates.sort(
            key=lambda item: (
                -float(
                    item.get(
                        "verification",
                        {},
                    ).get(
                        "score",
                        0.0,
                    )
                ),
                int(
                    item.get(
                        "latency_ms",
                        0,
                    )
                ),
                str(
                    item.get(
                        "teacher_alias",
                        "",
                    )
                ),
            )
        )

        deduplicated.extend(task_candidates)

        if task_candidates:
            best.append(task_candidates[0])

        if len(task_candidates) >= 2:
            chosen = task_candidates[0]
            rejected = task_candidates[1]

            chosen_score = float(
                chosen.get(
                    "verification",
                    {},
                ).get(
                    "score",
                    0.0,
                )
            )

            rejected_score = float(
                rejected.get(
                    "verification",
                    {},
                ).get(
                    "score",
                    0.0,
                )
            )

            if chosen_score > rejected_score:
                preferences.append(
                    {
                        "id": (
                            f"aion-preference-{task_id}-"
                            f"{chosen.get('teacher_alias', '')}-"
                            f"{rejected.get('teacher_alias', '')}"
                        ),
                        "task_id": task_id,
                        "group_id": task_id,
                        "prompt": chosen.get(
                            "prompt",
                            "",
                        ),
                        "capabilities": chosen.get(
                            "capabilities",
                            [],
                        ),
                        "chosen": chosen.get(
                            "response",
                            "",
                        ),
                        "rejected": rejected.get(
                            "response",
                            "",
                        ),
                        "chosen_teacher_alias": chosen.get(
                            "teacher_alias"
                        ),
                        "rejected_teacher_alias": rejected.get(
                            "teacher_alias"
                        ),
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "verification": {
                            "status": "verified",
                            "score": 1.0,
                            "method": "teacher_comparison",
                        },
                    }
                )

    report = {
        "input_candidates": len(rows),
        "deduplicated_candidates": len(
            deduplicated
        ),
        "tasks_with_candidates": len(grouped),
        "best_candidates": len(best),
        "preference_pairs": len(preferences),
        "verified_candidates": sum(
            row.get(
                "verification",
                {},
            ).get(
                "status",
            ) == "verified"
            for row in deduplicated
        ),
        "rejected_candidates": sum(
            row.get(
                "verification",
                {},
            ).get(
                "status",
            ) == "rejected"
            for row in deduplicated
        ),
        "errored_candidates": sum(
            row.get(
                "verification",
                {},
            ).get(
                "status",
            ) == "error"
            for row in deduplicated
        ),
    }

    return (
        deduplicated,
        best,
        preferences,
        report,
    )