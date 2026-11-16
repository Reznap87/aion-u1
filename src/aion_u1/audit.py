from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .capabilities import CapabilityRegistry
from .media import multimodal_content
from .teachers.openrouter import DEFAULT_MAX_TOKENS
from .models import (
    AuditCase,
    AuditReport,
    CaseResult,
    TeacherResolutionRecord,
    TeacherScore,
    VerifierSpec,
)
from .teachers.base import TeacherClient
from .teachers.resolver import TeacherResolution


def _extract_json(text: str) -> Any:
    stripped = text.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    starts = [
        position
        for position in (stripped.find("{"), stripped.find("["))
        if position >= 0
    ]

    if not starts:
        raise ValueError("Kein JSON gefunden")

    start = min(starts)
    opener = stripped[start]
    closer = "}" if opener == "{" else "]"
    end = stripped.rfind(closer)

    if end < start:
        raise ValueError("Unvollständiges JSON")

    return json.loads(stripped[start : end + 1])


def score_response(
    response: str,
    verifier: VerifierSpec,
) -> tuple[float, str]:
    text = response.strip()

    if verifier.type == "contains_all":
        missing = [value for value in verifier.values if value not in text]

        if missing:
            return 0.0, f"Fehlende Werte: {missing}"

        return 1.0, "Alle required values gefunden"

    if verifier.type == "not_contains":
        forbidden = [value for value in verifier.values if value in text]

        if forbidden:
            return 0.0, f"Verbotene Werte gefunden: {forbidden}"

        return 1.0, "Keine verbotenen Werte gefunden"

    if verifier.type == "exact_match":
        expected = verifier.values[0].strip()

        if text == expected:
            return 1.0, "Exakter Match"

        return 0.0, f"Erwartet: {expected!r}"

    if verifier.type == "regex":
        for pattern in verifier.values:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return 1.0, f"Regex gefunden: {pattern}"

        return 0.0, "Kein Regex gefunden"

    if verifier.type == "json_equals":
        try:
            actual = _extract_json(text)
        except (ValueError, json.JSONDecodeError) as exc:
            return 0.0, f"JSON ungültig: {exc}"

        if actual == verifier.expected:
            return 1.0, "JSON entspricht Erwartung"

        return 0.0, (
            f"JSON weicht ab: erwartet={verifier.expected!r}, "
            f"actual={actual!r}"
        )

    return 0.0, "Unbekannter Verifier"


def build_case_input(
    case: AuditCase,
    root: Path | None = None,
) -> str | list[dict[str, Any]]:
    if case.modality == "text":
        return case.prompt

    if not case.image_path:
        raise ValueError(f"Case {case.id} benötigt image_path.")

    return multimodal_content(
        case.prompt,
        case.image_path,
        root or Path.cwd(),
        case.image_mime_type,
    )


async def _run_case(
    client: TeacherClient,
    resolution: TeacherResolution,
    case: AuditCase,
    semaphore: asyncio.Semaphore,
    root: Path | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> CaseResult:
    async with semaphore:
        try:
            response = await client.generate(
                resolution.model_id or "",
                [
                    {
                        "role": "user",
                        "content": build_case_input(case, root),
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            score, _reason = score_response(
                response.content,
                case.verifier,
            )

            return CaseResult(
                teacher_alias=resolution.config.alias,
                teacher_model_id=resolution.model_id or "",
                case_id=case.id,
                capability=case.capability,
                prompt=case.prompt,
                response=response.content,
                score=score,
                passed=score == 1.0,
                verifier_type=case.verifier.type,
                latency_ms=response.latency_ms,
                finish_reason=response.finish_reason,
                usage=response.usage,
                max_tokens=response.max_tokens,
            )
        except Exception as exc:
            return CaseResult(
                teacher_alias=resolution.config.alias,
                teacher_model_id=resolution.model_id or "",
                case_id=case.id,
                capability=case.capability,
                prompt=case.prompt,
                response="",
                score=None,
                passed=False,
                verifier_type=case.verifier.type,
                latency_ms=0,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )


async def run_audit(
    client: TeacherClient,
    resolutions: list[TeacherResolution],
    cases: list[AuditCase],
    registry: CapabilityRegistry,
    *,
    concurrency: int = 4,
    root: Path | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AuditReport:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks: list[asyncio.Task[CaseResult]] = []

    for resolution in resolutions:
        if resolution.status != "verified" or not resolution.model_id:
            continue

        for case in cases:
            tasks.append(
                asyncio.create_task(
                    _run_case(
                        client,
                        resolution,
                        case,
                        semaphore,
                        root,
                        max_tokens,
                    )
                )
            )

    results = (
        await asyncio.gather(*tasks)
        if tasks
        else []
    )

    teacher_records = [
        TeacherResolutionRecord(
            alias=resolution.config.alias,
            display_name=resolution.config.display_name,
            provider=resolution.config.provider,
            model_id=resolution.model_id,
            status=resolution.status,
            source=resolution.source,
            suggestions=list(resolution.suggestions),
        )
        for resolution in resolutions
    ]

    scores = calculate_scores(results, registry)

    return AuditReport(
        generated_at=datetime.now(timezone.utc),
        suite="audit",
        suite_path="",
        teacher_resolutions=teacher_records,
        results=results,
        scores=scores,
    )


def calculate_scores(
    results: list[CaseResult],
    registry: CapabilityRegistry,
) -> dict[str, TeacherScore]:
    teacher_aliases = sorted({result.teacher_alias for result in results})
    scores: dict[str, TeacherScore] = {}

    for alias in teacher_aliases:
        teacher_results = [
            result for result in results
            if result.teacher_alias == alias
        ]

        evaluated_results = [
            result for result in teacher_results
            if result.error is None
            and result.score is not None
            and result.finish_reason != "length"
        ]
        truncated = sum(
            1 for result in teacher_results
            if result.finish_reason == "length"
        )
        capability_ids = sorted(
            {result.capability for result in evaluated_results}
        )
        capability_scores: dict[str, float] = {}

        for capability_id in capability_ids:
            values = [
                result.score
                for result in evaluated_results
                if result.capability == capability_id
            ]
            capability_scores[capability_id] = (
                sum(values) / len(values)
                if values
                else 0.0
            )

        tested_weight = sum(
            registry.get(capability_id).weight
            for capability_id in capability_scores
        )
        total_weight = registry.total_weight or 1.0
        coverage = min(1.0, tested_weight / total_weight)

        if capability_scores and tested_weight > 0:
            logarithmic_sum = 0.0

            for capability_id, score in capability_scores.items():
                normalized_weight = (
                    registry.get(capability_id).weight
                    / tested_weight
                )

                if score <= 0:
                    logarithmic_sum = -math.inf
                    break

                logarithmic_sum += (
                    normalized_weight * math.log(score)
                )

            measured_score = (
                0.0
                if logarithmic_sum == -math.inf
                else math.exp(logarithmic_sum)
            )
        else:
            measured_score = None  # Kein auswertbarer Case

        # Der AION-Index ist erst mit voller Capability-Abdeckung
        # auswertbar. Ohne volle Abdeckung wird None geliefert statt
        # eines hochgewerteten Teilscores.
        full_coverage = all(
            spec.id in capability_scores
            for spec in registry.list()
        )

        overall = (
            measured_score
            if measured_score is not None and full_coverage
            else None
        )

        scores[alias] = TeacherScore(
            teacher_alias=alias,
            overall_score=overall,
            measured_score=measured_score,
            coverage=coverage,
            cases=len(teacher_results),
            passed=sum(result.passed for result in teacher_results),
            errored=sum(result.error is not None for result in teacher_results),
            truncated=truncated,
            capability_scores=capability_scores,
        )

    return scores
