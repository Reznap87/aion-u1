"""Deterministic verification primitives; no execution of model output."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    score: float | None
    passed: bool
    reason: str
    method: str
    error: bool = False

    def __post_init__(self) -> None:
        if self.error:
            if self.score is not None or self.passed:
                raise ValueError("Infrastructure errors cannot have a score or pass")
        elif self.score is None or not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Measured results require a finite score in [0, 1]")

    @property
    def status(self) -> str:
        return "error" if self.error else "passed" if self.passed else "failed"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("Non-finite JSON constant")


def extract_json(text: str) -> Any:
    """Accept one JSON document or one unambiguous Markdown JSON fence."""
    stripped = text.strip()
    options = dict(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    try:
        return json.loads(stripped, **options)
    except json.JSONDecodeError:
        pass
    fences = re.findall(r"^```(?:json)?[ \t]*\r?\n(.*?)^```[ \t]*$", stripped,
                        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if len(fences) != 1 or stripped.count("```") != 2:
        raise ValueError("Expected one unambiguous JSON document")
    outside = re.sub(r"^```(?:json)?[ \t]*\r?\n.*?^```[ \t]*$", "", stripped,
                     flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if any(char in outside for char in "{}[]"):
        raise ValueError("Ambiguous JSON outside fence")
    return json.loads(fences[0], **options)


def score_text(
    response: str, *, kind: str, values: list[str] | None = None,
    expected: Any = None, case_insensitive: bool = True,
    partial: bool = False, pass_threshold: float = 1.0,
) -> VerificationResult:
    if not 0 < pass_threshold <= 1:
        raise ValueError("Threshold must be in (0, 1]")
    if partial and kind != "contains_all":
        raise ValueError("Partial scoring is only supported for contains_all")
    if kind != "json_equals" and (not values or any(not v for v in values)):
        raise ValueError("Nonempty conditions required")
    text = response.strip()
    needles = values or []
    haystack = text.lower() if case_insensitive else text
    normalized = [v.lower() if case_insensitive else v for v in needles]
    if kind == "contains_all":
        hits = sum(v in haystack for v in normalized)
        score = hits / len(needles) if partial else float(hits == len(needles))
    elif kind == "contains_any":
        score = float(any(v in haystack for v in normalized))
    elif kind == "not_contains":
        score = float(not any(v in haystack for v in normalized))
    elif kind == "exact_match":
        score = float(text == needles[0].strip())
    elif kind == "regex":
        score = float(any(re.search(v, text, re.IGNORECASE if case_insensitive else 0)
                          for v in needles))
    elif kind == "json_equals":
        try:
            score = float(extract_json(text) == expected)
        except ValueError:
            score = 0.0
    else:
        raise ValueError("Unknown text verifier")
    return VerificationResult(score, score >= pass_threshold, "Conditions evaluated", kind)
