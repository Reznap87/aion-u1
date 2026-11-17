"""Erkennung von Secrets und personenbezogenen Daten in Text und Datensaetzen.

AGENTS.md verlangt: "Keine Secrets oder personenbezogenen Daten in Logs oder
Datensaetzen." Bisher gab es dafuer keine Umsetzung — dieses Modul liefert sie.

`scan_text` findet Muster und maskiert den Treffer, damit selbst ein Befund
nicht die gleiche Stelle noch einmal ausgibt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

# Verbatim wiedergebbare Muster sind absichtlich spezifisch, damit Code und
# normale Prosa nicht sofort ausloesen.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern))
    for label, pattern in (
        ("openrouter_key", r"sk-or-v1-[A-Za-z0-9]{16,}"),
        ("openai_key", r"sk-(?!or-v1-)[A-Za-z0-9]{20,}"),
        ("github_token", r"gh[pousr]_[A-Za-z0-9]{20,}"),
        ("gitlab_token", r"glpat-[A-Za-z0-9_-]{20,}"),
        ("aws_access_key", r"AKIA[0-9A-Z]{12,}"),
        ("google_api_key", r"AIza[0-9A-Za-z_-]{30,}"),
        ("slack_token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
        ("stripe_key", r"sk_live_[A-Za-z0-9]{16,}"),
        ("private_key", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
        ("jwt", r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}"),
        ("bearer_token", r"Bearer\s+[A-Za-z0-9._~+/-]{20,}={0,2}"),
    )
)

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern))
    for label, pattern in (
        ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        ("iban", r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4,30}\b"),
    )
)

KIND_SECRET = "secret"
KIND_PII = "pii"


@dataclass(frozen=True)
class PrivacyFinding:
    kind: str
    label: str
    masked: str
    field: str = ""

    def describe(self) -> str:
        return f"{self.field or '<text>'}:{self.label}:{self.masked}"


def mask(value: str) -> str:
    """Maskiert einen Treffer, damit der Befund selbst nichts ausgibt."""

    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:6]}…{value[-4:]}"


def scan_text(text: str, field: str = "") -> list[PrivacyFinding]:
    findings: list[PrivacyFinding] = []

    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                PrivacyFinding(
                    kind=KIND_SECRET,
                    label=label,
                    masked=mask(match.group(0)),
                    field=field,
                )
            )

    for label, pattern in PII_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                PrivacyFinding(
                    kind=KIND_PII,
                    label=label,
                    masked=mask(match.group(0)),
                    field=field,
                )
            )

    return findings


def scan_row(
    row: Mapping[str, object],
    fields: Iterable[str] = ("prompt", "response"),
) -> list[PrivacyFinding]:
    findings: list[PrivacyFinding] = []

    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value:
            findings.extend(scan_text(value, field=field))

    return findings


def summarize(findings: Iterable[PrivacyFinding]) -> dict[str, int]:
    counts = {KIND_SECRET: 0, KIND_PII: 0}

    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1

    return counts
