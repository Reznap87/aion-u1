"""Entfernt Reasoning-Bloecke aus Teacher-Antworten.

AGENTS.md: "Private oder geschützte Chain-of-Thought-Daten nicht uebernehmen."
Manche Lehrermodelle geben ihren Loesungsweg inline in Tags aus. Der Loesungsweg
ist Teil der Messung, aber nicht Teil des Trainingslabels — er wird hier
entfernt, bevor die Antwort zum Datensatz wird.
"""

from __future__ import annotations

import re

# Bewusst nur die etablierten Reasoning-Tags; `<analysis>` o. ae. kommen in
# normalem Code vor und würden sonst Daten zerstoeren.
REASONING_BLOCK = re.compile(
    r"<(?:think|thinking|reasoning|thoughts)\b[^>]*>.*?"
    r"</(?:think|thinking|reasoning|thoughts)>",
    re.IGNORECASE | re.DOTALL,
)


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Entfernt Reasoning-Bloecke und sagt, ob etwas entfernt wurde."""

    cleaned, count = REASONING_BLOCK.subn("", text)
    return cleaned.strip(), count > 0
