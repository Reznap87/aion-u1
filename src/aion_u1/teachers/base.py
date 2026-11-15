from __future__ import annotations

from typing import Protocol

from ..models import TeacherResponse


class TeacherError(RuntimeError):
    """Fehler bei der Kommunikation mit einem Teacher-Provider."""


class TeacherTransientError(TeacherError):
    """Vorübergehender Provider-Fehler, bei dem ein Retry sinnvoll sein kann."""


class TeacherClient(Protocol):
    async def list_models(self) -> list[str]:
        """Gibt die aktuell sichtbaren Provider-Modell-IDs zurück."""

    async def generate(
        self,
        model_id: str,
        messages: list[dict[str, object]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> TeacherResponse:
        """Führt eine Teacher-Anfrage aus."""
