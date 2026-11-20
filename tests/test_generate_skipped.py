"""`generate_candidates` nennt, was es uebersprungen hat.

`--teachers <andere Config>` mit der Standard-Task-Bank liefert 0 Zeilen und
Exit 1. Ohne die neue `skipped`-Angabe sahe man nur die leere Datei.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aion_u1.models import TeacherConfig
from aion_u1.training import generate as mod
from aion_u1.training.generate import generate_candidates
from aion_u1.training.tasks import TaskBank, TaskSpec, TaskVerifier

SCHLUESSEL = "GENERATE_SKIPPED_TEST_KEY"


class _FakeClient:
    """Nur `list_models`; ohne Jobs wird nichts generiert."""

    def __init__(self, catalog):
        self._catalog = catalog
        self.geschlossen = False

    async def list_models(self):
        return list(self._catalog)

    async def close(self):
        self.geschlossen = True


def _teacher(alias: str, model_id: str) -> TeacherConfig:
    return TeacherConfig(
        alias=alias,
        display_name=alias,
        provider="openrouter",
        candidate_model_ids=[model_id],
        model_id_env=None,
        api_key_env=SCHLUESSEL,
    )


def _bank(*aliases: str) -> TaskBank:
    return TaskBank(
        tasks=[
            TaskSpec(
                id="t_eins",
                capability="code.synthesis",
                prompt="Schreibe etwas.",
                verifier=TaskVerifier(
                    type="contains_all",
                    values=["etwas"],
                ),
                teacher_aliases=list(aliases),
            )
        ]
    )


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setenv(SCHLUESSEL, "platzhalter")

    def fabrik(**_kwargs):
        return _FakeClient(["erwin/alpha"])

    monkeypatch.setattr(mod, "OpenRouterClient", fabrik)
    return _FakeClient


def test_nur_unbekanntes_alias(fake_client):
    rows, errors, _res, skipped = asyncio.run(
        generate_candidates(
            [_teacher("erwin", "erwin/alpha")],
            _bank("garnicht_da"),
            root=Path("."),
            concurrency=1,
        )
    )

    assert rows == []
    assert errors == []
    assert skipped == {
        "unbekannte_aliase": ["garnicht_da"],
        "tasks_ohne_lehrer": ["t_eins"],
    }


def test_unbekanntes_alias_neben_einem_gueltigen(fake_client):
    """Der gueltige Alias rettet den Task vor der Task-Meldung, der
    unbekannte wird trotzdem genannt."""

    _rows, _errors, _res, skipped = asyncio.run(
        generate_candidates(
            [_teacher("erwin", "erwin/alpha")],
            _bank("erwin", "garnicht_da"),
            root=Path("."),
            concurrency=1,
        )
    )

    assert skipped["unbekannte_aliase"] == ["garnicht_da"]
    assert skipped["tasks_ohne_lehrer"] == []


def test_task_ohne_lehrer_wird_geradezig_gemeldet(fake_client):
    rows, _errors, _resolutions, skipped = asyncio.run(
        generate_candidates(
            [_teacher("erwin", "erwin/alpha")],
            _bank("garnicht_da"),
            root=Path("."),
            concurrency=1,
        )
    )

    assert rows == []
    assert skipped["unbekannte_aliase"] == ["garnicht_da"]
    assert skipped["tasks_ohne_lehrer"] == ["t_eins"]


def test_verifizierter_lehrer_bleibt_silent(fake_client):
    """Passt Alias und Config, gibt es nichts zu melden."""

    _rows, _errors, _resolutions, skipped = asyncio.run(
        generate_candidates(
            [_teacher("erwin", "erwin/alpha")],
            _bank("erwin"),
            root=Path("."),
            concurrency=1,
        )
    )

    # Jobs existieren, also lief die eigentliche Generierung (hier ohne
    # Netz: der Fake antwortet nicht, die Fehler landen in `errors`).
    assert skipped == {
        "unbekannte_aliase": [],
        "tasks_ohne_lehrer": [],
    }
