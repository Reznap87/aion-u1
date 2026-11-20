"""`verify-teachers` muss einen Toten Eintrag kippen.

`cohere/north-mini-code` stand in `configs/teachers.free.yaml`, obwohl es
im Provider-Katalog nie existierte — und `verify-teachers` meldete trotzdem
Erfolg, weil der Befehl immer `0` zurckgab und die Free-Config nie
pruefte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aion_u1.cli import command_verify_teachers
from aion_u1.models import TeacherConfig
from aion_u1.teachers.resolver import TeacherResolution


class _FakeClient:
    async def close(self) -> None:
        return None


def _config(alias: str, *ids: str) -> TeacherConfig:
    return TeacherConfig(
        alias=alias,
        display_name=alias,
        provider="openrouter",
        candidate_model_ids=list(ids),
        model_id_env=None,
        api_key_env="OPENROUTER_API_KEY",
    )


def _args(tmp_path: Path, teachers: str | None = None):
    class A:
        output = str(tmp_path / "out.json")
        pass

    A.teachers = teachers
    return A


def test_unverified_teacher_gibt_eins_zuruck(
    monkeypatch, tmp_path, capsys
):
    cfg = _config("north_mini_code", "cohere/north-mini-code:free")
    monkeypatch.setattr(
        "aion_u1.cli.load_teachers", lambda path: [cfg]
    )

    async def fake_resolve(configs):
        return _FakeClient(), [
            TeacherResolution(
                config=cfg,
                model_id=None,
                status="unverified",
                source="catalog",
                suggestions=("cohere/command-a",),
            )
        ]

    monkeypatch.setattr("aion_u1.cli.resolve_all", fake_resolve)

    assert command_verify_teachers(_args(tmp_path)) == 1

    err = capsys.readouterr().err
    assert "north_mini_code" in err
    assert "0/1 Teacher verifiziert." in err


def test_verified_teacher_gibt_null_zuruck(
    monkeypatch, tmp_path, capsys
):
    cfg = _config("ernie", "baidu/ernie-5.5-flash-preview:free")
    monkeypatch.setattr(
        "aion_u1.cli.load_teachers", lambda path: [cfg]
    )

    async def fake_resolve(configs):
        return _FakeClient(), [
            TeacherResolution(
                config=cfg,
                model_id="baidu/ernie-5.5-flash-preview:free",
                status="verified",
                source="catalog",
            )
        ]

    monkeypatch.setattr("aion_u1.cli.resolve_all", fake_resolve)

    assert command_verify_teachers(_args(tmp_path)) == 0
    assert "1/1 Teacher verifiziert." in capsys.readouterr().err


def test_teachers_schalter_wird_verwendet(monkeypatch, tmp_path):
    """Ohne `--teachers` gilt die Ziel-Config, mit ihm die angegebene."""
    gesehen: list[str] = []

    def fake_load(path):
        gesehen.append(str(path))
        return []

    async def fake_resolve(configs):
        return _FakeClient(), []

    monkeypatch.setattr("aion_u1.cli.load_teachers", fake_load)
    monkeypatch.setattr("aion_u1.cli.resolve_all", fake_resolve)

    command_verify_teachers(_args(tmp_path, teachers="configs/teachers.free.yaml"))
    assert gesehen[-1].replace("\\", "/").endswith(
        "configs/teachers.free.yaml"
    )

    command_verify_teachers(_args(tmp_path))
    assert gesehen[-1].replace("\\", "/").endswith(
        "configs/teachers.yaml"
    )


@pytest.mark.parametrize("status", ["verified", "configured"])
def test_nur_unverified_kippt(status, monkeypatch, tmp_path):
    """`configured` ist ein bewusster Env-Override — kein Grund, zu failen."""
    cfg = _config("xx", "irgendwas/modell")
    monkeypatch.setattr("aion_u1.cli.load_teachers", lambda p: [cfg])

    async def fake_resolve(configs):
        return _FakeClient(), [
            TeacherResolution(
                config=cfg,
                model_id="irgendwas/modell",
                status=status,
                source="environment",
            )
        ]

    monkeypatch.setattr("aion_u1.cli.resolve_all", fake_resolve)

    assert command_verify_teachers(_args(tmp_path)) == 0
