"""Prueft die Open-Weight-Lehrerliste."""

from pathlib import Path

import yaml

ERLAUBTE_LIZENZEN = {"Apache-2.0", "MIT", "nvidia-open-model-license"}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _eintraege() -> list[dict]:
    roh = yaml.safe_load(
        (_root() / "configs" / "teachers.open.yaml").read_text(encoding="utf-8")
    )
    return roh["teachers"]


def test_alle_lehrer_tragen_eine_gepruefte_lizenz() -> None:
    """Ohne gesetzte Lizenz gaebe es keine Freigabe-Basis (AGENTS.md)."""

    eintraege = _eintraege()
    assert len(eintraege) >= 8
    for eintrag in eintraege:
        assert eintrag["license"] in ERLAUBTE_LIZENZEN, (
            f"{eintrag['alias']}: ungepruefte Lizenz {eintrag['license']!r}"
        )
        assert eintrag["license"] != "unreviewed"
        assert eintrag["candidate_model_ids"], f"{eintrag['alias']}: keine Modell-ID"
        assert eintrag["provider"] == "openrouter"


def test_open_liste_ist_disjunkt_zur_cloud_liste() -> None:
    """Nur so bleibt klar, welche Daten aus welcher Lizenz stammen."""

    cloud = yaml.safe_load(
        (_root() / "configs" / "teachers.yaml").read_text(encoding="utf-8")
    )["teachers"]
    cloud_aliase = {e["alias"] for e in cloud}
    cloud_modelle = {mid for e in cloud for mid in e["candidate_model_ids"]}

    for eintrag in _eintraege():
        assert eintrag["alias"] not in cloud_aliase
        assert set(eintrag["candidate_model_ids"]).isdisjoint(cloud_modelle), (
            f"{eintrag['alias']}: Modell kommt auch in der Cloud-Liste vor"
        )


def test_lizenzfeld_ist_im_schema_verankert() -> None:
    from aion_u1.models import TeacherConfig

    assert "license" in TeacherConfig.model_fields
    assert TeacherConfig.model_fields["license"].default == "unreviewed"
