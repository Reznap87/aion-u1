"""Konsistenz zwischen kostenloser Teacher-Konfiguration und Task-Bank."""

from pathlib import Path

import yaml


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _aliase(pfad: Path) -> set[str]:
    roh = yaml.safe_load(pfad.read_text(encoding="utf-8"))
    return {eintrag["alias"] for eintrag in roh["teachers"]}


def test_freie_task_bank_nutzt_nur_konfigurierte_aliase() -> None:
    """Eine Task, die einen unbekannten Alias nennt, waere ein stiller No-Op."""

    auswahl = _aliase(_root() / "configs" / "teachers.free.yaml")
    bank = yaml.safe_load(
        (_root() / "training" / "configs" / "tasks" / "pilot_free.yaml")
        .read_text(encoding="utf-8")
    )

    assert bank["tasks"], "Freie Task-Bank ist leer."
    for task in bank["tasks"]:
        erwartet = set(task["teacher_aliases"])
        unbekannt = erwartet - auswahl
        assert unbekannt == set(), (
            f"Task {task['id']}: Aliase ohne Lehrer in teachers.free.yaml: "
            f"{sorted(unbekannt)}"
        )
        assert erwartet, f"Task {task['id']} ohne Lehrer-Aliase."


def test_freie_konfiguration_tragt_die_provider_und_schluessel() -> None:
    eintraege = yaml.safe_load(
        (_root() / "configs" / "teachers.free.yaml").read_text(encoding="utf-8")
    )["teachers"]

    assert len(eintraege) >= 10
    for eintrag in eintraege:
        assert eintrag["provider"] == "openrouter"
        assert eintrag["api_key_env"] == "OPENROUTER_API_KEY"
        assert eintrag["candidate_model_ids"], f"{eintrag['alias']}: keine Modell-ID"


def test_freie_bank_deckt_alles_ausser_multimodal_ab() -> None:
    """Die freien Textmodelle koennen keine Bilder; genau eine Capability
    bleibt deshalb offen — und nur diese."""

    from aion_u1.capabilities import CapabilityRegistry

    registry = CapabilityRegistry.load(_root() / "configs" / "capabilities.yaml")
    bank = yaml.safe_load(
        (_root() / "training" / "configs" / "tasks" / "pilot_free.yaml")
        .read_text(encoding="utf-8")
    )
    abgedeckt = {task["capability"] for task in bank["tasks"]}
    offen = {spec.id for spec in registry.list()} - abgedeckt
    assert offen == {"multimodal.grounding"}, (
        f"Unerwartete Luecke in der freien Bank: {sorted(offen)}"
    )


def test_freie_task_bank_ist_disjunkt_zur_ziel_task_bank() -> None:
    """Die freie Bank darf die Ziel-Tasks nicht umbenennen — sonst entsteht
    Verwechslung, welche Daten aus welcher Quelle stammen."""

    ziel = yaml.safe_load(
        (_root() / "training" / "configs" / "tasks" / "pilot.yaml")
        .read_text(encoding="utf-8")
    )
    frei = yaml.safe_load(
        (_root() / "training" / "configs" / "tasks" / "pilot_free.yaml")
        .read_text(encoding="utf-8")
    )
    ziel_ids = {task["id"] for task in ziel["tasks"]}
    frei_ids = {task["id"] for task in frei["tasks"]}
    assert ziel_ids.isdisjoint(frei_ids)
