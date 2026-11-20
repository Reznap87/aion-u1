"""Struktur der zweiten Pilot-Runde."""

from pathlib import Path

import yaml

from aion_u1.capabilities import CapabilityRegistry
from aion_u1.cli import load_cases


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bank(name: str) -> dict:
    return yaml.safe_load(
        (_root() / "training" / "configs" / "tasks" / name).read_text(encoding="utf-8")
    )


def _aliase() -> set[str]:
    roh = yaml.safe_load(
        (_root() / "configs" / "teachers.yaml").read_text(encoding="utf-8")
    )
    return {eintrag["alias"] for eintrag in roh["teachers"]}


def test_v2_nutzt_nur_ziel_lehrer_aliase() -> None:
    auswahl = _aliase()
    for task in _bank("pilot_v2.yaml")["tasks"]:
        unbekannt = set(task["teacher_aliases"]) - auswahl
        assert unbekannt == set(), f"{task['id']}: {sorted(unbekannt)}"


def test_v2_ist_disjunkt_zur_ersten_runde() -> None:
    erste = {task["id"] for task in _bank("pilot.yaml")["tasks"]}
    zweite = {task["id"] for task in _bank("pilot_v2.yaml")["tasks"]}
    assert erste.isdisjoint(zweite)

    prompts_erste = {task["prompt"].strip() for task in _bank("pilot.yaml")["tasks"]}
    for task in _bank("pilot_v2.yaml")["tasks"]:
        assert task["prompt"].strip() not in prompts_erste, task["id"]


def test_v2_deckt_zwoelf_capabilities_ab() -> None:
    """Multimodal bleibt offen — die Lehrer liefern hier keine Bilder."""

    registry = CapabilityRegistry.load(_root() / "configs" / "capabilities.yaml")
    abgedeckt = {task["capability"] for task in _bank("pilot_v2.yaml")["tasks"]}
    offen = {spec.id for spec in registry.list()} - abgedeckt
    assert offen == {"multimodal.grounding"}, sorted(offen)


def test_v2_prompts_sind_keine_evaluationsprompts() -> None:
    """Ausbilden der Eval-Prompts als Trainingsdaten waere Data Leakage."""

    suite = load_cases(_root() / "evaluations" / "capability_suite" / "smoke.yaml")
    eval_prompts = {case.prompt.strip() for case in suite}
    for task in _bank("pilot_v2.yaml")["tasks"]:
        assert task["prompt"].strip() not in eval_prompts, task["id"]
