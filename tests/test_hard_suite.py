"""Struktur-Tests der schweren Capability-Suite."""

from pathlib import Path

import yaml

from aion_u1.capabilities import CapabilityRegistry
from aion_u1.cli import load_cases

ERLAUBTE_VERIFIER = {
    "contains_all",
    "not_contains",
    "exact_match",
    "regex",
    "json_equals",
}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _suite() -> Path:
    return _root() / "evaluations" / "capability_suite" / "benchmark_hard_v1.yaml"


def test_hard_suite_deckt_jede_capability_mindestens_zweifach_ab() -> None:
    registry = CapabilityRegistry.load(_root() / "configs" / "capabilities.yaml")
    cases = load_cases(_suite())

    zaehler: dict[str, int] = {}
    for case in cases:
        zaehler[case.capability] = zaehler.get(case.capability, 0) + 1

    erwartet = {spec.id for spec in registry.list()}
    assert set(zaehler) == erwartet, (
        f"Abdeckung weicht ab: fehlend={sorted(erwartet - set(zaehler))}, "
        f"fremd={sorted(set(zaehler) - erwartet)}"
    )
    duenn = {name: n for name, n in zaehler.items() if n < 2}
    assert duenn == {}, f"Zu wenige Faelle je Capability: {duenn}"


def test_case_ids_sind_eindeutig() -> None:
    cases = load_cases(_suite())
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids)), "Doppelte Case-ID"


def test_verifier_sind_bekannt_und_vollstaendig() -> None:
    for case in load_cases(_suite()):
        assert case.verifier.type in ERLAUBTE_VERIFIER, case.id
        if case.verifier.type in ("contains_all", "not_contains", "regex"):
            assert case.verifier.values, f"{case.id}: Verifier ohne Werte"
        if case.verifier.type == "json_equals":
            assert case.verifier.expected is not None, f"{case.id}: ohne Erwartung"
        if case.verifier.type == "python_code":
            assert case.verifier.test_code, f"{case.id}: ohne Testcode"


def test_regex_verifier_kompilieren() -> None:
    import re

    for case in load_cases(_suite()):
        if case.verifier.type == "regex":
            for pattern in case.verifier.values:
                re.compile(pattern)


def test_bildverweise_liegen_im_repo() -> None:
    for case in load_cases(_suite()):
        if case.modality == "image":
            assert case.image_path
            assert (_root() / case.image_path).is_file(), case.image_path


def test_hard_prompts_sind_neu_gegenueber_den_leichten_suite_und_tasks() -> None:
    """Leakage: Eval-Prompts duerfen nicht aus Trainings- oder leichten
    Eval-Faecken stammen."""

    hart = {case.prompt.strip() for case in load_cases(_suite())}

    for bank in ("pilot.yaml", "pilot_v2.yaml", "pilot_free.yaml",
                 "pilot_open.yaml", "pilot_open_v2.yaml", "pilot_open_v3.yaml"):
        pfad = _root() / "training" / "configs" / "tasks" / bank
        if not pfad.is_file():
            continue
        tasks = yaml.safe_load(pfad.read_text(encoding="utf-8"))["tasks"]
        for task in tasks:
            assert task["prompt"].strip() not in hart, (bank, task["id"])

    for suite in ("smoke.yaml", "benchmark_v1.yaml"):
        pfad = _root() / "evaluations" / "capability_suite" / suite
        for case in load_cases(pfad):
            assert case.prompt.strip() not in hart, (suite, case.id)
