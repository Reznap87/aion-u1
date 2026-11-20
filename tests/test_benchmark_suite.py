"""Struktur-Tests der vollständigen Capability-Suite."""

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
    return _root() / "evaluations" / "capability_suite" / "benchmark_v1.yaml"


def test_suite_covert_jede_capability_mindestens_dreifach() -> None:
    """AGENTS.md: jede Kernfähigkeit braucht eigene Evaluationsabdeckung."""

    registry = CapabilityRegistry.load(
        _root() / "configs" / "capabilities.yaml"
    )
    cases = load_cases(_suite())

    zaehler: dict[str, int] = {}
    for case in cases:
        zaehler[case.capability] = zaehler.get(case.capability, 0) + 1

    erwartet = {spec.id for spec in registry.list()}
    assert set(zaehler) == erwartet, (
        f"Abdeckung weicht ab: fehlend={sorted(erwartet - set(zaehler))}, "
        f"fremd={sorted(set(zaehler) - erwartet)}"
    )
    duenn = {name: anzahl for name, anzahl in zaehler.items() if anzahl < 3}
    assert duenn == {}, f"Zu wenig Tiefe je Capability: {duenn}"


def test_case_ids_sind_eindeutig() -> None:
    cases = load_cases(_suite())
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids)), "Doppelte Case-ID"


def test_verifier_sind_bekannt_und_vollstaendig() -> None:
    cases = load_cases(_suite())
    assert cases
    for case in cases:
        assert case.verifier.type in ERLAUBTE_VERIFIER, (
            f"{case.id}: unbekannter Verifier {case.verifier.type!r}"
        )
        if case.verifier.type in ("contains_all", "not_contains", "regex"):
            assert case.verifier.values, f"{case.id}: Verifier ohne Werte"
        if case.verifier.type == "json_equals":
            assert case.verifier.expected is not None, (
                f"{case.id}: json_equals ohne Erwartung"
            )


def test_regex_verifier_kompilieren() -> None:
    import re

    for case in load_cases(_suite()):
        if case.verifier.type != "regex":
            continue
        for pattern in case.verifier.values:
            re.compile(pattern)


def test_bildverweise_liegen_im_repo() -> None:
    for case in load_cases(_suite()):
        if case.modality != "image":
            continue
        assert case.image_path
        assert (_root() / case.image_path).is_file(), (
            f"Bild fehlt: {case.image_path}"
        )


def test_suite_teilt_keine_prompts_mit_der_pilot_task_bank() -> None:
    from aion_u1.training.tasks import load_task_bank

    cases = load_cases(_suite())
    prompts = {case.prompt.strip() for case in cases}
    bank = load_task_bank(
        _root() / "training" / "configs" / "tasks" / "pilot.yaml"
    )
    for task in bank.tasks:
        assert task.prompt.strip() not in prompts, (
            f"Leakage: Task {task.id} ist identisch mit einem Suite-Case"
        )


def test_yaml_datei_ist_wohlgeformt() -> None:
    with open(_suite(), encoding="utf-8") as fh:
        daten = yaml.safe_load(fh)
    assert daten["suite"] == "capability_benchmark_v1"
    assert len(daten["cases"]) >= 39
