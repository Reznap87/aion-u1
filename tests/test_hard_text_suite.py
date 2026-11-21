"""Prueft die Text-Variante der schweren Suite."""

from pathlib import Path

import yaml

from aion_u1.capabilities import CapabilityRegistry
from aion_u1.cli import load_cases


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _suite() -> Path:
    return (
        _root()
        / "evaluations"
        / "capability_suite"
        / "benchmark_hard_text_v1.yaml"
    )


def test_text_suite_hat_keine_bilder() -> None:
    """Qwen2.5-1.5B-Instruct ist ein Textmodell; Bild-Faecke klammern die
    Coverage aus und verbieten den Index."""
    for case in load_cases(_suite()):
        assert case.modality != "image", case.id


def test_text_suite_deckt_zwoelf_capabilities_zweifach_ab() -> None:
    registry = CapabilityRegistry.load(
        _root() / "configs" / "capabilities.yaml"
    )
    text_capabilities = {
        spec.id
        for spec in registry.list()
        if spec.id != "multimodal.grounding"
    }
    zaehler: dict[str, int] = {}
    for case in load_cases(_suite()):
        zaehler[case.capability] = zaehler.get(case.capability, 0) + 1

    assert set(zaehler) == text_capabilities
    duenn = {name: n for name, n in zaehler.items() if n < 2}
    assert duenn == {}, f"Zu wenige Faelle: {duenn}"


def test_text_suite_kommt_aus_der_harten_suite() -> None:
    """Keine neuen Prompts — nur die Bild-Faelle wurden weggelassen."""
    hart = yaml.safe_load(
        (_root() / "evaluations" / "capability_suite" / "benchmark_hard_v1.yaml")
        .read_text(encoding="utf-8")
    )
    hart_ohne_bilder = [
        fall for fall in hart["cases"] if fall.get("modality") != "image"
    ]
    text = load_cases(_suite())
    assert [case.id for case in text] == [
        fall["id"] for fall in hart_ohne_bilder
    ]
