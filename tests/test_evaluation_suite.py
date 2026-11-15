from pathlib import Path

from aion_u1.capabilities import CapabilityRegistry
from aion_u1.cli import load_cases


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_audit_suite_covers_every_capability() -> None:
    """AGENTS.md: Jede Kernfähigkeit erhält eine eigene Evaluationsabdeckung."""

    registry = CapabilityRegistry.load(
        _root() / "configs" / "capabilities.yaml"
    )
    cases = load_cases(
        _root()
        / "evaluations"
        / "capability_suite"
        / "smoke.yaml"
    )

    covered = {case.capability for case in cases}
    missing = sorted(
        {spec.id for spec in registry.list()} - covered
    )

    assert missing == [], (
        f"Audit-Suite ohne Abdeckung: {missing}"
    )


def test_image_cases_reference_existing_assets() -> None:
    cases = load_cases(
        _root()
        / "evaluations"
        / "capability_suite"
        / "smoke.yaml"
    )

    image_cases = [
        case for case in cases if case.modality == "image"
    ]

    assert image_cases, "Die Suite braucht mindestens einen Image-Case."

    for case in image_cases:
        assert case.image_path
        assert (
            _root() / case.image_path
        ).is_file(), f"Bild fehlt: {case.image_path}"


def test_image_case_builds_multimodal_content() -> None:
    from aion_u1.audit import build_case_input

    cases = load_cases(
        _root()
        / "evaluations"
        / "capability_suite"
        / "smoke.yaml"
    )
    image_case = next(
        case for case in cases if case.modality == "image"
    )

    content = build_case_input(image_case, _root())

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == image_case.prompt
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_pilot_and_suite_share_no_prompts() -> None:
    """Trainings- und Eval-Items duerfen nicht identisch sein."""

    from aion_u1.training.tasks import load_task_bank

    root = _root()
    cases = load_cases(
        root
        / "evaluations"
        / "capability_suite"
        / "smoke.yaml"
    )
    bank = load_task_bank(
        root
        / "training"
        / "configs"
        / "tasks"
        / "pilot.yaml"
    )

    case_prompts = {
        case.prompt.strip() for case in cases
    }

    for task in bank.tasks:
        assert task.prompt.strip() not in case_prompts, (
            f"Leakage: Task {task.id} ist identisch mit einem Suite-Case"
        )
