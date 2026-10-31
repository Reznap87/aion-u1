from pathlib import Path

import pytest

from aion_u1.training.config import load_sft_config
from aion_u1.training.data import (
    select_canonical,
    split_rows,
)


def make_row(
    teacher: str,
    score: float,
    *,
    approved: bool = False,
) -> dict:
    return {
        "id": f"aion-{teacher}",
        "case_id": "case-1",
        "messages": [
            {
                "role": "user",
                "content": "Erstelle einen Plan.",
            },
            {
                "role": "assistant",
                "content": f"Plan von {teacher}",
            },
        ],
        "capabilities": [
            "reasoning.logical"
        ],
        "provenance": {
            "source": "teacher_generated",
            "license": (
                "reviewed-test-license"
                if approved
                else "unreviewed"
            ),
            "training_approved": approved,
            "teacher_alias": teacher,
            "teacher_model_id": (
                f"provider/{teacher}"
            ),
        },
        "verification": {
            "status": "verified",
            "score": score,
            "method": "test",
        },
    }


def test_selects_highest_scoring_teacher() -> None:
    selected = select_canonical(
        [
            make_row("teacher-a", 0.6),
            make_row("teacher-b", 0.95),
        ]
    )

    assert len(selected) == 1

    provenance = selected[0]["provenance"]

    assert (
        provenance["selected_teacher_alias"]
        == "teacher-b"
    )

    assert provenance[
        "candidate_teacher_aliases"
    ] == [
        "teacher-a",
        "teacher-b",
    ]


def test_require_approved_excludes_unreviewed_data() -> None:
    selected = select_canonical(
        [
            make_row(
                "teacher-a",
                0.99,
            ),
            make_row(
                "teacher-b",
                0.8,
                approved=True,
            ),
        ],
        require_approved=True,
    )

    assert len(selected) == 1

    assert (
        selected[0]["provenance"][
            "selected_teacher_alias"
        ]
        == "teacher-b"
    )

    assert selected[0]["provenance"][
        "training_approved"
    ] is True


def test_split_has_no_group_overlap() -> None:
    rows = select_canonical(
        [
            make_row("teacher-a", 0.8),
            {
                **make_row("teacher-c", 0.8),
                "id": "aion-teacher-c",
                "case_id": "case-2",
            },
        ]
    )

    train_rows, eval_rows, stats = split_rows(
        rows,
        eval_fraction=0.5,
        seed=42,
    )

    assert stats["group_overlap"] == 0

    assert not (
        {
            row["group_id"]
            for row in train_rows
        }
        & {
            row["group_id"]
            for row in eval_rows
        }
    )


def test_project_sft_config_is_valid() -> None:
    root = Path(
        __file__
    ).resolve().parents[1]

    config = load_sft_config(
        root
        / "training"
        / "configs"
        / "sft_lora.yaml"
    )

    assert config.peft.enabled is True
    assert (
        config.data.require_training_approval
        is True
    )
    assert config.training.num_train_epochs > 0


def test_invalid_split_fraction_is_rejected() -> None:
    with pytest.raises(ValueError):
        split_rows(
            [],
            eval_fraction=1.0,
        )
