from pathlib import Path

import pytest

from aion_u1.capabilities import CapabilityRegistry


def test_loads_and_validates_capabilities(tmp_path: Path) -> None:
    path = tmp_path / "capabilities.yaml"
    path.write_text(
        """
capabilities:
  - id: base.skill
    name: Base
    description: Basis
    weight: 0.6
  - id: advanced.skill
    name: Advanced
    description: Erweitert
    weight: 0.4
    parents:
      - base.skill
""",
        encoding="utf-8",
    )

    registry = CapabilityRegistry.load(path)

    assert registry.total_weight == pytest.approx(1.0)
    assert registry.get("advanced.skill").parents == ["base.skill"]


def test_rejects_unknown_parent(tmp_path: Path) -> None:
    path = tmp_path / "capabilities.yaml"
    path.write_text(
        """
capabilities:
  - id: child.skill
    name: Child
    description: Kind
    weight: 1.0
    parents:
      - missing.skill
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unbekannte"):
        CapabilityRegistry.load(path)
