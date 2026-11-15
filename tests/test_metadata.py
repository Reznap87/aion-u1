import tomllib
from pathlib import Path

import aion_u1


def test_version_matches_pyproject() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert aion_u1.__version__ == pyproject["project"]["version"]
