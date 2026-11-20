from __future__ import annotations

import hashlib
from pathlib import Path

from aion_u1 import __version__
from aion_u1.models import AuditReport
from aion_u1.provenance import collect, git_commit, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_sha256_file_berechnet_hash(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"AION")

    assert sha256_file(target) == hashlib.sha256(b"AION").hexdigest()


def test_sha256_file_fehler_fuehrt_zu_none(tmp_path: Path) -> None:
    assert sha256_file(tmp_path / "fehlt.txt") is None


def test_git_commit_liefert_sha() -> None:
    commit = git_commit(ROOT)

    if (ROOT / ".git").exists():
        assert commit is not None and len(commit) == 40
    else:
        assert commit is None


def test_collect_traegt_code_und_konfiguration() -> None:
    config = ROOT / "configs" / "capabilities.yaml"

    provenance = collect(
        ROOT,
        {"capabilities": config},
        command=["audit", "--suite", "smoke.yaml"],
    )

    assert provenance["package_version"] == __version__
    assert provenance["python"]
    assert provenance["command"] == ["audit", "--suite", "smoke.yaml"]
    assert provenance["inputs"]["capabilities"]["sha256"] == (
        hashlib.sha256(config.read_bytes()).hexdigest()
    )
    assert provenance["git_commit"] is None or len(
        provenance["git_commit"]
    ) == 40


def test_report_ohne_provenance_weiterhin_gueltig() -> None:
    """Aeltere Reports muessen ohne das neue Feld gelesen werden."""

    report = AuditReport.model_validate(
        {
            "generated_at": "2026-11-17T00:00:00+00:00",
            "suite": "smoke",
            "suite_path": "smoke.yaml",
            "teacher_resolutions": [],
            "results": [],
            "scores": {},
        }
    )

    assert report.provenance is None


def test_git_commit_worktree_nutzt_den_common_dir(tmp_path):
    """Refs eines Worktrees liegen im Common-Dir des Haupt-Repos."""

    sha = "b" * 40
    main_git = tmp_path / "main" / ".git"
    (main_git / "refs" / "heads").mkdir(parents=True)
    (main_git / "refs" / "heads" / "main").write_text(
        sha + "\n", encoding="utf-8"
    )

    wt_git = main_git / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text(
        "ref: refs/heads/main\n", encoding="utf-8"
    )
    (wt_git / "commondir").write_text("../..", encoding="utf-8")

    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(
        f"gitdir: {wt_git}\n", encoding="utf-8"
    )

    assert git_commit(worktree) == sha
