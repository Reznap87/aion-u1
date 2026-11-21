"""Provenienz des Trainings-Manifests.

`_git_commit` darf nicht ueber `subprocess` laufen: auf Windows wird das
Git-Binary im Subprozess mit `OSError: [WinError 5]` abgelehnt, und das
Manifest bekam still `git_commit: null`.
"""

from pathlib import Path

from aion_u1.training.sft import _git_commit, _local_weights_digest

SHA = "a" * 40


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    return root


def test_git_commit_liest_den_aktuellen_commit(tmp_path):
    root = _repo(tmp_path)
    (root / ".git" / "HEAD").write_text(
        "ref: refs/heads/main\n", encoding="utf-8"
    )
    (root / ".git" / "refs" / "heads" / "main").write_text(
        SHA + "\n", encoding="utf-8"
    )

    assert _git_commit(root) == SHA


def test_git_commit_detached_head(tmp_path):
    root = _repo(tmp_path)
    (root / ".git" / "HEAD").write_text(SHA, encoding="utf-8")

    assert _git_commit(root) == SHA


def test_git_commit_aus_packed_refs(tmp_path):
    root = _repo(tmp_path)
    (root / ".git" / "HEAD").write_text(
        "ref: refs/heads/main\n", encoding="utf-8"
    )
    (root / ".git" / "packed-refs").write_text(
        f"# pack-refs\n{SHA} refs/heads/main\n",
        encoding="utf-8",
    )

    assert _git_commit(root) == SHA


def test_git_commit_worktree_datei(tmp_path):
    root = _repo(tmp_path)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    git_dir = root / ".git"
    (git_dir / "HEAD").write_text(
        "ref: refs/heads/main\n", encoding="utf-8"
    )
    (git_dir / "refs" / "heads" / "main").write_text(
        SHA + "\n", encoding="utf-8"
    )
    (worktree / ".git").write_text(
        f"gitdir: {git_dir}\n", encoding="utf-8"
    )

    assert _git_commit(worktree) == SHA


def test_git_commit_ohne_repo(tmp_path):
    assert _git_commit(tmp_path) is None


def test_git_commit_leere_head_datei(tmp_path):
    root = _repo(tmp_path)
    (root / ".git" / "HEAD").write_text("", encoding="utf-8")

    assert _git_commit(root) is None


def test_git_commit_worktree_refs_liegen_im_common_dir(tmp_path):
    """In einem Worktree stehen refs/ und packed-refs im Common-Dir."""

    main_git = tmp_path / "main" / ".git"
    (main_git / "refs" / "heads").mkdir(parents=True)
    (main_git / "refs" / "heads" / "main").write_text(
        SHA + "\n", encoding="utf-8"
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

    assert _git_commit(worktree) == SHA


def _erwartete_summe(dateien: dict[str, bytes]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(dateien):
        digest.update(name.encode("utf-8"))
        digest.update(dateien[name])
    return digest.hexdigest()


def test_weights_summe_ist_stabil(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"gewicht")

    assert _local_weights_digest(str(tmp_path)) == _erwartete_summe(
        {"model.safetensors": b"gewicht"}
    )


def test_weights_summe_ist_unabhaengig_von_der_reihenfolge(tmp_path):
    """Die Dateien werden sortiert gelesen — die Summe muss eindeutig sein."""

    inhalt = {
        "a.safetensors": b"erstens",
        "b.safetensors": b"zweitens",
    }
    for name, data in inhalt.items():
        (tmp_path / name).write_bytes(data)

    assert _local_weights_digest(str(tmp_path)) == _erwartete_summe(inhalt)


def test_weights_summe_ignoriert_andere_dateien(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"gewicht")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert _local_weights_digest(str(tmp_path)) == _erwartete_summe(
        {"model.safetensors": b"gewicht"}
    )


def test_weights_summe_bei_hub_modell_id():
    """Eine Hub-ID ist kein Verzeichnis — nichts zu hashen."""

    assert _local_weights_digest("Qwen/Qwen2.5-1.5B-Instruct") is None


def test_weights_summe_ohne_gewichte(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert _local_weights_digest(str(tmp_path)) is None
