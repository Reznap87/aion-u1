"""Provenienz fuer reproduzierbare Auswertungen.

GOAL.md, Definition of Done Punkt 7: "ein reproduzierbarer Evaluationsbericht
vorliegt." Ein Report ueber Modellstaerken ist nur dann reproduzierbar, wenn
man nachlesen kann, mit welchem Code, welcher Konfiguration und welcher Suite
er entstanden ist.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

from . import __version__


def sha256_file(path: Path) -> str | None:
    """Hash einer Eingabedatei; None, wenn sie nicht lesbar ist."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _common_dir(dot_git: Path) -> Path:
    """Gitdir des Haupt-Repos.

    In einem Worktree liegen `refs/` und `packed-refs` nicht im Gitdir des
    Worktrees, sondern im Common-Dir, auf den `.git/worktrees/<name>/commondir`
    verweist.
    """

    commondir = dot_git / "commondir"

    if not commondir.is_file():
        return dot_git

    text = commondir.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not text:
        return dot_git

    return (dot_git / text).resolve()


def _read_ref(dot_git: Path, ref: str) -> str | None:
    """Löst einen Branch-Namen ueber die Ref-Dateien auf."""

    for base in (dot_git, _common_dir(dot_git)):
        ref_file = base / ref
        if ref_file.is_file():
            return ref_file.read_text(
                encoding="utf-8", errors="replace"
            ).strip() or None

        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "^")):
                    continue
                sha, _, refname = line.partition(" ")
                if refname == ref:
                    return sha

    return None


def git_commit(root: Path) -> str | None:
    """HEAD-Commit aus `.git` lesen.

    Absichtlich ohne `git`-Aufruf: unter Windows liegt git oft nur als
    WindowsApps-Alias vor, dessen direkte Ausfuehrung mit WinError 5
    ("Zugriff verweigert") scheitert. Der Commit steht in `.git/HEAD`.
    """

    dot_git = root / ".git"

    if dot_git.is_file():
        # Worktree: Inhalt ist "gitdir: <pfad>"
        text = dot_git.read_text(encoding="utf-8", errors="replace")
        text = text.strip()
        if not text.startswith("gitdir:"):
            return None
        dot_git = Path(text.split(":", 1)[1].strip())

    if not dot_git.is_dir():
        return None

    head = dot_git / "HEAD"
    if not head.is_file():
        return None

    content = head.read_text(
        encoding="utf-8", errors="replace"
    ).strip()

    if not content:
        return None

    if content.startswith("ref:"):
        return _read_ref(dot_git, content.split(":", 1)[1].strip())

    return content


def collect(
    root: Path,
    inputs: dict[str, Path],
    *,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Sammelt Code-, Konfigurations- und Eingabe-Provenienz."""

    return {
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(root),
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in inputs.items()
        },
        "command": list(command or []),
    }
