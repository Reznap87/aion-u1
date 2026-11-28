"""Model artifact identity without executing model files or calling a provider."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def artifact_digest(directory: Path) -> str:
    """Hash file paths and contents in deterministic order, using bounded memory."""
    if not directory.is_dir():
        raise ValueError("Expected a local model artifact directory")
    entries = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if ".git" in relative.parts:
            continue
        if path.is_symlink() and path.is_dir():
            raise ValueError("Directory symlinks are not supported in model artifacts")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Model artifact contains a non-regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        entries.append([relative.as_posix(), digest.hexdigest()])
    if not entries:
        raise ValueError("Model artifact directory is empty")
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def model_identity(base_model: str, revision: str | None, adapter: str | None) -> dict[str, Any]:
    path = Path(base_model)
    if path.is_dir():
        if revision is not None:
            raise ValueError("Local model directories do not accept a Hub revision")
        base = {"kind": "local", "sha256": artifact_digest(path)}
    elif revision is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Hub revision must be a full commit SHA")
        base = {"kind": "hub", "repo_id": base_model, "revision": revision}
    else:
        # Legacy evaluation remains usable, but cannot establish comparable identity.
        base = {"kind": "unpinned", "repo_id": base_model}
    adapter_identity = {"sha256": artifact_digest(Path(adapter))} if adapter else None
    return {"base": base, "adapter": adapter_identity}


def validate_identity(identity: Any, base_model: Any, adapter: Any) -> dict[str, Any]:
    if not isinstance(identity, dict) or set(identity) != {"base", "adapter"}:
        raise ValueError("Student model identity required")
    base = identity["base"]
    if not isinstance(base, dict):
        raise ValueError("Student base model identity required")
    if base.get("kind") == "local":
        valid = set(base) == {"kind", "sha256"} and _digest(base.get("sha256"), 64)
    elif base.get("kind") == "hub":
        valid = (set(base) == {"kind", "repo_id", "revision"}
                 and isinstance(base_model, str) and bool(base_model)
                 and base.get("repo_id") == base_model
                 and _digest(base.get("revision"), 40))
    else:
        valid = False
    if not valid:
        raise ValueError("Student base model must have immutable identity")
    entry = identity["adapter"]
    if adapter is None:
        valid = entry is None
    else:
        valid = (isinstance(entry, dict) and set(entry) == {"sha256"}
                 and _digest(entry.get("sha256"), 64))
    if not valid:
        raise ValueError("Student adapter identity does not match report")
    return identity


def _digest(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None
