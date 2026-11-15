"""Exclusive, append-only candidate storage with a durable row boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class CandidateCheckpoint:
    def __init__(self, path: Path, *, resume: bool = False):
        self.path = path
        self.resume = resume
        self.rows: list[dict[str, Any]] = []
        self.lock_path = path.with_name(path.name + ".lock")
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("Kandidatendatei darf kein Symlink sein.")
        try:
            lock = self.lock_path.open("x", encoding="utf-8")
        except FileExistsError:
            raise ValueError("Kandidatendatei ist gesperrt; laufenden Writer prüfen.") from None
        lock.close()
        try:
            if self.resume:
                raw = self.path.read_bytes()
                if raw and not raw.endswith(b"\n"):
                    raise ValueError("Checkpoint endet mit einer unvollständigen Zeile.")
                for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line, object_pairs_hook=_unique)
                        if not isinstance(row, dict):
                            raise ValueError
                    except ValueError:
                        raise ValueError(f"Ungültiger Checkpoint in Zeile {number}.") from None
                    self.rows.append(row)
                self._handle = self.path.open("a", encoding="utf-8", newline="\n")
            else:
                self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        except BaseException:
            self.lock_path.unlink()
            raise
        return self

    def append(self, row: dict[str, Any]) -> None:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        self._handle.write(encoded)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.rows.append(row)

    def __exit__(self, *args):
        try:
            if self._handle is not None:
                self._handle.close()
        finally:
            self.lock_path.unlink()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result
