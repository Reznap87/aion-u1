"""One-shot execution reservations on a shared persistent local volume."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


def plan_id(proposal: dict[str, Any]) -> str:
    encoded = json.dumps(proposal, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RunLedger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    plan_id TEXT PRIMARY KEY,
                    candidate_count INTEGER NOT NULL CHECK(candidate_count > 0),
                    state TEXT NOT NULL CHECK(state IN ('reserved', 'completed', 'failed')),
                    reserved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT
                )
            """)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def reserve(self, proposal: dict[str, Any]) -> str:
        identifier = plan_id(proposal)
        count = proposal["planned_candidates"]
        if type(count) is not int or count <= 0:
            raise ValueError("Positive candidate count required")
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "INSERT INTO runs (plan_id, candidate_count, state) VALUES (?, ?, 'reserved')",
                    (identifier, count),
                )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError("Plan already reserved; automatic replay denied") from exc
        return identifier

    def finish(self, identifier: str, *, success: bool) -> None:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE runs SET state=?, finished_at=CURRENT_TIMESTAMP WHERE plan_id=? AND state='reserved'",
                ("completed" if success else "failed", identifier),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Missing or finalized reservation")
