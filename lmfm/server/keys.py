"""SQLite-backed API-key store with quota metering and usage logging.

Drop-in replacement for the original in-memory KeyStore.  The public
interface (``authorize`` / ``consume``) is unchanged so ``app.py`` and
existing tests that pass ``valid_keys=`` to ``create_app`` need no edits.

Thread safety: WAL journal mode + ``check_same_thread=False``.  All
writes go through a single ``threading.Lock`` so concurrent FastAPI
workers can't race on quota decrements.

Production usage::

    store = KeyStore("/var/lib/lmfm/keys.db")
    store.provision("sk-abc123", quota=100, label="customer-acme")

Test / in-memory usage (default)::

    store = KeyStore()          # :memory: — disappears when object is GC'd
    store.provision("k", 5)
    store.authorize("k")        # → "ok"
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


_DDL = """
CREATE TABLE IF NOT EXISTS api_keys (
    key        TEXT PRIMARY KEY,
    quota      INTEGER NOT NULL DEFAULT 0,
    label      TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS usage_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL,
    called_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    outcome    TEXT NOT NULL
);
"""


class KeyStore:
    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,   # autocommit
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DDL)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core interface (unchanged from the original in-memory version)
    # ------------------------------------------------------------------

    def authorize(self, key: str | None) -> str:
        """Return ``'ok'`` | ``'unauthorized'`` | ``'exhausted'``."""
        if key is None:
            return "unauthorized"
        with self._lock:
            row = self._conn.execute(
                "SELECT quota FROM api_keys WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return "unauthorized"
        return "ok" if row[0] > 0 else "exhausted"

    def consume(self, key: str) -> None:
        """Decrement quota by 1 and write a usage-log entry."""
        with self._lock:
            self._conn.execute(
                "UPDATE api_keys SET quota = quota - 1 WHERE key = ?", (key,)
            )
            self._conn.execute(
                "INSERT INTO usage_log (key, outcome) VALUES (?, 'ok')", (key,)
            )

    # ------------------------------------------------------------------
    # Admin helpers
    # ------------------------------------------------------------------

    def provision(self, key: str, quota: int, *, label: str = "") -> None:
        """Insert or update a key.  Existing quota is replaced, not added."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO api_keys (key, quota, label)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET quota = excluded.quota,
                                                  label = excluded.label""",
                (key, quota, label),
            )

    def revoke(self, key: str) -> bool:
        """Delete a key.  Returns ``True`` if the key existed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM api_keys WHERE key = ?", (key,)
            )
        return cur.rowcount > 0

    def remaining(self, key: str) -> int | None:
        """Return remaining quota, or ``None`` if the key doesn't exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT quota FROM api_keys WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row is not None else None

    def usage(self, key: str, *, limit: int = 100) -> list[dict]:
        """Return the most recent ``limit`` log entries for ``key``."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT called_at, outcome FROM usage_log "
                "WHERE key = ? ORDER BY id DESC LIMIT ?",
                (key, limit),
            ).fetchall()
        return [{"called_at": r[0], "outcome": r[1]} for r in rows]

    # ------------------------------------------------------------------
    # Backward-compatibility factory (for tests that pass valid_keys=)
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, valid_keys: dict, db_path: str = ":memory:") -> "KeyStore":
        """Seed a new store from a ``{key: {"quota": int}}`` dict."""
        store = cls(db_path)
        for key, meta in valid_keys.items():
            store.provision(key, meta.get("quota", 0), label=meta.get("label", ""))
        return store

    def close(self) -> None:
        self._conn.close()
