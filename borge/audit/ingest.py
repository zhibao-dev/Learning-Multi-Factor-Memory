"""Canonical-JSON ingest for ``borge audit``.

Reads a memory dump (a JSON list of records) into typed ``MemoryRecord``
objects. Pure parse: no embedding, no DB, no scoring. Rows missing or with
empty ``id`` / ``text`` / ``timestamp`` are skipped with a warning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    id: str
    text: str
    timestamp: str
    role: str = "user"
    metadata: dict = field(default_factory=dict)


def load_dump(path: str | Path) -> list[MemoryRecord]:
    """Load a canonical-JSON memory dump into ``MemoryRecord`` objects.

    Each row must carry a non-empty ``id``, ``text`` and ``timestamp``;
    rows that don't are skipped with a ``logging.warning``.
    """
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)

    records: list[MemoryRecord] = []
    for i, row in enumerate(rows):
        rec_id = row.get("id")
        text = row.get("text")
        timestamp = row.get("timestamp")
        if not rec_id or not text or not timestamp:
            logger.warning(
                "skipping malformed memory row %d (id=%r): missing id/text/timestamp",
                i,
                rec_id,
            )
            continue
        records.append(
            MemoryRecord(
                id=rec_id,
                text=text,
                timestamp=timestamp,
                role=row.get("role", "user"),
                metadata=row.get("metadata") or {},
            )
        )
    return records
