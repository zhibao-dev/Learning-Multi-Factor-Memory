"""Markdown ingest for ``borge audit``.

Target agents (Claude Code / Hermes / Codex) keep their memory in Markdown,
not canonical JSON. This module reads such a dump into the SAME
``MemoryRecord`` objects that :mod:`borge.audit.ingest` produces, so the
downstream value pipeline is untouched.

Splitting modes:

* ``"heading"`` — one record per ``##``..``######`` section.
* ``"bullet"`` — one record per top-level list item (``- ``/``* ``).
* ``"dated"`` — one record per dated entry (a ``# YYYY-MM-DD`` heading or a
  ``- [YYYY-MM-DD]`` bullet), with each entry's own date recovered as its
  timestamp.
* ``"auto"`` — detects the shape (dated entries → ``dated``; more top-level
  bullets than ``##`` headings → ``bullet``; otherwise ``heading``).

YAML-frontmatter parsing is hand-rolled (no pyyaml dependency); timestamps
follow a recovery chain (inline date → frontmatter → file mtime).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from borge.audit.ingest import MemoryRecord

_HEADING_RE = re.compile(r"^#{2,6}\s+")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_BULLET_RE = re.compile(r"^[-*]\s+")
_DATED_HEADING_RE = re.compile(r"^#{1,6}\s*(\d{4}-\d{2}-\d{2})")
_DATED_BULLET_RE = re.compile(r"^[-*]\s*\[?(\d{4}-\d{2}-\d{2})")
_GOAL_HEADING_RE = re.compile(
    r"\b(goal|task|objective|project|deadline|plan|agenda|todo|milestone|work)\b",
    re.IGNORECASE,
)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split optional ``---``-fenced YAML frontmatter off the top of ``text``.

    Returns ``(meta, body)``. Only simple ``key: value`` lines are parsed;
    ``tags: [a, b]`` becomes a list. Without frontmatter, returns ``({}, text)``.
    """
    if not text.startswith("---\n"):
        return {}, text

    lines = text.split("\n")
    meta: dict = {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:  # unterminated fence — treat as plain body
        return {}, text

    for line in lines[1:end]:
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            value = [v.strip() for v in inner.split(",") if v.strip()]
        meta[key] = value

    body = "\n".join(lines[end + 1 :])
    return meta, body


def _recover_ts(inline_date: str | None, meta: dict, path: str | Path) -> str:
    """Recover an ISO timestamp: inline date → frontmatter → file mtime."""
    if inline_date:
        return inline_date
    fm = meta.get("modified") or meta.get("created") or meta.get("timestamp")
    if fm:
        return fm
    return (
        datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _slug(heading_text: str) -> str:
    """Lowercase, non-alphanumeric → ``-``, collapse and trim dashes."""
    s = re.sub(r"[^a-z0-9]+", "-", heading_text.lower())
    return s.strip("-")


def _heading_split(body: str, meta: dict, path: str | Path) -> list[MemoryRecord]:
    """Split ``body`` into one ``MemoryRecord`` per heading section."""
    lines = body.split("\n")
    # Locate heading lines; everything before the first is the preamble.
    heading_idx = [i for i, ln in enumerate(lines) if _HEADING_RE.match(ln)]

    sections: list[tuple[str | None, list[str]]] = []
    first = heading_idx[0] if heading_idx else len(lines)
    preamble = lines[:first]
    if any(ln.strip() for ln in preamble):
        sections.append((None, preamble))
    for n, start in enumerate(heading_idx):
        end = heading_idx[n + 1] if n + 1 < len(heading_idx) else len(lines)
        sections.append((lines[start], lines[start:end]))

    stem = Path(path).stem
    role = meta.get("role", "user")
    tags = meta.get("tags", [])
    seen: dict[str, int] = {}
    records: list[MemoryRecord] = []

    for heading_line, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if not text:
            continue

        if heading_line is None:
            slug = "_preamble"
            heading_text = "_preamble"
            sec_role = role
        else:
            heading_text = _HEADING_RE.sub("", heading_line).strip()
            slug = _slug(heading_text) or "section"
            sec_role = "goal" if _GOAL_HEADING_RE.search(heading_text) else role

        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"

        m = _DATE_RE.search(text)
        inline = m.group(1) if m else None

        records.append(
            MemoryRecord(
                id=f"{stem}#{slug}",
                text=text,
                timestamp=_recover_ts(inline, meta, path),
                role=sec_role,
                metadata={
                    "source_file": str(path),
                    "heading_path": heading_text,
                    "tags": tags,
                },
            )
        )
    return records


def _dated_marker(line: str) -> str | None:
    """Return the ISO date if ``line`` starts a dated entry, else ``None``."""
    m = _DATED_HEADING_RE.match(line) or _DATED_BULLET_RE.match(line)
    return m.group(1) if m else None


def _bullet_split(body: str, meta: dict, path: str | Path) -> list[MemoryRecord]:
    """Split ``body`` into one ``MemoryRecord`` per top-level list item.

    A top-level item is a line matching ``^[-*]\\s+``; it absorbs any
    following indented/blank continuation lines until the next top-level
    bullet or heading.
    """
    lines = body.split("\n")
    starts = [
        i
        for i, ln in enumerate(lines)
        if _BULLET_RE.match(ln) and not ln[:1].isspace()
    ]

    stem = Path(path).stem
    role = meta.get("role", "user")
    tags = meta.get("tags", [])
    records: list[MemoryRecord] = []

    for n, start in enumerate(starts):
        # Absorb continuation lines until the next top-level bullet or heading.
        end = start + 1
        next_start = starts[n + 1] if n + 1 < len(starts) else len(lines)
        while end < next_start and not _HEADING_RE.match(lines[end]):
            end += 1
        text = "\n".join(lines[start:end]).strip()
        if not text:
            continue

        m = _DATE_RE.search(text)
        inline = m.group(1) if m else None

        records.append(
            MemoryRecord(
                id=f"{stem}#b{n}",
                text=text,
                timestamp=_recover_ts(inline, meta, path),
                role=role,
                metadata={
                    "source_file": str(path),
                    "tags": tags,
                },
            )
        )
    return records


def _dated_split(body: str, meta: dict, path: str | Path) -> list[MemoryRecord]:
    """Split ``body`` into one ``MemoryRecord`` per dated entry.

    A dated marker is a ``# YYYY-MM-DD`` heading or a ``- [YYYY-MM-DD]``
    bullet. Each marker starts a record spanning until the next marker; the
    captured date becomes THAT record's timestamp (so per-entry dates differ).
    """
    lines = body.split("\n")
    markers = [(i, d) for i, ln in enumerate(lines) if (d := _dated_marker(ln))]

    stem = Path(path).stem
    role = meta.get("role", "user")
    tags = meta.get("tags", [])
    records: list[MemoryRecord] = []

    for n, (start, date) in enumerate(markers):
        end = markers[n + 1][0] if n + 1 < len(markers) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        if not text:
            continue

        records.append(
            MemoryRecord(
                id=f"{stem}#{date}-{n}",
                text=text,
                timestamp=_recover_ts(date, meta, path),
                role=role,
                metadata={
                    "source_file": str(path),
                    "entry_date": date,
                    "tags": tags,
                },
            )
        )
    return records


def _detect_split(body: str) -> str:
    """Detect the best split mode for ``body``: dated / bullet / heading."""
    lines = body.split("\n")
    dated = sum(1 for ln in lines if _dated_marker(ln))
    if dated >= 2:
        return "dated"
    bullets = sum(
        1 for ln in lines if _BULLET_RE.match(ln) and not ln[:1].isspace()
    )
    headings = sum(1 for ln in lines if _HEADING_RE.match(ln))
    if bullets > headings:
        return "bullet"
    return "heading"


def load_markdown_dump(path: str | Path, *, split: str = "auto") -> list[MemoryRecord]:
    """Load a Markdown memory dump into ``MemoryRecord`` objects.

    * ``"heading"`` — one record per ``##``..``######`` section.
    * ``"bullet"`` — one record per top-level list item.
    * ``"dated"`` — one record per dated entry, each keeping its own date.
    * ``"auto"`` — detect the shape (see :func:`_detect_split`).
    """
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()

    meta, body = _parse_frontmatter(raw)

    if split == "auto":
        split = _detect_split(body)

    if split == "heading":
        return _heading_split(body, meta, path)
    if split == "bullet":
        return _bullet_split(body, meta, path)
    if split == "dated":
        return _dated_split(body, meta, path)
    raise ValueError(f"unknown split mode: {split!r}")
