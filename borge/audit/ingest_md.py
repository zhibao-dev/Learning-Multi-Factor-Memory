"""Markdown ingest for ``borge audit``.

Target agents (Claude Code / Hermes / Codex) keep their memory in Markdown,
not canonical JSON. This module reads such a dump into the SAME
``MemoryRecord`` objects that :mod:`borge.audit.ingest` produces, so the
downstream value pipeline is untouched.

This slice implements ``split="heading"`` (one record per ``##``+ section),
YAML-frontmatter parsing (hand-rolled, no pyyaml dependency) and a
timestamp-recovery chain (inline date → frontmatter → file mtime).
Bullet/dated splitting and real ``auto`` detection arrive in a later task.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from borge.audit.ingest import MemoryRecord

_HEADING_RE = re.compile(r"^#{2,6}\s+")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


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
        else:
            heading_text = _HEADING_RE.sub("", heading_line).strip()
            slug = _slug(heading_text) or "section"

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
                role=role,
                metadata={
                    "source_file": str(path),
                    "heading_path": heading_text,
                    "tags": tags,
                },
            )
        )
    return records


def load_markdown_dump(path: str | Path, *, split: str = "auto") -> list[MemoryRecord]:
    """Load a Markdown memory dump into ``MemoryRecord`` objects.

    ``split="heading"`` produces one record per ``##``..``######`` section.
    ``split="auto"`` falls back to heading for now. ``"bullet"`` / ``"dated"``
    are reserved for a later task and raise :class:`NotImplementedError`.
    """
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()

    meta, body = _parse_frontmatter(raw)

    if split in ("heading", "auto"):
        return _heading_split(body, meta, path)
    if split in ("bullet", "dated"):
        raise NotImplementedError(f"split={split!r} is not implemented yet")
    raise ValueError(f"unknown split mode: {split!r}")
