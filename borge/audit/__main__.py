"""``borge-audit`` CLI: audit a memory dump → markdown report + dry-run forget list.

Reads a JSON memory dump, runs :func:`borge.audit.report.build_audit`, and writes
the markdown report to ``-o`` (default ``<dump>.audit.md``) plus the dry-run
forget script to ``<out>.forget.json``. **The input dump is never modified or
deleted.**
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..values.self_model import SBertEmbedder
from .report import build_audit


def _load_soul_centroid(soul_path: str) -> list[float] | None:
    """Embed each non-empty line of a SOUL/values file into one centroid vector."""
    lines = [ln.strip() for ln in Path(soul_path).read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    embedder = SBertEmbedder()
    embs = [embedder(ln) for ln in lines]
    dim = len(embs[0])
    return [sum(e[i] for e in embs) / len(embs) for i in range(dim)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="borge-audit",
        description="Audit a memory dump: bloat / contradiction / hygiene report (dry-run only).",
    )
    parser.add_argument("dump", help="path to the memory dump (.json or .md)")
    parser.add_argument("--soul", help="path to a SOUL.md / values file for value-alignment scoring")
    parser.add_argument("--md-split", choices=["heading", "bullet", "dated", "auto"], default="auto",
                        help="how to split a Markdown dump into memories (default auto; ignored for .json)")
    parser.add_argument("--budget", type=float, default=0.5,
                        help="target keep-fraction for the headline forget set (default 0.5)")
    parser.add_argument("--retrieval-freq", type=float, default=30.0,
                        help="assumed re-injections/month for the savings estimate (default 30)")
    parser.add_argument("--weights", help="path to a JSON file of factor weights to override the defaults")
    parser.add_argument("--llm-endpoint",
                        help="OpenAI-compatible base URL of YOUR OWN LLM / local Ollama "
                             "(e.g. http://localhost:11434/v1) to LLM-verify contradiction "
                             "candidates; off by default (NLI-only, local)")
    parser.add_argument("--llm-model",
                        help="model name passed to --llm-endpoint (e.g. llama3, gpt-4o-mini)")
    parser.add_argument("--llm-key",
                        help="API key for --llm-endpoint (optional; a local Ollama needs none)")
    parser.add_argument("-o", "--output", help="report markdown path (default <dump>.audit.md)")
    args = parser.parse_args(argv)

    # now_iso is computed here (CLI), not inside the pure functions, then passed down.
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    soul_centroid = _load_soul_centroid(args.soul) if args.soul else None
    weights = json.loads(Path(args.weights).read_text(encoding="utf-8")) if args.weights else None

    result = build_audit(
        args.dump,
        now_iso=now_iso,
        soul_centroid=soul_centroid,
        budget=args.budget,
        retrieval_freq=args.retrieval_freq,
        md_split=args.md_split,
        weights=weights,
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        llm_key=args.llm_key,
    )

    out_md = Path(args.output) if args.output else Path(str(args.dump) + ".audit.md")
    out_forget = Path(str(out_md) + ".forget.json")

    out_md.write_text(result["markdown"], encoding="utf-8")
    out_forget.write_text(json.dumps(result["forget_script"], indent=2), encoding="utf-8")

    print(f"Report written to {out_md}")
    print(f"Dry-run forget list written to {out_forget}")
    print("Input dump was not modified; nothing was deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
