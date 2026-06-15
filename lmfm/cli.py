"""lmfm command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io.markdown import load_markdown
from .factors.annotate import annotate_memories
from .export import build_matrix


def _embedder(use_hash: bool):
    from .factors.embedder import SBertEmbedder, hash_embed
    if use_hash:
        return hash_embed
    return SBertEmbedder()


def _cmd_export(args) -> int:
    try:
        recs = load_markdown(args.dump, split=args.md_split)
    except FileNotFoundError:
        print(f"error: input dump not found: {args.dump}", file=sys.stderr)
        return 1
    gold = set()
    if args.gold:
        gold = {ln.strip() for ln in Path(args.gold).read_text().splitlines() if ln.strip()}
    ann = annotate_memories(recs, embedder=_embedder(args.hash_embed), gold_ids=gold)
    mat = build_matrix([ann], keep_frac=args.keep_frac)
    out = Path(args.out)
    out.write_text(json.dumps(mat, indent=2))
    print(f"wrote {out}  ({len(ann)} memories, {sum(a['gold'] for a in ann)} gold)")
    return 0


def _cmd_learn(args) -> int:
    from . import client
    payload = json.loads(Path(args.matrix).read_text())
    result = client.post_learn(args.endpoint, payload, args.key)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"learned weights → {args.out}  (train_retention={result.get('train_retention')})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lmfm")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export-factors",
                       help="locally extract memories into a numeric factor matrix")
    e.add_argument("dump", help="path to a markdown memory dump")
    e.add_argument("-o", "--out", default="factors.json")
    e.add_argument("--gold", help="file of gold memory ids (one per line)")
    e.add_argument("--keep-frac", type=float, default=0.3)
    e.add_argument("--md-split", default="auto",
                   choices=["heading", "bullet", "dated", "auto"])
    e.add_argument("--hash-embed", action="store_true",
                   help="use deterministic hash embedding (no model download)")
    e.set_defaults(func=_cmd_export)

    l = sub.add_parser("learn", help="upload a factor matrix → learned weights")
    l.add_argument("matrix", help="factors.json from export-factors")
    l.add_argument("--endpoint", required=True)
    l.add_argument("--key", default=None,
                   help="API key (optional — omit for free rate-limited access)")
    l.add_argument("-o", "--out", default="weights.json")
    l.set_defaults(func=_cmd_learn)

    args = p.parse_args(argv)
    return args.func(args)
