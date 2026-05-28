"""
Build the per-turn factor cache for paper2's multi-factor evaluation.

This script is the **one** SBert-heavy step in the paper2 pipeline. It
lifts the per-turn factor derivation out of
`experiments/lme_blind_forgetting.py::annotate_dual` into a standalone
pass that:

  1. loads LongMemEval-S cases,
  2. embeds the question + every turn ONCE with SBert,
  3. computes the four regime-independent factors plus BOTH goal
     variants (oracle = cos(turn, question); blind = cos(turn,
     per-session user-centroid)),
  4. writes one JSONL record per usable case to
     `results/lme_factor_cache.jsonl`.

Every downstream evaluator (Task 2 keep-frac sweep, Task 3 bootstrap CI,
Task 4 neural ablation, etc.) reads this cache in seconds and never
re-runs SBert. This is what unblocks the full-500 run that previously
got killed for being too slow.

Formulas are **byte-identical** to `annotate_dual` so the 74-case
headline numbers (learned 0.811, uniform 0.634, reliability_only 0.541,
recency 0.380) reproduce from the cache:

    sim01(a, b) = 0.5 + 0.5 * cosine(a, b)
    goal_oracle  = sim01(turn_emb, question_emb)
    goal_blind   = sim01(turn_emb, session-user-centroid)
    self         = sim01(turn_emb, global μ_user)   (0.5 if no user turns)
    emotion      = clamp01(|ΔV| · (0.5 + ΔA))
    reliability  = 0.7 (user)  /  0.4 (assistant)

Filter: cases with zero `has_answer` turns are dropped (same filter
as `lme_blind_forgetting.py`).

Output schema (one JSON object per line):

    {"qid": "...", "turns": [
        {"emotion": 0.0, "self": 0.61, "reliability": 0.7,
         "goal_oracle": 0.83, "goal_blind": 0.55,
         "has_answer": false, "sidx": 3},
        ...
    ]}

`turns` preserves `flatten_to_messages` order, so the cache is a
faithful 1-to-1 chronological view of the haystack.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from borge.affective.signal_extractor import EmotionalSignalExtractor  # noqa: E402
from borge.eval.longmemeval import (  # noqa: E402
    LongMemEvalCase,
    flatten_to_messages,
    load_longmemeval,
)
from borge.values.self_model import cosine  # noqa: E402


# ── Factor formulas (verbatim from `annotate_dual`) ───────────────────────

_EXTRACTOR = EmotionalSignalExtractor()


def _sim01(a: list[float], b: list[float]) -> float:
    """Cosine remapped to [0, 1]. Identical to lme_blind_forgetting._sim01."""
    return 0.5 + 0.5 * cosine(a, b)


def _centroid(vs: list[list[float]]) -> list[float]:
    """Element-wise mean of a list of vectors. Identical to lme_blind_forgetting._centroid."""
    if not vs:
        return []
    dim = len(vs[0])
    return [sum(v[i] for v in vs) / len(vs) for i in range(dim)]


def case_to_record(case: LongMemEvalCase, embedder: Callable[[str], list[float]]) -> dict:
    """
    Compute the per-turn factor record for one LongMemEvalCase.

    Returns `{"qid": case.question_id, "turns": [{...}, ...]}` where
    `turns` is in `flatten_to_messages` order. Returns a record with an
    empty `turns` list iff the case has no messages — the caller is
    responsible for the has-answer filter.
    """
    msgs = flatten_to_messages(case)
    contents = [m["content"] for m in msgs]
    if not contents:
        return {"qid": case.question_id, "turns": []}

    # One SBert call per text — the whole point of this cache.
    q_emb = embedder(case.question)
    turn_embs = [embedder(c) for c in contents]

    # Global μ_user anchors `self_relevance`.
    user_embs = [e for e, m in zip(turn_embs, msgs) if m["role"] == "user"]
    mu_user = _centroid(user_embs)

    # Per-session user-centroid anchors `goal_blind` ("what's this session about").
    by_sess: dict[int, list[list[float]]] = {}
    for e, m in zip(turn_embs, msgs):
        if m["role"] == "user":
            by_sess.setdefault(m["session_idx"], []).append(e)
    sess_anchor = {s: _centroid(es) for s, es in by_sess.items()}

    turns: list[dict] = []
    for m, emb in zip(msgs, turn_embs):
        dv, da = _EXTRACTOR.extract(m["content"], [])
        emotion = max(0.0, min(1.0, abs(dv) * (0.5 + da)))
        self_rel = _sim01(emb, mu_user) if mu_user else 0.5
        reliability = 0.7 if m["role"] == "user" else 0.4
        goal_oracle = _sim01(emb, q_emb)
        anchor = sess_anchor.get(m["session_idx"]) or mu_user
        goal_blind = _sim01(emb, anchor) if anchor else 0.5

        turns.append({
            "emotion":      emotion,
            "self":         self_rel,
            "reliability":  reliability,
            "goal_oracle":  goal_oracle,
            "goal_blind":   goal_blind,
            "has_answer":   bool(m["has_answer"]),
            "sidx":         int(m["session_idx"]),
        })

    return {"qid": case.question_id, "turns": turns}


# ── Builder ────────────────────────────────────────────────────────────────

def build(
    data_path: str | Path,
    out_path: str | Path,
    n_cases: int,
    embedder: Callable[[str], list[float]],
    *,
    progress_every: int = 25,
    cases: Iterable[LongMemEvalCase] | None = None,
) -> int:
    """
    Stream cases through `case_to_record` and write JSONL to `out_path`.

    Cases with zero `has_answer` turns are dropped (same filter as
    `lme_blind_forgetting.py`). Returns the number of records written.

    The function is intentionally thin so the test can inject a fake
    embedder and a synthetic dataset path. The CLI wraps this with an
    SBert embedder.
    """
    if cases is None:
        # Validate the input BEFORE truncating the output, so a bad --data
        # path can't clobber an existing (slow-to-regenerate) cache.
        if not Path(data_path).exists():
            raise FileNotFoundError(f"data file not found: {data_path}")
        cases = load_longmemeval(data_path)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_seen = 0
    with out.open("w") as f:
        for case in cases:
            if n_seen >= n_cases:
                break
            n_seen += 1
            rec = case_to_record(case, embedder)
            if any(t["has_answer"] for t in rec["turns"]):
                f.write(json.dumps(rec))
                f.write("\n")
                n_written += 1
            if n_seen % progress_every == 0:
                print(f"  processed {n_seen}  (kept {n_written})")
    return n_written


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build per-turn factor cache (one SBert pass) for paper2 eval."
    )
    ap.add_argument("--data", default="data/longmemeval_s_cleaned.json",
                    help="path to LongMemEval-S JSON")
    ap.add_argument("--n-cases", type=int, default=500,
                    help="max input cases to scan (default 500)")
    ap.add_argument("--out", default="results/lme_factor_cache.jsonl",
                    help="JSONL output path")
    args = ap.parse_args()

    print("Loading SBert (one-shot embedder for the full pass)…")
    from borge.values.self_model import SBertEmbedder
    embedder = SBertEmbedder()
    _ = embedder("warm-up")

    print(f"Scanning ≤{args.n_cases} cases from {args.data}; "
          f"writing usable cases to {args.out}")
    n_written = build(
        data_path=args.data,
        out_path=args.out,
        n_cases=args.n_cases,
        embedder=embedder,
    )
    print(f"\n  wrote {n_written} usable cases (≥1 has_answer turn) → {args.out}")


if __name__ == "__main__":
    main()
