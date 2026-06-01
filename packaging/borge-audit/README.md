# borge-audit — standalone private wheel

Build a **self-contained `borge-audit` wheel** to ship to customers without
exposing the research core. The wheel contains **only the thin audit slice**
— the multi-factor value model, the SBert embedder, the emotion signal
extractor, and `borge/audit/` — and **never** the full BorgeAgent runtime or
the paper1 self-FEP code.

## What ships vs what doesn't

| In the wheel | NOT in the wheel |
|---|---|
| `borge/memory/value.py` (value model + shipped weights — already public in paper2) | `borge/agent.py`, `runner.py`, `cli.py` (the agent) |
| `borge/values/self_model.py` (embedder) | `borge/memory/forgetting.py`, `consolidation.py`, `store.py`, `retrieval.py` (paper1 runtime) |
| `borge/affective/signal_extractor.py` | `borge/beliefs/`, `inference/`, `meta/`, `value_net.py` |
| `borge/audit/*` (the product) | everything else |

The build **fails loudly** if any forbidden module sneaks into the wheel (IP guard).

> The slice is re-copied fresh from the research source on every build, so it never diverges. `src/` and `dist/` are gitignored — only the build script + pyproject + this README are tracked.

## Build (you)

```bash
bash packaging/borge-audit/build_wheel.sh
# → packaging/borge-audit/dist/borge_audit-0.1.0-py3-none-any.whl
```

Requires `pip` + `setuptools` in the active Python. The wheel itself has no bundled
models; deps (`sentence-transformers`, `tiktoken`) install on the customer side.

## Install (customer)

Send the `.whl` directly (email / private index / signed URL — **not public PyPI**):

```bash
pip install borge_audit-0.1.0-py3-none-any.whl
borge-audit memories.json            # → memories.json.audit.md + .forget.json
```

First run downloads the local SBert + NLI models to the customer's Hugging Face
cache; everything then runs offline — **no memory data leaves their machine.**

## Distribution policy

- **Do NOT `twine upload` to public PyPI.** This is a private wheel for paying pilots.
- For wider OSS distribution later, switch to open-core: publish the readable
  source, and monetise the un-copyable layer (per-workload tuned weights, managed
  control plane, support/SLA, enterprise compliance) — not the code.
- Tiered/licensed features belong in a separate gated module, not in this wheel.
