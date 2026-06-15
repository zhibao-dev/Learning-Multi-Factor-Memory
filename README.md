<div align="center">

# lmfm

**Personalized memory weights for LLM agents — private by design**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CPU only](https://img.shields.io/badge/compute-CPU%20only-orange?style=flat-square)]()
[![No raw data upload](https://img.shields.io/badge/upload-numbers%20only-blueviolet?style=flat-square)]()

<br/>

*Teach your agent what to remember — without sending it your memories.*

</div>

---

## What It Does

`lmfm` learns a personalized set of **retention weights** from your agent's memory store.
The weights tell the agent which memories are worth keeping and which are safe to forget,
based on seven cognitively grounded factors (emotion, goal relevance, self-relevance,
reliability, usage, …).

**Privacy-first architecture**: factor extraction runs entirely on your machine.
Only a compact numeric matrix — no text, no embeddings, no IDs — is sent to the
learning service.

---

## Quickstart

```bash
pip install lmfm
```

### Step 1 — extract factors locally

```bash
lmfm export-factors memory.md -o factors.json
```

`memory.md` is your agent's Markdown memory dump (heading-split, bullet-split,
dated-log, or auto-detected). Outputs `factors.json` — a numeric matrix of the
7 retention factors. No raw text leaves your machine.

```bash
# Options
lmfm export-factors memory.md \
    --keep-frac 0.3          \  # target keep fraction (default 0.3)
    --gold gold_ids.txt      \  # optional: known high-value memory ids
    --md-split heading       \  # force split mode (default: auto)
    --hash-embed             \  # skip model download, use hash embedding
    -o factors.json
```

### Step 2 — learn personalized weights

```bash
lmfm learn factors.json --endpoint https://api.lmfm.dev/learn
```

Returns `weights.json` — your personalized retention weights, ready to load
into your agent's memory pipeline.

```bash
# With an API key (unlimited calls, bypasses rate limit)
lmfm learn factors.json \
    --endpoint https://api.lmfm.dev/learn \
    --key sk-your-key \
    -o weights.json
```

> **No key required.** The free tier allows 20 calls/hour per IP.
> An API key removes the rate limit.

---

## How It Works

Every memory $m$ receives a value score:

$$V(m) = \sum_{i=1}^{7} w_i \, f_i(m)$$

Seven factors grounded in cognitive psychology:

| # | Factor | Signal source |
|---|--------|--------------|
| 1 | **Emotional intensity** | Linguistic rules on the text |
| 2 | **Goal relevance** | Semantic distance to task-goal centroid |
| 3 | **Value alignment** | Proximity to agent's stated principles |
| 4 | **Self / user relevance** | Self-reference effect |
| 5 | **Task utility** | LLM-judge score (optional) |
| 6 | **Reliability** | Role heuristic (user-stated > model-stated) |
| 7 | **Usage history** | Retrieval count, saturating in [0, 1] |

`lmfm learn` fits the weights $\mathbf{w}$ on your data using a gradient-free
optimiser (random-restart hill-climb) that maximises gold-evidence retention
under a fixed memory budget. Training runs server-side in seconds.

---

## What Is and Isn't Uploaded

```
Your machine                         lmfm server
─────────────────────────────────────────────────
memory.md  →  [factor extraction]
               ↓
          factors.json  ──────────→  /learn
          (7 numbers per memory,      ↓
           gold flag, keep_frac)   weight optimisation
                                      ↓
          weights.json  ←──────────  personalized weights
```

The server **never sees** raw memory text, embeddings, or original IDs.

---

## API Reference

### `POST /learn`

Learn personalized retention weights from a factor matrix.

- **No key**: open access, 20 calls/hour per IP
- **With key**: unlimited, quota tracked per key

**Request body** (`factors.json` from `export-factors`):

```json
{
  "schema": "lmfm.factors.v1",
  "keep_frac": 0.3,
  "factor_order": ["emotion", "goal_relevance", "value_alignment",
                   "self_relevance", "task_utility", "reliability", "usage"],
  "cases": [
    {
      "memories": [
        {"factors": {"emotion": 0.8, "reliability": 0.9, ...}, "gold": true},
        {"factors": {"emotion": 0.1, "reliability": 0.4, ...}, "gold": false}
      ]
    }
  ]
}
```

**Response**:

```json
{
  "weights": {"reliability": 0.64, "emotion": 0.55, "self_relevance": 0.23, ...},
  "train_retention": 0.770,
  "model_version": "v1",
  "factors_learned": ["emotion", "goal_relevance", ...]
}
```

### `GET /quota`

Check remaining quota for an API key.

```bash
curl -H "Authorization: Bearer sk-your-key" https://api.lmfm.dev/quota
# → {"quota_remaining": 94}
```

### `GET /health`

```bash
curl https://api.lmfm.dev/health
# → {"status": "ok"}
```

---

## Self-Hosting

```bash
pip install lmfm[cloud]   # fastapi + uvicorn

python - <<'EOF'
from lmfm.server.app import create_app
import uvicorn

app = create_app(
    db_path="/var/lib/lmfm/keys.db",   # persistent SQLite key store
    rate_limit=20,                      # free-tier calls per hour per IP
)
uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
```

Provision an API key:

```python
from lmfm.server.keys import KeyStore
store = KeyStore("/var/lib/lmfm/keys.db")
store.provision("sk-customer-abc", quota=500, label="acme-corp")
```

---

## Citation

```bibtex
@article{chen2026multifactor,
  title   = {Learning What to Remember: A Cognitively Grounded
             Multi-Factor Value Model for Agentic Memory},
  author  = {Chen, Zhibao and Cheng, Qian},
  journal = {arXiv preprint arXiv:2606.12945},
  year    = {2026},
  url     = {https://arxiv.org/pdf/2606.12945}
}
```

---

## License

MIT © 2026 Zhibao Chen, Qian Cheng
