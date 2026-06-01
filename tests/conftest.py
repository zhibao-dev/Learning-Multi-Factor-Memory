"""Pytest session setup.

Model-based tests (SBert embedder, NLI cross-encoder) otherwise hit the Hugging
Face Hub for an online cache-validation round-trip on every run. Under load that
round-trip intermittently times out and produces FLAKY failures (e.g.
test_audit_bloat / test_audit_contradiction) that have nothing to do with the
code under test.

If the models are ALREADY cached locally, force offline mode so the suite is
deterministic and fast. A fresh machine with no cache still downloads on first
run (we only flip offline when the cache is present).
"""

import os
from pathlib import Path


def _models_cached() -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if not hub.exists():
        return False
    names = [p.name for p in hub.iterdir()]
    # The SBert embedder is needed by every model-based test; if it's cached the
    # NLI cross-encoder is too (they're fetched together by the audit tests).
    return any("all-MiniLM-L6-v2" in n for n in names)


if _models_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
