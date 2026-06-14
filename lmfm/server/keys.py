"""In-memory API-key store + metering. Swap for a DB in production."""

from __future__ import annotations


class KeyStore:
    def __init__(self, valid_keys: dict):
        # {key: {"quota": int}}  (quota = remaining /learn calls)
        self._keys = {k: dict(v) for k, v in valid_keys.items()}

    def authorize(self, key: str | None) -> str:
        """Return 'ok' | 'unauthorized' | 'exhausted'."""
        if key is None or key not in self._keys:
            return "unauthorized"
        if self._keys[key].get("quota", 0) <= 0:
            return "exhausted"
        return "ok"

    def consume(self, key: str) -> None:
        self._keys[key]["quota"] -= 1
