"""FastAPI app — open access with IP rate limiting, optional key auth."""

from __future__ import annotations

import collections
import threading
import time

from fastapi import FastAPI, Request, HTTPException

from ..learn.service import learn_from_matrix
from .keys import KeyStore


class _IPRateLimiter:
    """Sliding-window rate limiter keyed by IP address."""

    def __init__(self, limit: int, window: int = 3600) -> None:
        self._limit = limit
        self._window = window
        self._calls: dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            dq = self._calls.setdefault(ip, collections.deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._limit:
                return False
            dq.append(now)
            return True


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else None


def create_app(
    valid_keys: dict | None = None,
    db_path: str = ":memory:",
    rate_limit: int = 20,
    rate_window: int = 3600,
) -> FastAPI:
    """Create the lmfm FastAPI application.

    Parameters
    ----------
    valid_keys:
        Seed an in-memory SQLite store from ``{key: {"quota": int}}``.
        Intended for tests.  Production: leave ``None`` and set ``db_path``.
    db_path:
        SQLite database path.  Defaults to ``":memory:"``.
    rate_limit:
        Max ``/learn`` calls per IP per ``rate_window`` for unauthenticated
        requests.  Authenticated (keyed) requests bypass this limit.
    rate_window:
        Sliding-window size in seconds (default 3600 = 1 hour).
    """
    if valid_keys is not None:
        store = KeyStore.from_dict(valid_keys, db_path=db_path)
    else:
        store = KeyStore(db_path)

    limiter = _IPRateLimiter(limit=rate_limit, window=rate_window)
    app = FastAPI(title="lmfm learn service")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/quota")
    async def quota(request: Request):
        """Return remaining quota for the authenticated key."""
        key = _bearer(request)
        if key is None:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        remaining = store.remaining(key)
        if remaining is None:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return {"quota_remaining": remaining}

    @app.post("/learn")
    async def learn(request: Request):
        key = _bearer(request)

        if key is not None:
            # Explicit Bearer token — must be valid
            status = store.authorize(key)
            if status == "unauthorized":
                raise HTTPException(status_code=401, detail="invalid API key")
            if status == "exhausted":
                raise HTTPException(
                    status_code=429,
                    detail="quota exhausted — contact support to top up",
                )
        else:
            # No key — open access subject to per-IP rate limit
            ip = (request.client.host if request.client else "unknown")
            if not limiter.allow(ip):
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"rate limit: {rate_limit} calls/hour per IP. "
                        "Add an API key for unlimited access."
                    ),
                )

        matrix = await request.json()
        if not isinstance(matrix, dict) or "keep_frac" not in matrix or "cases" not in matrix:
            raise HTTPException(status_code=422, detail="matrix requires 'keep_frac' and 'cases'")

        result = learn_from_matrix(matrix)

        if key is not None:
            store.consume(key)

        return result

    return app
