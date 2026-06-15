"""FastAPI app exposing the paid /learn endpoint."""

from __future__ import annotations

from fastapi import FastAPI, Request, HTTPException

from ..learn.service import learn_from_matrix
from .keys import KeyStore


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else None


def create_app(
    valid_keys: dict | None = None,
    db_path: str = ":memory:",
) -> FastAPI:
    """Create the lmfm FastAPI application.

    Parameters
    ----------
    valid_keys:
        Dict ``{key: {"quota": int}}`` — seeds an in-memory SQLite store.
        Intended for tests and single-process dev.  If ``None``, the store
        is opened from ``db_path`` with no pre-seeding.
    db_path:
        Path to the SQLite database file.  Defaults to ``":memory:"``.
        In production, pass an absolute path such as
        ``"/var/lib/lmfm/keys.db"``.
    """
    if valid_keys is not None:
        store = KeyStore.from_dict(valid_keys, db_path=db_path)
    else:
        store = KeyStore(db_path)

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
        status = store.authorize(key)
        if status == "unauthorized":
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        if status == "exhausted":
            raise HTTPException(
                status_code=429,
                detail="quota exhausted — contact support to top up your quota",
            )
        assert key is not None
        matrix = await request.json()
        if not isinstance(matrix, dict) or "keep_frac" not in matrix or "cases" not in matrix:
            raise HTTPException(status_code=422, detail="matrix requires 'keep_frac' and 'cases'")
        result = learn_from_matrix(matrix)
        store.consume(key)
        return result

    return app
