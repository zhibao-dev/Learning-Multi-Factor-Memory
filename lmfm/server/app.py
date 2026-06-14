"""FastAPI app exposing the paid /learn endpoint."""

from __future__ import annotations

from fastapi import FastAPI, Request, HTTPException

from ..learn.service import learn_from_matrix
from .keys import KeyStore


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else None


def create_app(valid_keys: dict) -> FastAPI:
    app = FastAPI(title="lmfm learn service")
    store = KeyStore(valid_keys)

    @app.post("/learn")
    async def learn(request: Request):
        key = _bearer(request)
        status = store.authorize(key)
        if status == "unauthorized":
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        if status == "exhausted":
            raise HTTPException(status_code=429, detail="quota exhausted")
        matrix = await request.json()
        result = learn_from_matrix(matrix)
        store.consume(key)
        return result

    return app
