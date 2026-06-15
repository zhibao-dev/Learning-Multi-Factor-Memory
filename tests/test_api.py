# tests/test_api.py
from fastapi.testclient import TestClient
from lmfm.server.app import create_app


def _matrix():
    return {"keep_frac": 0.5,
            "factor_order": ["reliability", "emotion"],
            "cases": [{"memories": [
                {"factors": {"reliability": 0.9, "emotion": 0.1}, "gold": True},
                {"factors": {"reliability": 0.1, "emotion": 0.9}, "gold": False},
            ]}]}


# ---------------------------------------------------------------------------
# Open access (no key) — rate-limited
# ---------------------------------------------------------------------------

def test_learn_without_key_succeeds():
    app = create_app(rate_limit=10)
    c = TestClient(app)
    r = c.post("/learn", json=_matrix())
    assert r.status_code == 200
    assert "weights" in r.json()


def test_rate_limit_blocks_after_limit():
    app = create_app(rate_limit=2)
    c = TestClient(app)
    assert c.post("/learn", json=_matrix()).status_code == 200
    assert c.post("/learn", json=_matrix()).status_code == 200
    r = c.post("/learn", json=_matrix())
    assert r.status_code == 429
    assert "rate limit" in r.json()["detail"].lower()


def test_rate_limit_error_mentions_api_key():
    """429 message must tell users they can escape with a key."""
    app = create_app(rate_limit=0)    # limit=0 → blocks immediately
    c = TestClient(app)
    r = c.post("/learn", json=_matrix())
    assert r.status_code == 429
    assert "api key" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Authenticated access (with valid key) — bypasses rate limit
# ---------------------------------------------------------------------------

def test_learn_with_valid_key_returns_weights():
    app = create_app(valid_keys={"k-good": {"quota": 5}}, rate_limit=0)
    c = TestClient(app)
    r = c.post("/learn", json=_matrix(), headers={"Authorization": "Bearer k-good"})
    assert r.status_code == 200
    body = r.json()
    assert "weights" in body and "train_retention" in body


def test_keyed_user_bypasses_rate_limit():
    """A key holder is not subject to the IP rate limiter."""
    app = create_app(valid_keys={"k": {"quota": 5}}, rate_limit=0)
    c = TestClient(app)
    h = {"Authorization": "Bearer k"}
    for _ in range(3):
        assert c.post("/learn", json=_matrix(), headers=h).status_code == 200


def test_quota_decrements_and_blocks():
    app = create_app(valid_keys={"k-lim": {"quota": 1}})
    c = TestClient(app)
    h = {"Authorization": "Bearer k-lim"}
    assert c.post("/learn", json=_matrix(), headers=h).status_code == 200
    r = c.post("/learn", json=_matrix(), headers=h)
    assert r.status_code == 429
    assert "quota exhausted" in r.json()["detail"].lower()


def test_invalid_bearer_token_returns_401():
    """Explicitly supplying a wrong Bearer token → 401 (not silent fallthrough)."""
    app = create_app(valid_keys={"k-real": {"quota": 5}})
    c = TestClient(app)
    r = c.post("/learn", json=_matrix(), headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Malformed Authorization header → treated as no key → rate-limited
# ---------------------------------------------------------------------------

def test_malformed_auth_header_falls_through_to_rate_limit():
    """'Basic xyz', bare 'Bearer', raw key string → no token extracted → open access."""
    app = create_app(rate_limit=100)
    c = TestClient(app)
    for bad in ("Basic xyz", "Bearer", "k-good", ""):
        r = c.post("/learn", json=_matrix(), headers={"Authorization": bad})
        assert r.status_code == 200, f"expected 200 for header {bad!r}, got {r.status_code}"


# ---------------------------------------------------------------------------
# Body validation — no charge on malformed request
# ---------------------------------------------------------------------------

def test_missing_keys_body_returns_422():
    app = create_app(rate_limit=100)
    c = TestClient(app)
    r = c.post("/learn", json={})
    assert r.status_code == 422


def test_quota_preserved_on_bad_body():
    app = create_app(valid_keys={"k-lim": {"quota": 1}})
    c = TestClient(app)
    h = {"Authorization": "Bearer k-lim"}
    assert c.post("/learn", json={}, headers=h).status_code == 422     # malformed, no charge
    assert c.post("/learn", json=_matrix(), headers=h).status_code == 200  # quota intact


# ---------------------------------------------------------------------------
# /quota endpoint
# ---------------------------------------------------------------------------

def test_quota_endpoint_returns_remaining():
    app = create_app(valid_keys={"k": {"quota": 7}})
    c = TestClient(app)
    r = c.get("/quota", headers={"Authorization": "Bearer k"})
    assert r.status_code == 200
    assert r.json()["quota_remaining"] == 7


def test_quota_endpoint_requires_valid_key():
    app = create_app()
    c = TestClient(app)
    assert c.get("/quota").status_code == 401
    assert c.get("/quota", headers={"Authorization": "Bearer ghost"}).status_code == 401


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health():
    app = create_app()
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
