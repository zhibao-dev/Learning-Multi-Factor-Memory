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


def test_learn_requires_api_key():
    app = create_app(valid_keys={"k-good": {"quota": 5}})
    c = TestClient(app)
    r = c.post("/learn", json=_matrix())
    assert r.status_code == 401


def test_learn_with_valid_key_returns_weights():
    app = create_app(valid_keys={"k-good": {"quota": 5}})
    c = TestClient(app)
    r = c.post("/learn", json=_matrix(), headers={"Authorization": "Bearer k-good"})
    assert r.status_code == 200
    body = r.json()
    assert "weights" in body and "train_retention" in body


def test_quota_decrements_and_blocks():
    app = create_app(valid_keys={"k-lim": {"quota": 1}})
    c = TestClient(app)
    h = {"Authorization": "Bearer k-lim"}
    assert c.post("/learn", json=_matrix(), headers=h).status_code == 200
    assert c.post("/learn", json=_matrix(), headers=h).status_code == 429


def test_malformed_bearer_returns_401():
    app = create_app(valid_keys={"k-good": {"quota": 5}})
    c = TestClient(app)
    for bad in ("Basic xyz", "Bearer", "k-good", ""):
        r = c.post("/learn", json=_matrix(), headers={"Authorization": bad})
        assert r.status_code == 401


def test_missing_keys_body_returns_422():
    app = create_app(valid_keys={"k-good": {"quota": 5}})
    c = TestClient(app)
    r = c.post("/learn", json={}, headers={"Authorization": "Bearer k-good"})
    assert r.status_code == 422


def test_quota_preserved_on_bad_body():
    app = create_app(valid_keys={"k-lim": {"quota": 1}})
    c = TestClient(app)
    h = {"Authorization": "Bearer k-lim"}
    assert c.post("/learn", json={}, headers=h).status_code == 422   # malformed, no charge
    assert c.post("/learn", json=_matrix(), headers=h).status_code == 200  # quota intact
