"""Tests for the SQLite-backed KeyStore."""

from __future__ import annotations

import threading
import pytest
from lmfm.server.keys import KeyStore


# ---------------------------------------------------------------------------
# Basic interface (mirrors original in-memory behaviour)
# ---------------------------------------------------------------------------

def test_authorize_unknown_key():
    store = KeyStore()
    assert store.authorize("not-a-key") == "unauthorized"
    assert store.authorize(None) == "unauthorized"


def test_provision_and_authorize():
    store = KeyStore()
    store.provision("k1", quota=5)
    assert store.authorize("k1") == "ok"


def test_quota_decrements_via_consume():
    store = KeyStore()
    store.provision("k", quota=2)
    assert store.authorize("k") == "ok"
    store.consume("k")
    assert store.authorize("k") == "ok"
    store.consume("k")
    assert store.authorize("k") == "exhausted"


def test_remaining_returns_current_quota():
    store = KeyStore()
    store.provision("k", quota=3)
    assert store.remaining("k") == 3
    store.consume("k")
    assert store.remaining("k") == 2


def test_remaining_returns_none_for_unknown_key():
    store = KeyStore()
    assert store.remaining("ghost") is None


def test_revoke_removes_key():
    store = KeyStore()
    store.provision("k", quota=10)
    assert store.revoke("k") is True
    assert store.authorize("k") == "unauthorized"


def test_revoke_returns_false_for_missing_key():
    store = KeyStore()
    assert store.revoke("never-existed") is False


# ---------------------------------------------------------------------------
# Persistence across instances (file-based SQLite)
# ---------------------------------------------------------------------------

def test_persistence_across_instances(tmp_path):
    db = str(tmp_path / "keys.db")

    # Write in one instance
    s1 = KeyStore(db)
    s1.provision("persistent-key", quota=7, label="acme")
    s1.consume("persistent-key")
    s1.close()

    # Read back in a new instance pointing at the same file
    s2 = KeyStore(db)
    assert s2.remaining("persistent-key") == 6
    assert s2.authorize("persistent-key") == "ok"
    s2.close()


def test_provision_upsert_replaces_quota(tmp_path):
    db = str(tmp_path / "keys.db")
    s = KeyStore(db)
    s.provision("k", quota=10)
    s.consume("k")           # quota → 9
    s.provision("k", quota=5)  # upsert resets to 5
    assert s.remaining("k") == 5
    s.close()


# ---------------------------------------------------------------------------
# Usage log
# ---------------------------------------------------------------------------

def test_usage_log_records_consume_calls():
    store = KeyStore()
    store.provision("k", quota=3)
    store.consume("k")
    store.consume("k")
    log = store.usage("k")
    assert len(log) == 2
    assert all(e["outcome"] == "ok" for e in log)


def test_usage_log_empty_for_unknown_key():
    store = KeyStore()
    assert store.usage("ghost") == []


# ---------------------------------------------------------------------------
# from_dict backward-compat factory
# ---------------------------------------------------------------------------

def test_from_dict_seeds_store():
    store = KeyStore.from_dict({"a": {"quota": 3}, "b": {"quota": 0}})
    assert store.authorize("a") == "ok"
    assert store.authorize("b") == "exhausted"
    assert store.authorize("c") == "unauthorized"


# ---------------------------------------------------------------------------
# Thread safety — concurrent consume calls must not race
# ---------------------------------------------------------------------------

def test_concurrent_consume_does_not_undercount(tmp_path):
    db = str(tmp_path / "keys.db")
    store = KeyStore(db)
    store.provision("k", quota=50)

    errors: list[Exception] = []

    def worker():
        try:
            if store.authorize("k") == "ok":
                store.consume("k")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    remaining = store.remaining("k")
    assert remaining is not None
    assert 20 <= remaining <= 50   # 30 threads consumed at most 30
    store.close()
