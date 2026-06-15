import json
from pathlib import Path
import lmfm.client as client
from lmfm.cli import main


def _matrix_file(tmp_path):
    mat = tmp_path / "factors.json"
    mat.write_text(json.dumps({
        "keep_frac": 0.5,
        "factor_order": ["reliability"],
        "cases": [{"memories": [{"factors": {"reliability": 0.9}, "gold": True}]}],
    }))
    return mat


def test_learn_with_key_uploads_matrix_and_saves_weights(tmp_path, monkeypatch):
    mat = _matrix_file(tmp_path)
    out = tmp_path / "weights.json"

    def fake_post(url, payload, api_key=None):
        assert "text" not in json.dumps(payload)   # privacy guard
        assert api_key == "k-good"
        return {"weights": {"reliability": 0.7}, "train_retention": 1.0}

    monkeypatch.setattr(client, "post_learn", fake_post)
    rc = main(["learn", str(mat), "--endpoint", "https://x/learn",
               "--key", "k-good", "-o", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["weights"]["reliability"] == 0.7


def test_learn_without_key_sends_no_auth_header(tmp_path, monkeypatch):
    mat = _matrix_file(tmp_path)
    out = tmp_path / "weights.json"

    def fake_post(url, payload, api_key=None):
        assert api_key is None   # no key passed
        return {"weights": {"reliability": 0.6}, "train_retention": 0.9}

    monkeypatch.setattr(client, "post_learn", fake_post)
    rc = main(["learn", str(mat), "--endpoint", "https://x/learn", "-o", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["weights"]["reliability"] == 0.6
