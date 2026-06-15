import json
from pathlib import Path
import lmfm.client as client
from lmfm.cli import main


def test_learn_uploads_matrix_and_saves_weights(tmp_path, monkeypatch):
    mat = tmp_path / "factors.json"
    mat.write_text(json.dumps({"keep_frac": 0.5, "factor_order": ["reliability"],
        "cases": [{"memories": [{"factors": {"reliability": 0.9}, "gold": True}]}]}))
    out = tmp_path / "weights.json"

    def fake_post(url, payload, api_key):
        assert "text" not in json.dumps(payload)   # privacy guard
        return {"weights": {"reliability": 0.7}, "train_retention": 1.0}

    monkeypatch.setattr(client, "post_learn", fake_post)
    rc = main(["learn", str(mat), "--endpoint", "https://x/learn",
               "--key", "k-good", "-o", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["weights"]["reliability"] == 0.7
