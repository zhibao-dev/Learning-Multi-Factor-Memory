import json
from pathlib import Path
from lmfm.cli import main
from lmfm.learn.service import learn_from_matrix
from lmfm.weights import load_value_from_weights_file


def test_export_then_learn_then_load(tmp_path: Path):
    md = tmp_path / "mem.md"
    md.write_text("## A\nUser is allergic to penicillin.\n\n## B\nok thanks\n")
    gold = tmp_path / "gold.txt"; gold.write_text("mem#a\n")
    fac = tmp_path / "factors.json"
    assert main(["export-factors", str(md), "--gold", str(gold),
                 "-o", str(fac), "--hash-embed"]) == 0

    mat = json.loads(fac.read_text())
    result = learn_from_matrix(mat, iters=40)
    wf = tmp_path / "weights.json"; wf.write_text(json.dumps(result))

    mv = load_value_from_weights_file(wf)
    assert 0.0 <= result["train_retention"] <= 1.0
    assert mv.value({"reliability": 1.0}) >= 0.0
