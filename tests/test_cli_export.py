import json
from pathlib import Path
from lmfm.cli import main


def test_export_factors_writes_numeric_matrix(tmp_path: Path):
    md = tmp_path / "mem.md"
    md.write_text("## A\nUser allergic to penicillin.\n\n## B\nLikes dark mode.\n")
    gold = tmp_path / "gold.txt"
    gold.write_text("mem#a\n")
    out = tmp_path / "factors.json"
    rc = main(["export-factors", str(md), "--gold", str(gold),
               "-o", str(out), "--hash-embed"])
    assert rc == 0
    mat = json.loads(out.read_text())
    assert mat["schema"] == "lmfm.factors.v1"
    assert "text" not in out.read_text()


def test_missing_input_returns_error(tmp_path, capsys):
    rc = main(["export-factors", str(tmp_path / "nope.md"),
               "-o", str(tmp_path / "o.json"), "--hash-embed"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_gold_omitted_all_false(tmp_path):
    import json
    md = tmp_path / "mem.md"
    md.write_text("## A\nfact one.\n\n## B\nfact two.\n")
    out = tmp_path / "f.json"
    rc = main(["export-factors", str(md), "-o", str(out), "--hash-embed"])
    assert rc == 0
    mat = json.loads(out.read_text())
    assert all(m["gold"] is False for m in mat["cases"][0]["memories"])
