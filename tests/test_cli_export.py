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
