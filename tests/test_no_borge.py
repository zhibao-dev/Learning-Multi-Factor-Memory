import pathlib
import re


def test_no_forbidden_naming():
    root = pathlib.Path(__file__).resolve().parents[1] / "lmfm"
    pat = re.compile(r"borge", re.IGNORECASE)
    offenders = []
    for p in root.rglob("*.py"):
        if pat.search(p.read_text(encoding="utf-8")):
            offenders.append(str(p))
        if pat.search(p.name):
            offenders.append(p.name)
    assert not offenders, f"'borge' found in: {offenders}"
