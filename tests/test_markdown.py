from pathlib import Path
from lmfm.io.markdown import load_markdown, MemoryRecord


def test_heading_split(tmp_path: Path):
    p = tmp_path / "mem.md"
    p.write_text("## Allergy\nUser is allergic to penicillin.\n\n## Pref\nLikes dark mode.\n")
    recs = load_markdown(p, split="heading")
    assert all(isinstance(r, MemoryRecord) for r in recs)
    texts = " ".join(r.text for r in recs)
    assert "penicillin" in texts and "dark mode" in texts
    assert len(recs) == 2
