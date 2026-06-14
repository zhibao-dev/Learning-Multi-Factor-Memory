import json

from lmfm.io.markdown import MemoryRecord
from lmfm.factors.annotate import annotate_memories


class FakeEmbedder:
    def __call__(self, text):
        return [1.0, 0.0] if len(text) % 2 == 0 else [0.0, 1.0]


def test_annotate_emits_seven_factor_dict_and_gold():
    recs = [
        MemoryRecord(id="m1", text="User is allergic to penicillin.",
                     timestamp="2026-01-10T09:00:00Z", role="user"),
        MemoryRecord(id="m2", text="ok", timestamp="2026-01-10T09:00:01Z",
                     role="assistant"),
    ]
    ann = annotate_memories(recs, embedder=FakeEmbedder(), gold_ids={"m1"})
    assert len(ann) == 2
    f0 = ann[0]["factors"]
    assert set(f0) == {"emotion", "goal_relevance", "value_alignment",
                       "self_relevance", "task_utility", "reliability", "usage"}
    assert ann[0]["gold"] is True
    assert ann[1]["gold"] is False
    assert ann[0]["factors"]["reliability"] > ann[1]["factors"]["reliability"]
    for a in ann:
        for v in a["factors"].values():
            assert 0.0 <= v <= 1.0


def test_no_raw_text_leaks():
    from lmfm.io.markdown import MemoryRecord
    recs = [
        MemoryRecord(id="m1", text="User is allergic to penicillin.",
                     timestamp="t", role="user"),
        MemoryRecord(id="m2", text="secret project codename Falcon",
                     timestamp="t", role="assistant"),
    ]
    ann = annotate_memories(recs, embedder=FakeEmbedder())
    blob = json.dumps(ann)
    for r in recs:
        assert r.text not in blob
    for a in ann:
        assert set(a) == {"id", "ts", "gold", "factors"}
