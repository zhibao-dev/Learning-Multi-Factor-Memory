"""
Tests for the v0.2 pluggable embedder on SelfModel.

The default embedder (hash_embed) must keep its v0.1 semantics so all
existing tests pass. Custom embedders (SBertEmbedder or arbitrary
callables) must be honoured throughout the SelfModel surface.
"""
from __future__ import annotations

import pytest


def test_default_embedder_is_hash_embed():
    """SelfModel.empty() with no embedder uses hash_embed and 64-dim."""
    from borge.values.self_model import SelfModel
    m = SelfModel.empty()
    e = m._embed("hello world")
    assert isinstance(e, list)
    assert len(e) == 64
    # hash_embed is deterministic
    assert m._embed("hello world") == e


def test_custom_embedder_is_used():
    """Custom callable embedder overrides the default."""
    from borge.values.self_model import SelfModel

    received: list[str] = []
    def tagging_embedder(text: str) -> list[float]:
        received.append(text)
        # 4-dim signature with the text length as the first component.
        return [float(len(text)), 0.0, 0.0, 0.0]
    m = SelfModel(embedder=tagging_embedder)
    out = m._embed("hello")
    assert out == [5.0, 0.0, 0.0, 0.0]
    assert received == ["hello"]
    # Confirm SelfModel never falls back to hash_embed when embedder is set.
    out2 = m._embed("world!")
    assert out2 == [6.0, 0.0, 0.0, 0.0]
    assert received == ["hello", "world!"]


def test_from_seed_adapts_dim_to_embedder():
    """If a custom embedder returns 384-dim, SelfModel.dim auto-adjusts."""
    from borge.values.self_model import SelfModel
    def big_embedder(text: str) -> list[float]:
        return [0.5] * 128
    m = SelfModel.from_seed("seed", embedder=big_embedder)
    assert m.dim == 128
    assert len(m.mu_self) == 128


def test_back_compat_embed_alias_still_works():
    """`from borge.values.self_model import embed` still resolves."""
    from borge.values.self_model import embed, hash_embed
    assert embed is hash_embed
    assert len(embed("hello", dim=32)) == 32


# ── SBertEmbedder (skipped if sentence-transformers not installed) ──────

def _sbert_available() -> bool:
    try:
        import sentence_transformers  # noqa
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _sbert_available(), reason="sentence-transformers not installed")
def test_sbert_embedder_produces_384_dim_vector():
    from borge.values.self_model import SBertEmbedder
    sbert = SBertEmbedder()
    vec = sbert("I value intellectual honesty")
    assert isinstance(vec, list)
    assert len(vec) == 384      # MiniLM-L6-v2 default
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.skipif(not _sbert_available(), reason="sentence-transformers not installed")
def test_sbert_self_relevance_separates_self_from_nonself():
    """
    The killer use case: SBert can rank a self-referential prompt above
    an unrelated structural prompt for an agent whose seed is "I am honest careful learner".
    This is the property that bag-of-tokens cannot achieve.
    """
    from borge.values.self_model import SelfModel, SBertEmbedder
    sbert = SBertEmbedder()
    m = SelfModel.from_seed("I am an honest careful learner who values truth",
                            embedder=sbert)
    sr_self     = m.self_relevance_of("Does the word 'honest' describe me?")
    sr_phonemic = m.self_relevance_of("Does the word 'honest' rhyme with 'wind'?")
    assert sr_self > sr_phonemic, (
        f"SBert should rank self-ref > phonemic; got self={sr_self:.4f}, "
        f"phon={sr_phonemic:.4f}"
    )
