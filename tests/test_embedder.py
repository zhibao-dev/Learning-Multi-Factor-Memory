import pytest
from lmfm.factors.embedder import cosine


def test_cosine_basic():
    assert round(cosine([1, 0], [1, 0]), 5) == 1.0
    assert round(cosine([1, 0], [0, 1]), 5) == 0.0


def test_embedder_importable_without_st():
    from lmfm.factors.embedder import SBertEmbedder
    SBertEmbedder()  # construction must NOT import sentence-transformers (lazy)
