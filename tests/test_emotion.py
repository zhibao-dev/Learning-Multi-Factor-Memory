from lmfm.factors.emotion import EmotionSignalExtractor


def test_extract_returns_capped_deltas():
    ex = EmotionSignalExtractor()
    dv, da = ex.extract("This is absolutely terrible and broken!", [])
    assert -0.40 <= dv <= 0.40
    assert -0.30 <= da <= 0.30


def test_neutral_text_low_signal():
    ex = EmotionSignalExtractor()
    dv, da = ex.extract("The file is at path x.", [])
    assert abs(dv) < 0.2
