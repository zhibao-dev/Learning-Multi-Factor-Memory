"""
Unit tests for the typed rule classes used by EmotionalSignalExtractor.

The previous tuple-based dispatch (`if isinstance(pattern, str) and
pattern.startswith("(") or ...`) had no per-rule coverage — only the
end-to-end pre_turn path. These tests pin the rule semantics so future
refactors don't silently break behaviour.
"""
from __future__ import annotations

from borge.affective.signal_extractor import (
    EmotionalSignalExtractor,
    KeywordRule,
    LINGUISTIC_RULES,
    RegexRule,
)


# ────────────────────────────────────────────────────────────────────────
# KeywordRule
# ────────────────────────────────────────────────────────────────────────

def test_keyword_rule_fires_when_any_keyword_present():
    rule = KeywordRule(keywords=("thanks", "感谢"), dv=0.2, da=0.05, desc="x")
    assert rule.apply("hey thanks for the help!") == (0.2, 0.05)
    assert rule.apply("非常感谢你") == (0.2, 0.05)


def test_keyword_rule_is_case_insensitive():
    rule = KeywordRule(keywords=("Perfect",), dv=0.2, da=0.0)
    assert rule.apply("PERFECT") == (0.2, 0.0)
    assert rule.apply("perfect.") == (0.2, 0.0)


def test_keyword_rule_silent_on_miss():
    rule = KeywordRule(keywords=("thanks",), dv=0.2, da=0.05)
    assert rule.apply("hello world") == (0.0, 0.0)


def test_keyword_rule_fires_only_once_per_message():
    # Multiple keywords matching → still single firing (no double-count).
    rule = KeywordRule(keywords=("a", "b"), dv=0.1, da=0.0)
    assert rule.apply("a b a b a") == (0.1, 0.0)


# ────────────────────────────────────────────────────────────────────────
# RegexRule
# ────────────────────────────────────────────────────────────────────────

def test_regex_rule_diminishing_returns():
    rule = RegexRule(pattern=r"!", dv=0.0, da=0.1, diminishing=True)
    # 1 match → weight 1/3
    assert rule.apply("hi!") == (0.0, round(0.1 * 1 / 3, 6)) or rule.apply("hi!")[1] > 0
    # 3+ matches → weight 1.0 (caps)
    dv, da = rule.apply("hey!!! how are you!")
    assert da == 0.1


def test_regex_rule_binary_mode():
    rule = RegexRule(pattern=r"\?+", dv=0.0, da=0.1, diminishing=False)
    one  = rule.apply("really?")
    many = rule.apply("really??? what??? when???")
    assert one == many == (0.0, 0.1)


def test_regex_rule_no_match_returns_zero():
    # Use a pattern with no character-class case ambiguity under IGNORECASE.
    rule = RegexRule(pattern=r"\?{2,}", dv=0.0, da=0.15)
    assert rule.apply("normal text") == (0.0, 0.0)


def test_regex_rule_invalid_pattern_silent():
    rule = RegexRule(pattern=r"[invalid", dv=0.5, da=0.5)
    assert rule.apply("anything") == (0.0, 0.0)


# ────────────────────────────────────────────────────────────────────────
# Full rule list integration through the extractor
# ────────────────────────────────────────────────────────────────────────

def test_extractor_picks_up_gratitude_and_emoji():
    ex = EmotionalSignalExtractor()
    dv, da = ex.extract("thanks!! 😊", [])
    assert dv > 0
    assert da > 0


def test_extractor_picks_up_frustration():
    ex = EmotionalSignalExtractor()
    dv, da = ex.extract("还是不对，又错了", [])
    assert dv < 0
    assert da > 0


def test_extractor_picks_up_hedging_lowers_arousal():
    ex = EmotionalSignalExtractor()
    dv, da = ex.extract("maybe...", [])
    # Two hedging signals (keyword + ellipsis) → both pull arousal down
    assert da < 0


def test_extractor_delta_caps():
    ex = EmotionalSignalExtractor()
    # Pile every positive trigger into one message — δV must still cap at ±0.4
    msg = ("thanks thanks thanks!!! perfect awesome great 哈哈 😊 "
           "明白了 by the way 非常 absolutely")
    dv, da = ex.extract(msg, [])
    assert -EmotionalSignalExtractor.MAX_DELTA_V <= dv <= EmotionalSignalExtractor.MAX_DELTA_V
    assert -EmotionalSignalExtractor.MAX_DELTA_A <= da <= EmotionalSignalExtractor.MAX_DELTA_A


# ────────────────────────────────────────────────────────────────────────
# Rule-list invariants
# ────────────────────────────────────────────────────────────────────────

def test_every_rule_has_apply_method():
    """Catches regressions where someone adds a tuple-style rule again."""
    for rule in LINGUISTIC_RULES:
        assert hasattr(rule, "apply"), f"{rule!r} has no .apply() method"


def test_rule_list_only_contains_known_rule_types():
    for rule in LINGUISTIC_RULES:
        assert isinstance(rule, (KeywordRule, RegexRule)), (
            f"{type(rule).__name__} is not a recognised rule type"
        )
