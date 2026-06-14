"""
Emotion Signal Extractor

Converts raw user message text + conversation context into (ΔV, ΔA) deltas
for the Russell circumplex emotional state.

Three feature layers:
  1. Linguistic  — tone, sentiment, hedging, intensity (single message)
  2. Structural  — message length trends, topic continuity, response patterns
  3. Contextual  — injected externally as baseline shifts

All weights are calibrated so a single ordinary turn produces |ΔV|, |ΔA| < 0.15.
Capped at ±0.4 per turn to prevent emotional state jumps.

Linguistic rules use a typed object model — `KeywordRule` and `RegexRule`
each carry a `.apply(text)` method.  This replaces an earlier
type-sniffing dispatch (`isinstance + startswith + metachar scan`) that
mixed three code paths in one loop.  Adding a new rule is now a matter
of choosing the right class and appending an instance to the list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ── Typed rule objects ────────────────────────────────────────────────────

@dataclass(frozen=True)
class KeywordRule:
    """Fires (dv, da) once if ANY keyword appears in the message (case-insensitive)."""
    keywords: tuple[str, ...]
    dv: float
    da: float
    desc: str = ""

    def apply(self, text: str) -> tuple[float, float]:
        text_lower = text.lower()
        if any(kw.lower() in text_lower for kw in self.keywords):
            return self.dv, self.da
        return 0.0, 0.0


@dataclass(frozen=True)
class RegexRule:
    """
    Fires (dv, da) scaled by match count if the pattern matches.

    `diminishing` (default True) uses re.findall with weight =
    min(n_matches, 3) / 3 so repeated punctuation doesn't blow up the
    delta. Set False for binary semantics (fired or not).
    """
    pattern: str
    dv: float
    da: float
    desc: str = ""
    diminishing: bool = True

    def apply(self, text: str) -> tuple[float, float]:
        try:
            if self.diminishing:
                matches = re.findall(self.pattern, text, re.IGNORECASE)
                if matches:
                    weight = min(len(matches), 3) / 3.0
                    return self.dv * weight, self.da * weight
                return 0.0, 0.0
            if re.search(self.pattern, text, re.IGNORECASE):
                return self.dv, self.da
        except re.error:
            pass
        return 0.0, 0.0


# ── Linguistic rules (single-message features) ────────────────────────────

LINGUISTIC_RULES: list = [
    # Arousal boosters — punctuation density
    RegexRule(r"[！!]{2,}", 0.00, +0.20, "multiple exclamations"),
    RegexRule(r"[！!]",      0.00, +0.10, "exclamation"),
    RegexRule(r"[？?]{2,}",  0.00, +0.15, "multiple question marks"),
    RegexRule(r"[A-Z]{4,}",  0.00, +0.15, "ALL CAPS English"),

    # Valence: positive markers
    KeywordRule(
        ("谢谢", "感谢", "太好了", "棒", "完美", "厉害", "感激",
         "thank", "thanks", "great", "perfect", "awesome", "excellent",
         "love it", "well done", "appreciate"),
        +0.20, +0.05, "gratitude / praise",
    ),
    KeywordRule(
        ("哈哈", "哈哈哈", "lol", "haha", "😄", "😊", "🎉", "👍"),
        +0.15, +0.10, "laughter / positive emoji",
    ),
    KeywordRule(
        ("明白了", "清楚了", "懂了", "理解了",
         "got it", "makes sense", "understood"),
        +0.10, 0.00, "comprehension confirmation",
    ),

    # Valence: negative markers
    KeywordRule(
        ("又", "还是", "怎么还", "为什么还", "依然", "仍然",
         "again", "still not", "still wrong", "once more"),
        -0.20, +0.10, "repetition frustration",
    ),
    KeywordRule(
        ("烦", "讨厌", "糟糕", "垃圾", "废物", "不行",
         "terrible", "awful", "useless", "broken", "wrong", "bad"),
        -0.25, +0.05, "negative sentiment",
    ),
    KeywordRule(
        ("不对", "错了", "不是这个意思", "没有解决",
         "that's not", "that's wrong", "not what i", "didn't fix"),
        -0.15, +0.10, "correction / disagreement",
    ),
    KeywordRule(
        ("😠", "😤", "😡", "💢", "🤦"),
        -0.20, +0.15, "negative emoji",
    ),

    # Hedging → lower arousal
    KeywordRule(
        ("可能", "也许", "大概", "或许", "不确定", "不太清楚",
         "maybe", "perhaps", "not sure", "might", "could be", "i think"),
        0.00, -0.15, "hedging words",
    ),
    RegexRule(r"\.{3,}", 0.00, -0.10, "ellipsis (hesitation)"),

    # Intensity / certainty → higher arousal
    KeywordRule(
        ("非常", "极其", "绝对", "肯定", "一定", "必须",
         "definitely", "absolutely", "certainly", "must", "always"),
        0.00, +0.15, "intensifiers",
    ),

    # Topic-shift / curiosity signal
    KeywordRule(
        ("有个新想法", "换个话题", "顺便问", "另外", "还有个问题",
         "new idea", "by the way", "another thing", "also wondering"),
        +0.05, +0.10, "new curiosity signal",
    ),
]


# ── Structural context ────────────────────────────────────────────────────

@dataclass
class StructuralContext:
    current_length: int
    prev_lengths: list[int]         # last N message lengths
    is_topic_continuation: bool     # same topic as previous turn?
    repeated_question: bool         # same question asked before?
    detail_follow_up: bool          # asking for more detail?
    one_word_response: bool         # user sent ≤2 tokens?


class EmotionSignalExtractor:
    """
    Extracts (ΔV, ΔA) emotional deltas from a user message.

    Usage:
        extractor = EmotionSignalExtractor()
        dv, da = extractor.extract(message, conversation_history)
    """

    # Hard cap per turn to prevent state jumps
    MAX_DELTA_V = 0.40
    MAX_DELTA_A = 0.30

    def extract(
        self,
        message: str,
        conversation_history: list[dict],
    ) -> tuple[float, float]:
        """Returns (delta_v, delta_a) capped and normalised."""
        dv, da = 0.0, 0.0

        dv_l, da_l = self._linguistic(message)
        dv += dv_l
        da += da_l

        ctx = self._build_structural_context(message, conversation_history)
        dv_s, da_s = self._structural(ctx)
        dv += dv_s
        da += da_s

        dv = max(-self.MAX_DELTA_V, min(self.MAX_DELTA_V, dv))
        da = max(-self.MAX_DELTA_A, min(self.MAX_DELTA_A, da))
        return round(dv, 4), round(da, 4)

    # ── Linguistic layer ──────────────────────────────────────────────────

    @staticmethod
    def _linguistic(text: str) -> tuple[float, float]:
        dv, da = 0.0, 0.0
        for rule in LINGUISTIC_RULES:
            rdv, rda = rule.apply(text)
            dv += rdv
            da += rda
        return dv, da

    # ── Structural layer ──────────────────────────────────────────────────

    def _build_structural_context(
        self,
        message: str,
        history: list[dict],
    ) -> StructuralContext:
        user_msgs = [
            m for m in history
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        prev_lengths = [len(m["content"]) for m in user_msgs[-5:]]
        cur_len = len(message)

        length_increasing = (
            len(prev_lengths) >= 2 and cur_len > prev_lengths[-1] * 1.3
        )
        length_decreasing = (
            len(prev_lengths) >= 2 and cur_len < prev_lengths[-1] * 0.5
        )

        recent_user_texts = [m["content"].strip().lower() for m in user_msgs[-4:]]
        msg_lower = message.strip().lower()
        repeated = sum(1 for t in recent_user_texts if self._similar(t, msg_lower)) >= 1

        detail_patterns = (
            "能详细", "详细说", "举个例子", "具体", "能展开",
            "more detail", "explain more", "elaborate", "example", "can you expand",
        )
        detail = any(p in message.lower() for p in detail_patterns)

        is_continuation = True
        if len(user_msgs) >= 2:
            prev_words = set(user_msgs[-1]["content"].lower().split())
            cur_words = set(message.lower().split())
            overlap = len(prev_words & cur_words) / max(len(prev_words), 1)
            is_continuation = overlap > 0.15

        one_word = len(message.split()) <= 2

        return StructuralContext(
            current_length=cur_len,
            prev_lengths=prev_lengths,
            is_topic_continuation=is_continuation,
            repeated_question=repeated,
            detail_follow_up=detail,
            one_word_response=one_word,
        )

    @staticmethod
    def _structural(ctx: StructuralContext) -> tuple[float, float]:
        dv, da = 0.0, 0.0
        prev = ctx.prev_lengths

        if len(prev) >= 2 and ctx.current_length > prev[-1] * 1.3:
            dv += 0.10; da += 0.12
        if len(prev) >= 2 and ctx.current_length < prev[-1] * 0.5:
            dv -= 0.05; da -= 0.12
        if ctx.is_topic_continuation:
            dv += 0.05; da += 0.05
        else:
            dv -= 0.08; da += 0.08
        if ctx.repeated_question:
            dv -= 0.20; da += 0.12
        if ctx.detail_follow_up:
            dv += 0.10; da += 0.12
        if ctx.one_word_response:
            dv -= 0.05; da -= 0.18

        return dv, da

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _similar(a: str, b: str) -> bool:
        """Rough similarity: enough shared words."""
        wa = set(a.split())
        wb = set(b.split())
        if not wa or not wb:
            return False
        overlap = len(wa & wb) / min(len(wa), len(wb))
        return overlap > 0.6
