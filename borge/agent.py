"""
BorgeAgent — Cognitive Layer Core

Provides a framework-agnostic cognitive layer:
  - Affective state tracking (Russell circumplex)
  - Bayesian belief state
  - Active inference tool scoring (EFE)
  - Cognitive memory pipeline hooks
  - Extended free energy monitoring (MetaAgent)
  - Value system integration

Standalone usage (with BorgeRunner):
    from borge import BorgeRunner
    runner = BorgeRunner()
    runner.run("help me debug this")

Plugin usage (wrapping an existing agent backend):
    from borge.agent import BorgeAgent
    cognitive = BorgeAgent(agent_backend=my_agent)
    ctx = cognitive.pre_turn(user_message, history)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from .affective.emotional_state import AgentMode, EmotionalState
from .affective.loyalty_tracker import LoyaltyTracker, SessionSummary
from .affective.signal_extractor import EmotionalSignalExtractor
from .beliefs.belief_state import BeliefState
from .inference.active_inference import ActiveInferenceEngine
from .memory.consolidation import MemoryConsolidationPipeline
from .memory.forgetting import ForgettingEngine
from .memory.knowledge_graph import KnowledgeGraph
from .memory.retrieval import MemoryRetrieval
from .memory.store import MemoryStore
from .meta.free_energy import ExtendedFreeEnergy
from .meta.meta_agent import MetaAgent
from .values.self_model import SelfModel
from .values.soul_parser import parse_soul_frontmatter
from .values.value_system import ValueSystem

log = logging.getLogger(__name__)


class BorgeAgent:
    """
    Borge cognitive layer — framework-agnostic cognitive overlay.

    Can wrap any agent backend (Hermes, OpenClaw, raw Anthropic SDK, etc.)
    or run standalone via BorgeRunner.  All cognitive state lives here;
    the backend only handles LLM I/O and tool execution.
    """

    def __init__(
        self,
        agent_backend=None,                    # optional backend (any agent instance)
        db_path: Optional[str] = None,
        soul_path: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self._agent = agent_backend
        self._config = config or {}
        self._db_path = db_path or self._resolve_db_path()

        # ── Core cognitive state ──────────────────────────────────────────
        self.values: ValueSystem = parse_soul_frontmatter(
            soul_path or self._resolve_soul_path()
        )
        self.emotion: EmotionalState = EmotionalState(
            valence_baseline=self.values.emotional_defaults.valence_baseline,
            arousal_baseline=self.values.emotional_defaults.arousal_baseline,
            tau_valence=self.values.emotional_defaults.tau_valence,
            tau_arousal=self.values.emotional_defaults.tau_arousal,
            frustrated_threshold=self.values.emotional_defaults.frustrated_threshold,
            excited_threshold=self.values.emotional_defaults.excited_threshold,
        )
        self.beliefs: BeliefState = BeliefState()

        # ── Self model (FEP) ─────────────────────────────────────────────
        # Bootstrap μ_self from SOUL.md value descriptors when available,
        # else start empty (μ_self grows from first user turn).
        seed = " ".join(
            f"{v.id} {v.description}"
            for v in (self.values.primary_values or [])
        ).strip()
        self.self_model: SelfModel = (
            SelfModel.from_seed(seed) if seed else SelfModel.empty()
        )

        # ── Engines ──────────────────────────────────────────────────────
        self._signal_extractor = EmotionalSignalExtractor()
        self._loyalty_tracker  = LoyaltyTracker()
        self._meta             = MetaAgent(
            free_energy_fn=ExtendedFreeEnergy(),
            entropy_injection_threshold=self._cfg("beliefs.entropy_injection_threshold", 0.5),
        )
        self._afe = ActiveInferenceEngine(self.beliefs, self.emotion)

        # ── Memory infrastructure ─────────────────────────────────────────
        self._kg = KnowledgeGraph(self._db_path)
        self._forgetting = ForgettingEngine(
            prune_threshold=self._cfg("memory.forgetting.prune_threshold", 2.0),
        )
        self._memory_store = MemoryStore(self._db_path)
        self._consolidation = MemoryConsolidationPipeline(
            db_path=self._db_path,
            knowledge_graph=self._kg,
            llm_caller=None,
            forgetting_engine=self._forgetting,
            memory_store=self._memory_store,
            self_model=self.self_model,
        )
        self._retrieval = MemoryRetrieval(
            self._db_path,
            store=self._memory_store,
            self_model=self.self_model,
        )

        # ── Session state ─────────────────────────────────────────────────
        self._emotional_history: list[tuple[float, float]] = []
        self._session_f_history: list[float] = []
        self._turn_count: int = 0

        log.info("[BorgeAgent] Initialised")

    # ── Session start ─────────────────────────────────────────────────────

    def on_session_start(self, user_id: Optional[str] = None) -> None:
        if user_id:
            sessions = self._fetch_past_sessions(user_id)
            v_base, a_base = self._loyalty_tracker.compute_baseline(sessions)
            self.emotion.set_baseline(v_base, a_base)
            tier = self._loyalty_tracker.tier(v_base)
            log.info(f"[BorgeAgent] Loyalty: {tier.value} (V_base={v_base:.2f})")

        self._emotional_history.clear()
        self._session_f_history.clear()
        self._turn_count = 0
        self.beliefs = BeliefState()
        self._meta.reset()

    # ── Pre-turn hook ─────────────────────────────────────────────────────

    def pre_turn(
        self,
        user_message: str,
        conversation_history: list[dict],
    ) -> str:
        self._turn_count += 1

        dv, da = self._signal_extractor.extract(user_message, conversation_history)
        self.emotion.update(dv, da)
        self._emotional_history.append((self.emotion.valence, self.emotion.arousal))
        log.debug(f"[Emotion] {self.emotion}")

        if self._turn_count == 1:
            self.beliefs.task = user_message[:200]

        loyalty_hint = self._loyalty_tracker.system_prompt_hint(
            self.emotion.valence_baseline
        )

        signal = self._meta.tick(
            self.beliefs, self.emotion, self.values,
            loyalty_hint=loyalty_hint,
        )
        # Track F_total per user turn so that consolidation can stamp each
        # memory with f_total_at_encoding and compute delta_f vs. the
        # previous turn (positive Δ = the agent made progress).
        self._session_f_history.append(signal.f_total)

        mode = signal.suggested_mode
        if mode != AgentMode.NORMAL:
            log.info(f"[BorgeAgent] Mode: {mode.value}")

        return signal.context_injection

    # ── Post-tool hook ────────────────────────────────────────────────────

    def post_tool(
        self,
        tool_name: str,
        tool_result: str,
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> None:
        if self.beliefs.hypotheses:
            self.beliefs.bayesian_update(tool_result, tool_name, llm_caller)
            log.debug(f"[Belief] entropy={self.beliefs.shannon_entropy():.2f}bits")

        self.values.update_satisfaction(tool_result)

    # ── Tool scoring ──────────────────────────────────────────────────────

    def score_tool_candidates(
        self,
        candidates: list[dict],
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> list[dict]:
        if not candidates:
            return candidates

        self._afe.beliefs = self.beliefs
        self._afe.emotion = self.emotion
        scored = self._afe.score_and_rank(candidates, llm_caller)

        scored_names = [s.tool_name for s in scored]
        return sorted(
            candidates,
            key=lambda c: scored_names.index(c.get("name", ""))
                          if c.get("name") in scored_names else 999,
        )

    # ── Session end ───────────────────────────────────────────────────────

    def on_session_end(
        self,
        session_id: str,
        messages: list[dict],
    ) -> None:
        log.info(f"[BorgeAgent] Running consolidation for session {session_id}")
        report = self._consolidation.run(
            session_id=session_id,
            messages=messages,
            emotional_history=self._emotional_history,
            f_history=self._session_f_history,
        )
        log.info(
            f"[Consolidation] entities={report.entities_extracted} "
            f"relations={report.relations_added} "
            f"skills={report.skill_candidates} "
            f"forgotten={report.entries_forgotten}"
        )

    # ── Memory retrieval ──────────────────────────────────────────────────

    def recall(self, query: str = "", k: int = 5) -> list[dict]:
        """
        Mood-congruent recall against persisted memories.

        Ranks every row in `borge_memories` by a weighted combination of:
          • Gaussian similarity between (V, A) at encoding vs. the agent's
            current emotional state
          • recency (~weekly soft half-life)
          • token-overlap relevance to `query`
          • positive delta_f bonus (memories formed during progress)

        Each returned memory's `retrieval_count` is bumped, so frequent
        recall makes a memory harder to forget (closes the retrieval ↔
        forgetting loop).

        Returns the top-k rows as dicts.
        """
        cur_f = self._session_f_history[-1] if self._session_f_history else None
        return self._retrieval.recall(
            query=query,
            current_valence=self.emotion.valence,
            current_arousal=self.emotion.arousal,
            current_f_total=cur_f,
            k=k,
        )

    def build_system_prompt_suffix(self) -> str:
        signal = self._meta.tick(self.beliefs, self.emotion, self.values)
        return signal.context_injection

    # ── Internal helpers ──────────────────────────────────────────────────

    def _cfg(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        node = self._config
        for p in parts:
            if not isinstance(node, dict):
                return default
            node = node.get(p, default)
        return node if node is not None else default

    def _resolve_db_path(self) -> str:
        home = os.path.expanduser(os.environ.get("BORGE_HOME", "~/.borge"))
        os.makedirs(home, exist_ok=True)
        return os.path.join(home, "borge.db")

    def _resolve_soul_path(self) -> str:
        candidates = [
            os.path.join(os.getcwd(), "SOUL.md"),
            os.path.expanduser(os.path.join(os.environ.get("BORGE_HOME", "~/.borge"), "SOUL.md")),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[-1]

    def _fetch_past_sessions(self, user_id: str) -> list[SessionSummary]:
        import sqlite3
        summaries = []
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT id, created_at,
                              COALESCE(emotional_valence, 0.0) as avg_valence,
                              COALESCE(emotional_arousal, 0.5) as avg_arousal,
                              COALESCE(message_count, 1) as message_count
                       FROM sessions
                       WHERE source_user_id = ?
                       ORDER BY created_at DESC LIMIT 50""",
                    (user_id,),
                ).fetchall()
                from datetime import datetime
                for r in rows:
                    try:
                        summaries.append(SessionSummary(
                            session_id=r["id"],
                            created_at=datetime.fromisoformat(r["created_at"]),
                            avg_valence=r["avg_valence"],
                            avg_arousal=r["avg_arousal"],
                            message_count=r["message_count"],
                        ))
                    except Exception:
                        pass
        except Exception as e:
            log.debug(f"[BorgeAgent] Could not fetch sessions: {e}")
        return summaries
