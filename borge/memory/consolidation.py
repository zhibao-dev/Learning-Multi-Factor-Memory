"""
Memory Consolidation Pipeline

Offline pipeline run after each session (or on cron schedule).
Corresponds to sleep-based memory consolidation in cognitive neuroscience.

Steps:
  1. Entity & Relation Extraction    → raw material for semantic memory
  2. Schema Matching                 → integrate with existing knowledge graph
  3. Emotional Significance Update   → persist (V, A, depth, ΔF) to borge_memories
  4. ΔF_total → importance bonus     → progress-bearing turns resist forgetting
  5. Skill Candidate Detection       → find reusable procedural patterns
  6. Active Forgetting               → apply Ebbinghaus decay, prune SHALLOW entries
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from .cognitive_memory import EncodingDepth, MemoryEntry
from .forgetting import ForgettingEngine, apply_importance_from_delta_f
from .knowledge_graph import KnowledgeGraph
from .store import MemoryStore
from ..values.self_model import SelfModel, cosine, has_self_reference

log = logging.getLogger(__name__)


@dataclass
class ConsolidationReport:
    session_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    entities_extracted: int = 0
    relations_added: int = 0
    skill_candidates: list[str] = field(default_factory=list)
    entries_forgotten: int = 0
    entries_compressed: int = 0
    errors: list[str] = field(default_factory=list)


class MemoryConsolidationPipeline:
    """
    Orchestrates the 7-step memory consolidation process.

    Typical integration: called from a Cron job or session_end hook.

        pipeline = MemoryConsolidationPipeline(db_path, kg)
        report = pipeline.run(session_id, messages, emotional_history)
    """

    def __init__(
        self,
        db_path: str,
        knowledge_graph: KnowledgeGraph,
        llm_caller: Optional[Callable[[str], str]] = None,
        forgetting_engine: Optional[ForgettingEngine] = None,
        memory_store: Optional[MemoryStore] = None,
        self_model: Optional[SelfModel] = None,
        value_centroid: Optional[list[float]] = None,
    ):
        self.db_path = db_path
        self.kg = knowledge_graph
        self.llm = llm_caller
        self.forgetting = forgetting_engine or ForgettingEngine()
        self.store = memory_store or MemoryStore(db_path)
        # Optional FEP self model — when provided, Step 3 also computes
        # self_relevance per message and updates μ_self / π_self from the
        # session's user content.
        self.self_model = self_model
        # Sister paper (multi-factor value): centroid of the agent's value
        # embeddings (e.g. mean of SOUL value-descriptor embeddings). When
        # provided, Step 3 computes value_alignment = cos(content, centroid);
        # otherwise value_alignment degrades to 0.0.
        self.value_centroid = value_centroid

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        session_id: str,
        messages: list[dict],
        emotional_history: Optional[list[tuple[float, float]]] = None,
        f_history: Optional[list[float]] = None,
    ) -> ConsolidationReport:
        """
        Run all 7 pipeline steps for a completed session.
        Returns a ConsolidationReport with step-by-step metrics.

        Parameters
        ----------
        emotional_history
            Per-user-turn (valence, arousal) snapshots. Aligned with USER
            messages in order; assistant/tool messages inherit the latest.
        f_history
            Per-user-turn F_total snapshots. Same alignment as
            emotional_history. Used to compute delta_f_total (positive =
            progress made) and stamp encoding metadata.
        """
        report = ConsolidationReport(session_id=session_id)
        log.info(f"[Consolidation] Starting for session {session_id}")

        try:
            # Step 1: Entity & relation extraction
            entities, relations = self._step1_extract(messages, report)

            # Step 2: Schema matching & KG update
            self._step2_update_kg(entities, relations, report)

            # Step 3: Emotional significance update — persist to borge_memories
            self._step3_emotional_significance(
                session_id, messages, emotional_history, f_history, report
            )

            # Step 4: ΔF_total → importance bonus
            apply_importance_from_delta_f(self.db_path, gain=0.3)

            # Step 5: Skill candidate detection
            self._step5_detect_skills(messages, report)

            # Step 6: Active forgetting
            self._step6_forgetting(session_id, report)

        except Exception as e:
            log.error(f"[Consolidation] Pipeline error: {e}")
            report.errors.append(str(e))

        log.info(
            f"[Consolidation] Done. "
            f"entities={report.entities_extracted} "
            f"relations={report.relations_added} "
            f"skills={len(report.skill_candidates)} "
            f"forgotten={report.entries_forgotten}"
        )
        return report

    # ── Step 1: Entity & Relation Extraction ─────────────────────────────

    def _step1_extract(
        self,
        messages: list[dict],
        report: ConsolidationReport,
    ) -> tuple[list[dict], list[dict]]:
        text = self._messages_to_text(messages)

        if self.llm:
            entities, relations = self._llm_extract(text)
        else:
            entities, relations = self._heuristic_extract(text)

        report.entities_extracted = len(entities)
        return entities, relations

    def _llm_extract(self, text: str) -> tuple[list[dict], list[dict]]:
        prompt = f"""Extract entities and relationships from this conversation.

Conversation (excerpt):
{text[:2000]}

Return JSON:
{{
  "entities": [
    {{"type": "Concept|Person|Task|File|Preference|Fact", "label": "...", "properties": {{}}}}
  ],
  "relations": [
    {{"source": "<label>", "target": "<label>", "relation": "relates_to|requires|contradicts|..."}}
  ]
}}"""
        try:
            raw = self.llm(prompt)
            data = json.loads(raw)
            return data.get("entities", []), data.get("relations", [])
        except Exception as e:
            log.warning(f"[Step1] LLM extraction failed: {e}")
            return self._heuristic_extract(text)

    def _heuristic_extract(self, text: str) -> tuple[list[dict], list[dict]]:
        """Simple regex-based extraction for common patterns."""
        entities = []
        # File paths
        for path in re.findall(r'[\w/\-]+\.\w{2,4}', text):
            entities.append({"type": "File", "label": path, "properties": {}})
        # Quoted concepts
        for concept in re.findall(r'"([^"]{3,40})"', text):
            entities.append({"type": "Concept", "label": concept, "properties": {}})
        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            if e["label"] not in seen:
                seen.add(e["label"])
                unique.append(e)
        return unique[:20], []   # relations require LLM

    # ── Step 2: KG Update ─────────────────────────────────────────────────

    def _step2_update_kg(
        self,
        entities: list[dict],
        relations: list[dict],
        report: ConsolidationReport,
    ) -> None:
        label_to_id: dict[str, str] = {}
        for e in entities:
            node_id = self.kg.upsert_node(
                entity_type=e.get("type", "Concept"),
                label=e["label"],
                properties=e.get("properties", {}),
            )
            label_to_id[e["label"]] = node_id

        for r in relations:
            src_id = label_to_id.get(r.get("source", ""))
            tgt_id = label_to_id.get(r.get("target", ""))
            if src_id and tgt_id:
                self.kg.add_edge(src_id, tgt_id, r.get("relation", "relates_to"))
                report.relations_added += 1

    # ── Step 3: Emotional Significance ───────────────────────────────────

    def _step3_emotional_significance(
        self,
        session_id: str,
        messages: list[dict],
        emotional_history: Optional[list[tuple[float, float]]],
        f_history: Optional[list[float]],
        report: ConsolidationReport,
    ) -> None:
        """
        For every message:
          1. Pair with the (V, A) of the user-turn that produced it
             (assistant/tool messages inherit the most recent user emotion).
          2. Compute emotional significance = |V| · A.
          3. Map significance → EncodingDepth (Craik & Lockhart).
          4. Compute delta_f_total vs. previous turn (progress signal).
          5. Persist row to borge_memories so Forgetting + Retrieval can see it.
        """
        if not emotional_history:
            return

        # Pre-pass — session topic centroid for goal_relevance. The session's
        # "goal" is approximated by the mean embedding of its USER turns; each
        # memory's goal_relevance is its similarity to that centroid. hash_embed
        # is deterministic, so re-embedding here costs nothing semantically.
        session_topic_centroid: Optional[list[float]] = None
        if self.self_model is not None:
            user_embeddings = [
                self.self_model._embed(self._content_text(m))
                for m in messages
                if (m.get("role") or "").lower() == "user"
            ]
            if user_embeddings:
                n = len(user_embeddings)
                dim = len(user_embeddings[0])
                session_topic_centroid = [
                    sum(e[i] for e in user_embeddings) / n for i in range(dim)
                ]

        emo_idx = -1     # advances on each USER message
        prev_f: Optional[float] = None
        persisted = 0

        for msg in messages:
            role = (msg.get("role") or "").lower()
            if role == "user":
                emo_idx += 1

            if emo_idx < 0 or emo_idx >= len(emotional_history):
                continue

            v, a = emotional_history[emo_idx]
            significance = abs(v) * a
            depth = (
                EncodingDepth.META       if significance >= 0.7 else
                EncodingDepth.SCHEMATIC  if significance >= 0.4 else
                EncodingDepth.SEMANTIC   if significance >= 0.2 else
                EncodingDepth.SHALLOW
            )

            f_total: Optional[float] = None
            delta_f: Optional[float] = None
            if f_history and emo_idx < len(f_history):
                f_total = f_history[emo_idx]
                # Positive delta = F dropped between previous and this turn
                # = the agent made cognitive progress on this turn.
                if prev_f is not None:
                    delta_f = prev_f - f_total
                if role == "user":
                    prev_f = f_total

            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            content = str(content or "")[:2000]

            # FEP self-relevance: embed → query sr → then update μ_self with
            # this turn's emotional_significance as the update weight.
            # Embedding goes through whatever embedder the SelfModel was
            # constructed with (hash_embed default, SBertEmbedder for v0.2
            # human-data experiments).
            embedding = None
            self_relevance = 0.5
            mu_self_snapshot: Optional[list[float]] = None
            if self.self_model is not None:
                embedding = self.self_model._embed(content)
                self_relevance = self.self_model.self_relevance(embedding)
                # μ_self updates only on user content that contains explicit
                # self-references ("I", "me", "my", "我", …). This makes the
                # self prior identity-constitutive rather than chasing every
                # topic mentioned by the user.
                if role == "user" and has_self_reference(content):
                    self.self_model.update(embedding, weight=max(significance, 0.1))
                # L5 — snapshot μ_self AFTER any update on this turn so the
                # row is encoding-specific: retrieval can later compare the
                # agent's current μ_self with the μ_self that prevailed when
                # this memory was formed (Tulving encoding specificity).
                mu_self_snapshot = list(self.self_model.mu_self) if self.self_model.mu_self else None

            # Self-modulated encoding depth — vivid AND self-relevant content
            # gets bumped one tier higher (capped at META).
            if self.self_model is not None and self_relevance > 0.65:
                depth = EncodingDepth(min(int(depth) + 1, int(EncodingDepth.META)))

            # Sister paper (multi-factor value): LIVE value factors computed at
            # encode time, where session + SOUL context exists. usage=0 (count
            # starts 0) and task_utility=0 (LLM-gated) need no work here.
            #   reliability      role heuristic (user content is first-hand)
            #   goal_relevance   similarity to this session's topic centroid
            #   value_alignment  similarity to the agent's value centroid
            reliability = 0.7 if role == "user" else 0.4
            if embedding is not None and session_topic_centroid:
                goal_relevance = 0.5 + 0.5 * cosine(embedding, session_topic_centroid)
            else:
                goal_relevance = 0.5
            if embedding is not None and self.value_centroid:
                value_alignment = 0.5 + 0.5 * cosine(embedding, self.value_centroid)
            else:
                value_alignment = 0.0
            goal_relevance = max(0.0, min(1.0, goal_relevance))
            value_alignment = max(0.0, min(1.0, value_alignment))

            memory_id = msg.get("id") or msg.get("_borge_id") or str(uuid.uuid4())
            self.store.insert({
                "id":                     memory_id,
                "session_id":             session_id,
                "role":                   role,
                "content":                content,
                "timestamp":              msg.get("timestamp") or datetime.now().isoformat(),
                "emotional_valence":      float(v),
                "emotional_arousal":      float(a),
                "emotional_significance": round(significance, 4),
                "encoding_depth":         int(depth),
                "f_total_at_encoding":    f_total,
                "delta_f_total":          delta_f,
                "self_relevance_score":   round(self_relevance, 4),
                "embedding":              embedding,
                "mu_self_at_encoding":    mu_self_snapshot,
                "reliability":            round(reliability, 4),
                "goal_relevance":         round(goal_relevance, 4),
                "value_alignment":        round(value_alignment, 4),
                "task_utility":           0.0,
            })
            persisted += 1

            # Keep in-memory hint for any caller that wants it
            msg["_borge_encoding_depth"] = int(depth)
            msg["_borge_significance"]   = round(significance, 4)
            msg["_borge_self_relevance"] = round(self_relevance, 4)
            msg["_borge_id"]              = memory_id

        log.debug(f"[Step5] persisted {persisted} memory rows")

    # ── Step 5: Skill Candidate Detection ────────────────────────────────

    def _step5_detect_skills(
        self,
        messages: list[dict],
        report: ConsolidationReport,
    ) -> None:
        """
        Detect procedural patterns that might be worth saving as skills.
        Heuristic: sessions with ≥5 tool calls that completed successfully.
        """
        tool_calls = [m for m in messages if m.get("role") == "tool"]
        if len(tool_calls) >= 5 and self.llm:
            text = self._messages_to_text(messages)
            prompt = f"""This conversation involved a multi-step task.
Identify if a reusable skill pattern was demonstrated.

Conversation:
{text[:1500]}

If a reusable procedure exists, return:
{{"skill_name": "short-name", "description": "what it does", "worth_saving": true}}
Otherwise: {{"worth_saving": false}}"""
            try:
                raw = self.llm(prompt)
                data = json.loads(raw)
                if data.get("worth_saving") and data.get("skill_name"):
                    report.skill_candidates.append(data["skill_name"])
            except Exception:
                pass

    # ── Step 6: Active Forgetting ─────────────────────────────────────────

    def _step6_forgetting(
        self,
        session_id: str,
        report: ConsolidationReport,
    ) -> None:
        stats = self.forgetting.run_forgetting_pass(self.db_path)
        report.entries_forgotten = stats.get("deleted", 0)
        report.entries_compressed = stats.get("compressed", 0)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _content_text(msg: dict) -> str:
        """Flatten a message's content (str or list-of-blocks) to plain text."""
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        return str(content or "")

    @staticmethod
    def _messages_to_text(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)
