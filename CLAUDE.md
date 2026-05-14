# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project at a glance

Borge Agent is a **framework-agnostic cognitive layer** (affective state, Bayesian beliefs, active inference, cognitive memory, free-energy monitoring) for LLM agents. The same `BorgeAgent` cognitive core powers two deployment modes:

- **Standalone** — `BorgeRunner` drives a direct Anthropic SDK loop with three built-in tools (`bash`, `read_file`, `write_file`). Entry point: the `borge` console script.
- **Hermes plugin** — `plugins/hermes/__init__.py` wraps `BorgeAgent` and registers four lifecycle hooks on a host Hermes agent without modifying any core files.

The cognitive theory background (Russell circumplex, Friston FEP, Tulving, Ebbinghaus, etc.) lives in `README.md` and `docs/borge-agent-design.md` — this file focuses on the code surface.

## Common commands

| Task | Command |
|------|---------|
| Install dev (Anthropic SDK + pytest) | `pip install -e ".[dev]"` |
| Install standalone runtime only | `pip install -e ".[anthropic]"` |
| Install Hermes plugin runtime only | `pip install -e ".[hermes]"` |
| Run the full test suite | `pytest` |
| Run a single test | `pytest tests/plugins/test_borge_plugin.py::test_borge_agent_init` |
| Interactive REPL | `borge` |
| Single-turn execution | `borge "fix the auth bug"` |
| Custom model / Soul / DB | `borge -m claude-opus-4-7 -s ./SOUL.md --db /tmp/borge.db` |
| Smoke test (no API key needed) | `python -c "from borge.agent import BorgeAgent; a = BorgeAgent(None); print(a.pre_turn('hello', []))"` |

The REPL/Runner require `ANTHROPIC_API_KEY` in the environment (or `api_key=` passed to `BorgeRunner`). The model defaults to `claude-opus-4-7` (`BORGE_MODEL` env var overrides it).

`pyproject.toml` pins `testpaths = ["tests"]`, so `pytest` from anywhere in the repo picks up the suite.

## Big-picture architecture

Both deployment modes share **one** cognitive core. All state lives in `BorgeAgent`; the runner and plugin are thin wrappers.

```
borge/agent.py::BorgeAgent       ← single cognitive core (all state lives here)
        │
        ├── borge/runner.py::BorgeRunner   ← standalone agent loop (Anthropic SDK)
        │       └── borge/cli.py            ← console_script: `borge`
        │
        └── plugins/hermes/__init__.py     ← Hermes plugin wrapper (4 lifecycle hooks)
```

`BorgeAgent` exposes **four lifecycle hooks** (`borge/agent.py:111`–`216`) that both deployment modes call:

1. `on_session_start(user_id)` — fetches past sessions for `user_id`, computes a loyalty baseline, seeds `EmotionalState.valence_baseline / arousal_baseline`, resets per-session counters.
2. `pre_turn(user_message, history) -> str` — extracts (ΔV, ΔA) from the user message, updates `EmotionalState`, ticks `MetaAgent`, and returns a **context-injection string** (affective summary + belief summary + reflection nudge if F_total is stagnating). Empty string = nothing to inject.
3. `post_tool(tool_name, result, llm_caller=None)` — Bayesian belief update over the hypothesis distribution (if any), value-satisfaction update.
4. `on_session_end(session_id, messages)` — runs the 7-step consolidation pipeline: entity extraction → knowledge-graph fusion → contradiction detection → importance rescoring → emotional encoding depth → skill candidate detection → Ebbinghaus forgetting pass.

The runner glues `pre_turn`'s string in front of the user message; the Hermes plugin returns it from the `pre_llm_call` hook.

### Cognitive subsystems

Each directory is mostly self-contained; `BorgeAgent` is the only place where they're wired together.

| Directory | Key classes / responsibility |
|-----------|------------------------------|
| `borge/affective/` | `EmotionalState` (Russell V/A, EMA update with separate `tau_valence` / `tau_arousal`), `EmotionalSignalExtractor` (39 linguistic + structural rules → ΔV, ΔA), `LoyaltyTracker` (cross-session baseline) |
| `borge/beliefs/` | `BeliefState` (explicit hypothesis distribution, Shannon entropy, optional LLM-driven likelihood update), `Hypothesis` |
| `borge/inference/` | `ActiveInferenceEngine` — re-ranks candidate tool calls by expected free energy `G = -(epistemic + pragmatic)` |
| `borge/memory/` | `CognitiveMemory` (4-level encoding depth), `KnowledgeGraph` (**pure SQLite**, no networkx), `MemoryConsolidationPipeline`, `ForgettingEngine` (emotion-aware Ebbinghaus score), `MemoryStore` (owns `borge_memories` table), `MemoryRetrieval` (mood-congruent + ΔF ranking) |
| `borge/meta/` | `ExtendedFreeEnergy` (`F = F_ep × precision + F_pr × V_alignment + F_hm`), `MetaAgent` (Baddeley central executive — monitors F-trajectory, triggers reflection after 3 non-decreasing turns, builds the context-injection string) |
| `borge/values/` | `parse_soul_frontmatter` (YAML frontmatter parser), `ValueSystem` (typed prior preferences + hard constraints derived from SOUL.md) |

## Configuration & persistence

- **Personality** — `SOUL.md` with YAML frontmatter for `emotional_defaults` and `values`. Resolution order (`borge/agent.py:249-257`): `$CWD/SOUL.md` → `$BORGE_HOME/SOUL.md` (default `~/.borge/SOUL.md`).
- **Runtime config** — `$BORGE_HOME/config.yaml` (standalone) or `$HERMES_HOME/config.yaml` (plugin). Both keyed under a top-level `borge:` node. Every subsystem has an `enabled` flag, so you can disable e.g. `borge.beliefs.enabled` without changing call sites. Helper: `BorgeAgent._cfg("beliefs.entropy_injection_threshold", 0.5)`.
- **State store** — SQLite at `$BORGE_HOME/borge.db` (default `~/.borge/borge.db`). Holds session summaries, knowledge-graph nodes/edges, and the skill fitness table. `BorgeAgent.__init__` creates `BORGE_HOME` on demand.

## Conventions for modifying the cognitive layer

- **Funnel all cognitive state through `BorgeAgent`.** Both `runner.py` and `plugins/hermes/__init__.py` hold one `BorgeAgent` instance and call only the four hooks. New cognitive capabilities should be added as engines composed inside `BorgeAgent`, then exposed through an existing hook (or a new thin wrapper method). Don't leak `EmotionalState` / `BeliefState` etc. into the runner or plugin.
- **Plugin hooks must swallow exceptions.** Every callback in `plugins/hermes/__init__.py` wraps its body in `try/except` and degrades to an empty string / no-op. This is the implementation half of the README's "Zero Invasion" promise — a cognitive-layer bug must never crash the host agent.
- **`BorgeAgent(agent_backend=None)` is valid.** The constructor parameter was renamed from `hermes_agent` to `agent_backend` (commit `f2e51e0`). Pass `None` when there's no external backend to call back into.
- **`pre_turn` returns a context-injection string, not a side effect.** `BorgeRunner._turn` prepends it to the user message; the Hermes plugin returns it from `pre_llm_call`. Empty string means "inject nothing" — preserve this contract.
- **Aux LLM is optional.** `post_tool(tool_name, result, llm_caller=...)` and `score_tool_candidates(candidates, llm_caller=...)` accept an optional callable for likelihood / scoring calls. Code must work when it's `None` (falls back to deterministic heuristics).
- **Emotion ↔ free-energy ↔ memory loop is wired.** `BorgeAgent` tracks `_emotional_history` and `_session_f_history` per turn; `on_session_end` feeds both into `MemoryConsolidationPipeline.run(..., f_history=...)`. Step 5 persists each message to `borge_memories` with `(V, A, significance, depth, f_total_at_encoding, delta_f_total)`. `ForgettingEngine._compute_score` multiplies in `emotion_resistance = 1/(1 + α·|V|·A)`, and `apply_importance_from_delta_f` boosts `importance_score` for progress-bearing turns. `BorgeAgent.recall(query)` exposes mood-congruent retrieval through `MemoryRetrieval`. **When adding new memory subsystems, preserve this end-to-end loop** — emotion shapes encoding depth, F shapes importance, both shape forgetting, mood + F shape retrieval.

## Plugin naming note

The plugin's registered name in Hermes (`name: borge` in `plugins/hermes/plugin.yaml`) is independent of its Python module path (`plugins.hermes`). Don't conflate them — Hermes loads the plugin by the registered name; tests and Python imports go through `plugins.hermes`.

## Test-path setup

`pyproject.toml` configures `pythonpath = ["."]` so `pytest` can find the repo-root `plugins/` namespace package. `tests/plugins/` deliberately has **no `__init__.py`** — adding one would shadow the real `plugins/` package and break `from plugins.hermes import ...` in tests.
