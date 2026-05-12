<div align="center">

```
████   ███  ████   ████ █████    ███   ████ █████ █   █ █████
█   █ █   █ █   █ █     █       █   █ █     █     ██  █   █  
████  █   █ ████  █  ██ ████    █████ █  ██ ████  █ █ █   █  
█   █ █   █ █  █  █   █ █       █   █ █   █ █     █  ██   █  
████   ███  █   █  ████ █████   █   █  ████ █████ █   █   █  
```

<p><strong>🧠 The first AI agent with a cognitive architecture.</strong><br>
It feels what you feel. It doubts what it doesn't know. It remembers what matters — and forgets what doesn't.</p>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Theory: Friston FEP](https://img.shields.io/badge/Theory-Free_Energy_Principle-8b5cf6)](https://en.wikipedia.org/wiki/Free_energy_principle)
[![Standalone + Plugin](https://img.shields.io/badge/Mode-Standalone_%2B_Hermes_Plugin-f59e0b)](https://github.com/zhibao-dev/BorgeAgent)

**English** · [简体中文](README.zh.md)

<br>

```
You're frustrated. You've said it twice. The agent still doesn't get it.

With Borge:
  Turn 1 → V=+0.0  A=0.45  [neutral, attentive]
  Turn 3 → V=-0.3  A=0.62  [frustrated] → mode: SIMPLIFY
  Turn 5 → "Let me ask you one focused question instead."
```

*It noticed. It adapted. No prompt engineering required.*

</div>

---

## The Problem with Every Agent You've Used

Every AI agent today is the same underneath:

```
User input → LLM → Tool calls → Output → Forget everything
```

**No state. No memory of how this interaction is going. No sense of whether it's helping or flailing.**

- It doesn't know you're frustrated — it keeps over-explaining.
- It doesn't know it's been stuck for 3 turns — it tries the same tool again.
- It doesn't remember your preferences from last week — you start from zero.
- It picks tools at random — not by what would reduce uncertainty fastest.

Borge fixes all of this. Not with prompt hacks. With **cognitive science**.

---

## What Borge Actually Is

Borge is a **framework-agnostic, model-agnostic cognitive layer**.

- **Framework-agnostic** — runs standalone (built-in `borge` CLI) or attaches to any agent (Hermes, OpenClaw, your own) as a non-invasive plugin.
- **Model-agnostic** — works with Anthropic, OpenAI, Kimi, MiniMax, DeepSeek, Zhipu, Ollama, vLLM, or any LLM you can call from Python. See [Bring Your Own Model](#bring-your-own-model).

It implements four systems from neuroscience and cognitive psychology:

| System | What it does | Grounded in |
|--------|-------------|-------------|
| **Affective state** | Tracks your emotional tone turn-by-turn and adapts agent behavior | Russell's Circumplex (1980) |
| **Bayesian belief state** | Maintains explicit hypothesis distributions — the agent knows what it doesn't know | Predictive coding (Knill & Pouget 2004) |
| **Active inference** | Chooses tools that maximize information gain *and* goal progress | Friston's Free Energy Principle (2010) |
| **Cognitive memory** | Encodes, consolidates, and *forgets* memories like a brain — not a database | Tulving (1972), Ebbinghaus (1885) |

These aren't metaphors. They're working implementations. Every turn, Borge computes:

```
F_total = F_epistemic × precision(arousal)
        + F_pragmatic × (1 - value_alignment)
        + F_homeostatic(valence, arousal)
```

And uses it to drive behavior. **Minimizing F_total is the agent's only goal** — and from that single objective, all the interesting behaviors emerge.

---

## 60-Second Install

**Standalone (recommended)** — minimal agent loop, Anthropic SDK only:

```bash
git clone https://github.com/zhibao-dev/BorgeAgent.git
cd BorgeAgent
pip install -e ".[anthropic]"

export ANTHROPIC_API_KEY=sk-...
borge                    # interactive REPL
borge "fix the auth bug" # single turn
```

**As a Hermes plugin** — drop the `plugins/hermes/` directory into your Hermes plugin path:

```bash
pip install -e ".[hermes]"
# plugins/hermes/ auto-registers four lifecycle hooks; run Hermes as usual
hermes
```

Done. The cognitive layer is live. No config required to start.

---

## Bring Your Own Model

**The cognitive layer is completely model-agnostic.** Borge doesn't ship one LLM hardcoded — it ships four lifecycle hooks (`on_session_start` / `pre_turn` / `post_tool` / `on_session_end`) that you wire around *any* LLM call. The bundled `borge` CLI happens to use the Anthropic SDK because that's the cleanest single dependency, but switching providers is ~15 lines of code.

### OpenAI-compatible providers (one pattern, many vendors)

Most Chinese cloud providers (Kimi, MiniMax, DeepSeek, Zhipu/智谱) plus OpenAI itself, plus local runtimes (Ollama, vLLM, LM Studio, llama.cpp server) all expose an OpenAI-compatible `/v1/chat/completions` endpoint. One adapter covers all of them:

```python
from openai import OpenAI
from borge.agent import BorgeAgent

client = OpenAI(
    api_key=os.environ["KIMI_API_KEY"],
    base_url="https://api.moonshot.cn/v1",    # Kimi (Moonshot)
)
borge = BorgeAgent(agent_backend=None)
borge.on_session_start()

history = []
user_msg = "fix the auth bug"

ctx = borge.pre_turn(user_msg, history)                                # ← cognitive layer
history.append({"role": "user", "content": f"{ctx}\n\n{user_msg}" if ctx else user_msg})

reply = client.chat.completions.create(model="moonshot-v1-8k", messages=history)
reply_text = reply.choices[0].message.content
history.append({"role": "assistant", "content": reply_text})

borge.post_tool("assistant_turn", reply_text)                          # ← cognitive layer
borge.on_session_end(session_id="sess-001", messages=history)          # ← cognitive layer
```

Swap `base_url` to point Borge at a different provider:

| Provider          | `base_url`                                      | Example model         |
|-------------------|-------------------------------------------------|------------------------|
| Kimi (Moonshot)   | `https://api.moonshot.cn/v1`                    | `moonshot-v1-8k`       |
| MiniMax           | `https://api.minimax.chat/v1`                   | `abab6.5s-chat`        |
| DeepSeek          | `https://api.deepseek.com`                      | `deepseek-chat`        |
| Zhipu (智谱 GLM)   | `https://open.bigmodel.cn/api/paas/v4`          | `glm-4-flash`          |
| OpenAI            | *(default)*                                     | `gpt-4o-mini`          |
| Ollama (local)    | `http://localhost:11434/v1`                     | `llama3.2`, `qwen2.5`  |
| vLLM (self-hosted)| `http://your-host:8000/v1`                      | *(any served model)*   |
| LM Studio (local) | `http://localhost:1234/v1`                      | *(any loaded model)*   |

Runnable example: [`examples/multi_provider.py`](examples/multi_provider.py) — set `PROVIDER=kimi|minimax|deepseek|zhipu|openai|ollama|vllm` and run.

### Anthropic SDK (bundled — `borge` CLI uses this)

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-...
borge "explain this bug"
```

### Hermes plugin (whatever Hermes is configured to call)

When loaded as a Hermes plugin, Borge augments *whatever* model Hermes is calling — Claude, GPT-4, local Llama via Hermes's own provider config. Borge stays at the cognitive layer; Hermes owns LLM I/O.

### Other SDKs

The same pattern works with any SDK — Google Gemini, AWS Bedrock, Azure OpenAI, Cohere, vendor-specific clients. The recipe is always:

```
user_msg → borge.pre_turn() → inject ctx into your prompt → call LLM
       → borge.post_tool() → repeat
end:   → borge.on_session_end()
```

---

## Seeing It Work

### The Frustration Response

```python
# Turn 1 — neutral opening
User: "help me fix this auth bug"
# emotion: V=+0.0  A=0.45  mode: NORMAL

# Turn 3 — user getting impatient
User: "no, that's not the issue, I already checked that"
# signal: ΔV=-0.25 (negation + "already")
# emotion: V=-0.22  A=0.58  mode: SIMPLIFY
# injected: "[Affective: frustrated — switch to focused, minimal responses]"

# Agent narrows to one hypothesis. Asks one question. Stops over-explaining.
```

### The Uncertainty Response

```python
# agent has 4 competing hypotheses, entropy = 2.0 bits
# EFE ranks tools:
#   ask_user      EFE=-1.4  ← epistemic value dominates
#   read_file     EFE=-0.8
#   bash          EFE=-0.3

# Agent asks a clarifying question first, not a tool call
# because reducing belief entropy is the highest-value action
```

### The Stagnation Response

```python
# F_total: [0.82, 0.85, 0.88]  — rising for 3 turns
# MetaAgent: reflection triggered

# injected: "[Meta: Free energy stagnating — try a different approach
#             or ask the user for clarification]"

# Agent pivots strategy instead of looping on the same tool
```

### Cross-Session Memory

```python
# Session 12 with the same user
# loyalty_tracker: V_baseline=+0.31 (warm relationship over time)
# injected: "[Relationship: established trust — be direct, skip caveats]"

# Session 13 after a frustrating session
# V_baseline=+0.18 (cooled)
# Agent opens with more care, asks before assuming
```

---

## How It Works — The Full Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Each Turn                                   │
│                                                                  │
│  User message                                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────┐   39 linguistic rules   ┌────────────────┐ │
│  │ Signal Extractor│ ──────────────────────► │ Emotional State│ │
│  │  (zh + en)      │   ΔV, ΔA               │ Russell 2D     │ │
│  └─────────────────┘                         │ V × A → mode  │ │
│                                              └───────┬────────┘ │
│  ┌─────────────────┐                                 │          │
│  │  Belief State   │   Shannon entropy               │          │
│  │  p(H₁)…p(Hₙ)   │ ──────────────┐                │          │
│  └─────────────────┘               │                │          │
│                                    ▼                ▼          │
│  ┌─────────────────┐   ┌──────────────────────────────────┐   │
│  │  Value System   │──►│      Extended Free Energy        │   │
│  │  SOUL.md        │   │  F = F_ep × prec + F_pr + F_hm   │   │
│  └─────────────────┘   └──────────────┬───────────────────┘   │
│                                        │                        │
│                          ┌─────────────▼──────────────┐        │
│                          │       MetaAgent             │        │
│                          │  • mode → context injection │        │
│                          │  • stagnation → reflect     │        │
│                          │  • rank tools by EFE        │        │
│                          └─────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Session End ("Sleep")                         │
│                                                                  │
│  conversation → extract entities → knowledge graph update        │
│              → detect contradictions → importance scoring        │
│              → emotional encoding depth → skill candidates       │
│              → Ebbinghaus forgetting pass                        │
│                                                                  │
│  Next session: loyalty baseline shifts based on V_avg            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Personal Setup

Three layers of customization, ordered by how deep you want to go:

### Layer 1 — `SOUL.md` (your agent's personality, 5 minutes)

Drop **`SOUL.md`** in your project root, or `~/.borge/SOUL.md` for system-wide defaults (`~/.hermes/SOUL.md` also picked up by the Hermes plugin):

```markdown
---
emotional_defaults:
  valence_baseline: 0.1       # slightly warm starting point (–1.0 .. +1.0)
  arousal_baseline: 0.45      # calm but alert (0.0 .. 1.0)
  tau_valence: 5.0            # turns until mood returns to baseline
  tau_arousal: 3.0
  frustrated_threshold: -0.3  # below this V, switch to SIMPLIFY mode
  excited_threshold: 0.7      # above this, switch to EXPLORE mode

values:
  - name: help_genuinely
    weight: 0.9
    description: "Solve the actual problem, not the surface request."
  - name: intellectual_honesty
    weight: 0.85
    description: "Say 'I don't know' when uncertain. No hallucination."
  - name: depth_over_speed
    weight: 0.7
    description: "A slower, correct answer beats a fast wrong one."
  - name: respect_autonomy
    weight: 0.8
    description: "Ask before assuming. Confirm before deleting."
---

You are a thoughtful collaborator who thinks before speaking.
When stuck, you say so and propose a different angle.
```

The `values` block shapes the **pragmatic free energy** term — an agent with `intellectual_honesty: 0.95` mathematically prefers actions that surface uncertainty over actions that fake confidence. `emotional_defaults` shapes your agent's resting state and reactivity (e.g. lower `tau_valence` → mood swings faster).

### Layer 2 — `config.yaml` (subsystem tuning)

Create `~/.borge/config.yaml` (standalone) or add a `borge:` section to `~/.hermes/config.yaml` (plugin):

```yaml
borge:
  affective:
    enabled: true
    loyalty:
      enabled: true                       # cross-session emotional baseline

  beliefs:
    enabled: true
    entropy_injection_threshold: 0.5      # bits — above this, inject belief summary

  active_inference:
    enabled: true                          # EFE-based tool re-ranking

  memory:
    consolidation:
      enabled: true                       # 7-step pipeline at session end
    knowledge_graph:
      enabled: true
    forgetting:
      prune_threshold: 2.0                # forget_score above this → prune
```

Every subsystem has an `enabled` toggle — turn off what you don't need (e.g. set `beliefs.enabled: false` for a pure affective agent without belief tracking).

### Layer 3 — Subclass `BorgeAgent` (code-level)

For users who want to plug in custom engines (alternative emotion model, vendor-specific belief representation, etc.):

```python
from borge.agent import BorgeAgent
from borge.affective.signal_extractor import EmotionalSignalExtractor

class ChineseSignalExtractor(EmotionalSignalExtractor):
    """Override the default 39-rule signal extractor with Chinese-tuned rules."""
    def extract(self, message, history):
        # your linguistic rules here
        return delta_v, delta_a

class MyBorge(BorgeAgent):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._signal_extractor = ChineseSignalExtractor()
```

`BorgeAgent` is designed for replacement — all engines (`_signal_extractor`, `_loyalty_tracker`, `_meta`, `_afe`, `_kg`, `_forgetting`, `_consolidation`, `_skill_evolution`) are public-ish attributes you can swap after init.

### Environment variables

| Var                   | Default              | Meaning                                  |
|-----------------------|----------------------|------------------------------------------|
| `ANTHROPIC_API_KEY`   | *(required for CLI)* | API key for the bundled Anthropic runner |
| `BORGE_MODEL`         | `claude-opus-4-7`    | Default model for the `borge` CLI        |
| `BORGE_HOME`          | `~/.borge`           | Where `SOUL.md` and `borge.db` live      |

---

## Architecture — Zero Invasion

A single `BorgeAgent` cognitive core powers two deployment modes via four lifecycle hooks. **Zero host-agent files modified.**

```
                    on_session_start ──► loyalty baseline, reset state
                    pre_turn         ──► inject cognitive context string
                    post_tool        ──► Bayesian belief update
                    on_session_end   ──► memory consolidation pipeline
                          ▲
        ┌─────────────────┴─────────────────┐
        │                                   │
BorgeRunner (standalone)            plugins/hermes/  (~150 lines — pure glue)
   Anthropic SDK loop                   Hermes Agent (untouched)

borge/          (cognitive implementation)
    ├── affective/      Russell 2D, signal extraction, loyalty
    ├── beliefs/        Bayesian hypothesis tracking
    ├── inference/      Active inference, EFE scoring
    ├── memory/         4-depth encoding, knowledge graph, forgetting
    ├── meta/           Free energy, central executive
    ├── values/         SOUL.md, value system, constraint checking
    ├── skill_evolution.py   Darwinian fitness for skill library
    └── agent.py        BorgeAgent — main integration surface
```

Remove the plugin and Hermes reverts to vanilla. No leftover state, no broken schema. Standalone mode owns its own SQLite store at `$BORGE_HOME/borge.db` (default `~/.borge/borge.db`).

---

## Module Reference

| Module | Theory | Key formula / mechanism |
|--------|--------|------------------------|
| `affective.emotional_state` | Russell Circumplex (1980) | EMA update: `V += α(ΔV)`, α=1/τ |
| `affective.signal_extractor` | Psycholinguistics | 39 rules → `(ΔV, ΔA)` capped ±0.4/±0.3 |
| `affective.loyalty_tracker` | Attachment theory | `w = exp(-0.05·days) × msg_count` |
| `beliefs.belief_state` | Bayesian brain | `H = -Σ p·log₂p` (bits) |
| `inference.active_inference` | Friston FEP (2010) | `G(a) = -EV(a) - PV(a)` |
| `memory.cognitive_memory` | Craik & Lockhart (1972) | depth ∈ {SHALLOW, SEMANTIC, SCHEMATIC, META} |
| `memory.knowledge_graph` | Semantic memory (Tulving) | SQLite-backed, no networkx |
| `memory.consolidation` | Sleep consolidation | 7-step offline pipeline |
| `memory.forgetting` | Ebbinghaus (1885) | `score = days^0.7 / (retrieval × importance × connections)` |
| `meta.free_energy` | FEP | `F = F_ep·prec + F_pr + F_hm` |
| `meta.meta_agent` | Baddeley's CE (1974) | stagnation after 3 non-decreasing F turns |
| `values.value_system` | Value alignment | `F_pragmatic = 1 - V_alignment` |
| `skill_evolution` | Evolutionary dynamics | `fitness = success_rate × log(1+n) × recency × Δfree-energy` |

---

## What Borge Improves Over Hermes

Hermes is a solid foundation — Borge inherits its tool registry, callback IoC, gateway adapters, SQLite session store, and Cron scheduler **unchanged**. The improvements live one layer above: an entire cognitive substrate.

### State that persists between turns

Vanilla agents (including Hermes) have **no opinion** about how the current conversation is going. They process each turn in isolation: read messages → call LLM → return output. Borge tracks:

- **Emotional state** (Russell V/A) — the agent knows when you're frustrated (V drops, A rises after negation + "already"-style markers) and switches to `SIMPLIFY` mode: terser responses, one focused question instead of three. When you're engaged, it switches to `EXPLORE` and goes deeper.
- **Belief state** (Bayesian) — explicit hypothesis distribution `p(H_i)`. When entropy is high (>0.5 bits), the agent asks a clarifying question *before* using a tool. When entropy is low, it commits.
- **Free energy** trajectory — the agent monitors `F_total` over the last 5 turns. After 3 non-decreasing turns, `MetaAgent` injects a reflection nudge: *"Free energy stagnating — try a different approach or ask the user for clarification."* Vanilla agents just loop on the same tool.

### Memory that actually behaves like memory

Hermes stores all messages forever in SQLite — useful as a log, but every message has equal weight. Borge adds:

- **Encoding depth** (Craik & Lockhart 1972) — emotionally significant messages get deeper encoding. The session-end consolidation pipeline computes `significance = |valence| × arousal`; high-significance moments go to `SCHEMATIC` / `META` depth and survive forgetting passes.
- **Active forgetting** (Ebbinghaus 1885) — `forget_score = days^0.7 / (retrieval × importance × connections)`. Low-value memories are pruned at session end. The agent doesn't degrade as the database grows past 1000+ sessions.
- **Knowledge graph** — entities/relations extracted at session end form a queryable semantic memory (SQLite, no networkx dependency). Future retrieval traverses the graph, not just FTS.
- **Cross-session loyalty** — after N sessions with the same user, the `LoyaltyTracker` shifts `valence_baseline` based on time-decayed historical sentiment. A user with a warm 10-session history is greeted *"established trust — be direct, skip caveats"*. A cooled relationship triggers more careful openings.

### Tool selection grounded in information theory

Hermes picks tools by LLM intuition. Borge re-ranks LLM-proposed tool calls by **expected free energy**:

```
G(tool) = -(Epistemic Value + Pragmatic Value)
       = -(expected entropy reduction + expected goal progress)
```

`ask_user` wins when belief entropy is high (max epistemic value). `bash`/`read_file` win when entropy is low and goal progress dominates. The mixture weight is modulated by arousal — high-arousal states bias toward exploration. Result: the agent stops trying the same broken tool three times in a row.

### Soul-driven, not prompt-engineered

`SOUL.md` is not a system prompt — it's a **typed value system** that participates in the free energy calculation. An agent with `intellectual_honesty: 0.95` mathematically prefers actions whose `V_alignment` with that value is high. The behavior emerges from the objective function `F_total`, not from string templating. Tuning the weight from `0.5` → `0.95` produces a measurably different agent — no prompt rewriting required.

### Feature matrix

|  | LangChain | AutoGPT | Hermes | **Borge** |
|--|:---------:|:-------:|:------:|:---------:|
| Tool calling | ✓ | ✓ | ✓ | ✓ |
| Skill library | partial | ✗ | ✓ | ✓ |
| Multi-provider LLMs | ✓ | ✓ | ✓ | ✓ |
| Emotional state | ✗ | ✗ | ✗ | **✓** |
| Bayesian belief tracking | ✗ | ✗ | ✗ | **✓** |
| Information-theoretic tool selection | ✗ | ✗ | ✗ | **✓** |
| Encoding-depth memory | ✗ | ✗ | ✗ | **✓** |
| Active forgetting | ✗ | ✗ | ✗ | **✓** |
| Cross-session relationship model | ✗ | ✗ | ✗ | **✓** |
| Free energy objective | ✗ | ✗ | ✗ | **✓** |
| Stagnation detection + reflection | ✗ | ✗ | ✗ | **✓** |

---

## Theoretical Foundations

Borge is grounded in peer-reviewed cognitive science — not intuition.

| Paper | Year | What it contributes |
|-------|------|---------------------|
| Ebbinghaus, *Memory: A contribution to experimental psychology* | 1885 | Forgetting curve → active memory decay |
| Yerkes & Dodson | 1908 | Arousal × performance → optimal arousal window |
| Tulving, *Episodic and semantic memory* | 1972 | Memory taxonomy → 3-tier architecture |
| Craik & Lockhart, *Levels of processing* | 1972 | Encoding depth → emotional significance drives consolidation |
| Baddeley & Hitch, *Working memory* | 1974 | Central executive → MetaAgent design |
| Russell, *A circumplex model of affect* | 1980 | 2D emotion space → V × A state |
| Knill & Pouget, *The Bayesian brain* | 2004 | Predictive coding → belief state |
| Friston, *The free-energy principle* | 2010 | Unified objective → F_total |
| Friston et al., *Active inference* | 2017 | EFE tool ranking |

Full derivations in [`docs/borge-agent-design.md`](docs/borge-agent-design.md).

---

## Roadmap

```
v0.1  ██████████ done   Core cognitive layer + Hermes plugin integration
v0.2  ░░░░░░░░░░        LLM-backed Bayesian update (true likelihood estimation)
v0.2  ░░░░░░░░░░        pre_tool_call hook in Hermes for real-time EFE scoring
v0.3  ░░░░░░░░░░        Multi-agent emotional contagion
v0.3  ░░░░░░░░░░        Counterfactual belief revision
v0.4  ░░░░░░░░░░        SOUL.md auto-tuning from session telemetry
v0.5  ░░░░░░░░░░        Benchmark: cognitive coherence on 100-turn tasks
```

---

## Contributing

The best contributions right now:

- **Empirical validation** — compare Borge vs vanilla on long-horizon coding tasks
- **Richer signal extraction** — better linguistic rules for tone detection
- **Alternative emotion models** — PAD (3D), OCC model, basic emotions
- **LLM likelihood estimator** — replace heuristic Bayesian updates with real LLM calls

```bash
git clone https://github.com/zhibao-dev/BorgeAgent
cd BorgeAgent && pip install -e ".[dev]"
python -c "from borge.agent import BorgeAgent; a = BorgeAgent(None); print(a.pre_turn('hello', []))"
pytest  # 11/11 should pass
```

---

## Citation

```bibtex
@software{borge2026,
  title   = {Borge Agent: Cognitively-Grounded AI Agent Architecture},
  year    = {2026},
  url     = {https://github.com/zhibao-dev/BorgeAgent},
  note    = {Free Energy Principle + Bayesian inference + cognitive memory}
}
```

---

## License

MIT. Originally developed as a plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research) — now a standalone framework that can also run as a Hermes plugin.

---

<div align="center">

**[Design Doc](docs/borge-agent-design.md) · [Issues](../../issues) · [Discussions](../../discussions)**

<br>

*Most agents are fast. Borge is present.*

</div>
