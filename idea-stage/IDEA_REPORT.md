# Idea Discovery Report

**Direction**: 研究"情绪"和"自我"对记忆的提取和遗忘的影响，可量化建模；其中"自我"是否可以用自由能理论（FEP / Active Inference）建模

**Date**: 2026-05-14
**Pipeline**: research-lit (4 sources) → idea-creator (10 ideas) → novelty-check (web) → research-review (in-session reviewer)
**Constraints**: ARXIV_DOWNLOAD=false, AUTO_PROCEED=true, no Codex MCP, no GPU pilots

---

## Executive Summary

文献交叉点的**白空间是明确的**：

| 方向 | 有谁做了 | 缺什么 |
|------|---------|--------|
| 情绪 × 记忆 量化模型 | **CMR3** (Cohen & Kahana 2022) —— retrieved-context 加 multivalent emotion | 没有自我维度，不基于 FEP |
| 自我 × FEP | **Minimal Self Models** (Limanowski 2013) · **Free-Energy Self** (Apps & Tsakiris 2014) | 偏哲学/身体识别，**没有连到记忆动力学** |
| LLM agent 记忆 | **LUFY**（arousal+surprise）· **ACT-R 启发的 LLM memory**（2025）· **Nemori / ComoRAG / MemOS**（2025） | 没有 self 模型，没有 FEP 形式化 |
| Self-Reference Effect | **Symons & Johnson 1997 meta-analysis** · 2025 现代复制 | 仅经验研究，缺少 mechanistic computational model |

**结论**：把"自我"形式化为 FEP 框架下的 **precision-weighted generative prior**，让 self-precision 乘性调制 memory 的 encoding-depth / forget-score / retrieval-ranking —— **这是个无人占领的交叉点**，且 BorgeAgent 已经实现了 5 个所需组件中的 3 个。

**Recommended idea (auto-selected, AUTO_PROCEED=true)**：**Idea 1 — Self-FEP Memory**。

---

## Literature Landscape (Phase 1)

### 紧邻竞品（最值得对位）

1. **CMR3 — A retrieved-context model of emotional modulation of memory** (Cohen & Kahana, *Psychological Review* 2022) — [PDF](https://memory.psych.upenn.edu/files/pubs/CoheKaha22.pdf)
   - 在 CMR2 基础上加 multivalent emotion（neg/pos/mixed 两 cell 编码）
   - 解释 mood-congruent recall、intrusive memories、emotion dysregulation
   - **没有 self 维度，没有 FEP**。这是要超越的标杆。

2. **A memory-based theory of emotional disorders** (Cohen & Kahana, *PMC* 2022) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9256582/)
   - 上文的临床扩展；预测抑郁/PTSD 的 chronic mood persistence

### FEP-Self（理论侧）

3. **Minimal self-models and the free energy principle** (Limanowski & Blankenburg 2013) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3770917/)
   - MPS（minimal phenomenal selfhood）映射到 hierarchical generative model
   - 概念性，无计算实现

4. **The free-energy self: A predictive coding account of self-recognition** (Apps & Tsakiris 2014) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3848896/)
   - 身体识别、自我感的预测编码解释
   - 与 memory 无直接联系

### LLM agent memory（2025 一线）

5. **MemOS — Memory OS for AI Systems** (Memtensor 2025) — [PDF](https://statics.memtensor.com.cn/files/MemOS_0707.pdf)
6. **Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture** (HAI 2025)
7. **Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers** (arXiv 2603.07670)
8. **LUFY** — 用 emotional arousal + surprise 决定记忆保留
9. **Nemori / ComoRAG** (2025) — cognitive-inspired memory organization

**共性观察**：这一组都用 ACT-R 或 retrieval-augmented 思路；**没人把 FEP 拿来形式化 self-prior 与 memory 的耦合**。

### 经验侧

10. **The Self-Reference Effect in Memory: A Meta-Analysis** (Symons & Johnson 1997) — d′ effect = 0.50, 已被 2024/2025 老年人组、跨文化、源监控等多次复制
11. **Engram Memory Encoding and Retrieval: A Neurocomputational Perspective** (arXiv 2506.01659, 2025) — sparsity-capacity tradeoff，consolidation refinement，但同样没 self / 没 FEP

---

## Phase 2 — Ranked Ideas (10 generated → 8 surviving → top 3 detailed)

### 🏆 Idea 1: Self-FEP Memory — RECOMMENDED

**One-liner**: A precision-weighted self-model (FEP) drives memory encoding depth, forgetting rate, and retrieval ranking — replicating the Self-Reference Effect with a mechanistic computational model that interacts orthogonally with emotion.

**Hypothesis**:

H1. Self can be formalized as a generative model `M_self = (μ_self, π_self)` with prior mean and precision in the FEP framework.

H2. Memory self-relevance `sr_i = α + β · cos(e_i, μ_self)` where `e_i` is memory embedding.

H3. Encoding depth ≈ `|V|·A · (1 + λ_e · π_self · sr_i)` (multiplicative interaction with emotion).

H4. Forget score augmented: `score ← score × 1/(1 + λ_f · π_self · sr_i)`.

H5. Retrieval ranking augmented: `+ w_s · self_similarity(current_μ_self, memory.μ_self_at_encoding)`.

**Quantitative predictions**:

- P1 (replication): SRE meta-analysis d′ ≈ 0.50 falls inside the model's predicted band under default parameters.
- P2 (interaction): Mood × Self should be **multiplicative**, not additive — fit nested models to Bower (1981) + Rogers (1977) data.
- P3 (precision ablation): As `π_self → 0`, SRE attenuates monotonically toward 0 (matches dissociation / sleep deprivation literature).
- P4 (self-evidencing): F_total drops more on self-relevant turns than other-relevant turns (testable in BorgeAgent simulations).
- P5 (vs CMR3): On a self-referenced subset of stimuli, Self-FEP Memory outperforms CMR3 in fit; on emotion-only stimuli, they're comparable.

**Experiments**:

| # | Block | What | Compute |
|---|-------|------|---------|
| E1 | SRE replication | 4-condition (self/other/semantic/structural) × N=200 simulated agent encoding trials; recall after delay; compare d′ to Symons & Johnson | CPU only |
| E2 | Mood × Self factorial | 2×2 × 100 items per cell, AIC compare {mood-only, self-only, additive, multiplicative} | CPU only |
| E3 | π_self ablation | π_self ∈ {0.1, 0.3, 0.5, 0.7, 1.0}; measure SRE size | CPU only |
| E4 | F_total trajectory | Track ΔF on self vs other items; expect self-evidence accumulation | CPU only |
| E5 | CMR3 comparison | Fit both models to held-out human emotional memory data | Modest GPU for embeddings |

**Compatibility with BorgeAgent**:

| Needed component | BorgeAgent has it? | Add cost |
|-----------------|-------------------|---------|
| `emotional_valence` / `arousal` | ✅ in `borge_memories` | 0 |
| `encoding_depth` field | ✅ | 0 |
| `forget_score` formula with multipliers | ✅ | 0 |
| Retrieval ranking with mood term | ✅ in `MemoryRetrieval` | 0 |
| `self_relevance_score` field | ❌ | new column + Step 5b in consolidation |
| `SelfModel(μ_self, π_self)` class | ❌ | new `borge/values/self_model.py` |
| Embedding for memories | ❌ | sentence-transformers, ~1 day work |

Implementation cost ~3-5 engineering days; **0 architectural changes** to existing engines.

**Novelty (closest prior art and differentiation)**:

| Prior work | Closest claim | Differentiation |
|-----------|---------------|-----------------|
| CMR3 (Cohen & Kahana 2022) | emotion-modulated retrieval | We add self dimension; predict multiplicative emotion×self interaction |
| Free-Energy Self (Apps & Tsakiris 2014) | self under FEP for body | We connect FEP-self to **memory dynamics**, not body recognition |
| Minimal Self Models (Limanowski 2013) | MPS as generative model | We give it computational instantiation tied to memory forget/recall |
| LUFY | arousal+surprise for retention | We add explicit self prior, not just arousal |
| ACT-R LLM memory (2025) | activation decay + spreading | We use FEP precision, predict different decay shapes |

**Risk**:
- Med — need embedding-based self prior, which adds a dependency
- Med — π_self semantics need careful definition (not just "knob") — link to either dynamic precision update from prediction error or learned via Bayesian model comparison
- Low — testing against SRE meta-analysis is well-precedented

**Reviewer score (self-assessed, GPT-5.4 not available)**: **7/10**.
- Strong novelty
- Clean experimental design with public benchmarks
- BorgeAgent compatibility makes it implementable
- Weakest point: π_self needs principled derivation, not just a fitted parameter

**Next step**: Stage 2 implementation on BorgeAgent.

---

### 🥈 Idea 3: FEP-Reconsolidation — Strong Backup

**One-liner**: Memory becomes labile on retrieval and re-stabilizes via prediction-error-driven updates; learning rate scaled by self-precision (Goldstein Ch.7 reconsolidation mechanism formalized under FEP).

**Differentiation**: No prior FEP formalization of reconsolidation. Closest is Nader & Hardt 2011 review (no math).

**Compatibility**: Needs new `re_encode_on_recall()` hook in `MemoryRetrieval` + the **source monitoring schema** I previously flagged. Higher engineering cost than Idea 1.

**Why backup not primary**: The "self-precision × reconsolidation rate" claim is harder to test against existing data — most reconsolidation studies are animal models without computational equivalents. Better single-paper scope = Idea 1.

---

### 🥉 Idea 4: Mood × Self Factorial — Safe Fallback

**One-liner**: A pre-registered factorial study comparing additive vs multiplicative models of mood and self on memory bias, using BorgeAgent as a controllable testbed.

**Differentiation**: Empirically motivated, low theoretical surface. Easier to publish (Cognitive Science / Computational Brain & Behavior), but less novel.

**Why fallback not primary**: This is Idea 1's experiment E2 standing alone. If Idea 1's full theoretical story is too ambitious, this is the safe extraction.

---

## Phase 2 — Eliminated Ideas (5 of 10)

| Idea | Reason eliminated |
|------|-------------------|
| #2 Active inference + self-schema tool selection | BorgeAgent's `active_inference` is currently experimental & unwired. Reviving it before testing the self-memory hypothesis adds confounded engineering dependency. |
| #5 Markov-blanket self for memory boundary | Markov blanket is operationally squishy in single-agent LLM context. Without clear "outside the blanket" sensory data, can't define a testable boundary. |
| #7 Self-vs-other source monitoring via FEP | Strong novelty, but needs source-monitoring schema (which IS on the BorgeAgent roadmap). Better done after Idea 1 establishes self_relevance machinery. Defer to v2. |
| #8 Dual-system retrieval (Kahneman) | Too narrow on its own — better as a sub-experiment within Idea 1 (high-arousal → faster, less self-modulated retrieval). |
| #10 Generative replay during consolidation | Replay buffers are mature in RL; FEP-version exists in animat literature. Less novel as a standalone paper. Same fate — sub-experiment of Idea 1's session-end pass. |

---

## Phase 3 — Novelty Verification

Targeted searches over arXiv / Scholar / Semantic Scholar (via WebSearch, May 2026) returned:

| Query | Top hit | Verdict |
|-------|---------|---------|
| `"self-reference effect" computational model 2024 2025` | Symons & Johnson 1997 meta still cited as gold standard; 2025 papers add aging / older adults / source-recollection but **no computational model** | Confirmed novel |
| `"free energy" "self" memory retrieval` | Apps & Tsakiris 2014 + Limanowski 2013 (philosophical); no recent computational memory tie | Confirmed novel |
| `"CMR3" OR "retrieved-context" self-reference` | CMR3 explicitly emotion-only; no self extension | Confirmed novel |
| `LLM agent self model memory 2025` | Cognitive Architectures for Language Agents (Sumers 2024) mentions self abstractly; no formalization | Confirmed novel |

**Concurrent risk** (last 3 months): No direct preprint found. Closest near-misses are (a) MemOS family from China, which focuses on storage/retrieval ops not theoretical self, and (b) Engram Memory Encoding (arXiv 2506.01659) — neurocomputational but no self/FEP.

---

## Phase 4 — Critical Review (in-session, best-effort)

**Setup**: GPT-5.4 / Codex MCP not available in this session. Acting as my own reviewer using internalized NeurIPS/ICLR norms. **Treat as paper-level critique, not external validation**.

### Strengths
1. ✅ **Clear theoretical contribution** — formalizing self via FEP precision and tying it mechanistically to memory dynamics is a genuine framework move, not just a benchmark gain.
2. ✅ **Quantitative predictions** — 5 separate predictions, each falsifiable against existing human data benchmarks (Symons & Johnson meta-analysis is gold standard).
3. ✅ **Implementation path is concrete** — BorgeAgent gives us 60% of the substrate; the remaining engineering is well-scoped.
4. ✅ **Multiplicative vs additive prediction** is a sharp test that distinguishes the model from Bower-style associative networks.

### Weaknesses
1. ⚠️ **π_self semantics need a principled grounding.** "Self-precision" can't just be a free parameter. Reviewers will demand: how does π_self change over time? From prediction error? From explicit value alignment? Must commit before submission.
2. ⚠️ **Embedding choice may dominate self-similarity** — using sentence-transformers means our "self prior" is partly the embedding model's artifact. Need an ablation across 2+ embedding models.
3. ⚠️ **CMR3 comparison is hard** — it's not packaged as a library. Need to reimplement key components for E5. Budget time.
4. ⚠️ **The "BorgeAgent fits perfectly" framing** can read as engineering-led research. Strengthen the cognitive science framing: emphasize the SRE replication + Mood×Self interaction prediction as primary, system as instrument.
5. ⚠️ **Where does this fit?** Theoretical contribution best at NeurIPS / ICLR. Empirical SRE story best at CogSci / Computational Brain & Behavior. Need to pick one and frame accordingly.

### Reviewer recommendations
- **Pre-register** the multiplicative-vs-additive comparison BEFORE running.
- **Add a human-data fit baseline** alongside CMR3 — Symons & Johnson 1997 effect sizes are public.
- **Derive π_self update rule from FEP first principles**, don't just fit it. Most defensible: `π_self ∝ 1 / Var[prediction error on self-relevant items]` (high-precision = low PE variance).
- **Lead with SRE replication + mood × self interaction**, save π_self ablation for the technical contribution section.

### Score
- Self-assessed: **7/10**
- Path to 8+: tighten π_self derivation, add human-data benchmark fit.

---

## Refined Proposal Anchor

**Problem Anchor**: Existing computational memory models (CMR3, ACT-R, retrieval-augmented LLMs) modulate memory by emotion but lack a **principled self model that interacts with emotion mechanistically**. Cognitive science has documented the Self-Reference Effect for 40+ years without a unified computational account that meshes with the Free Energy Principle.

**Method Thesis**: Treat the self as a generative model `M_self = (μ_self, π_self)` and let its precision multiplicatively gate memory encoding depth, forget rate, and retrieval ranking — generating quantitative predictions for SRE that fit existing meta-analytic data and predict Mood × Self interaction as multiplicative not additive.

**Dominant Contribution**: First FEP-grounded computational model of self that drives memory dynamics, instantiated as a thin extension over BorgeAgent.

**Must-run experiments**: E1 (SRE replication), E2 (Mood × Self factorial — primary test), E3 (π_self ablation), E5 (vs CMR3). E4 (F-trajectory) is icing.

---

## Ranked Ideas Summary

| # | Idea | Pilot signal* | Novelty | Self-review | Status |
|---|------|---------------|---------|-------------|--------|
| 1 | **Self-FEP Memory** | conceptual + | confirmed | 7/10 | 🏆 **RECOMMENDED** |
| 3 | FEP-Reconsolidation | conceptual + | confirmed | 6/10 | 🥈 Backup |
| 4 | Mood × Self factorial | conceptual + | partial (familiar design) | 6/10 | 🥉 Safe fallback |
| 6 | FEP Forgetting Schedule | weak conceptual | partial | 5/10 | Defer |
| 9 | Bayesian Loyalty as Self-Identity | conceptual + | partial | 5/10 | Defer (sub-experiment of #1) |
| 2 | Active Inference + Self-Schema | weak (active_inference unwired) | weak (extends Friston) | 4/10 | Eliminated |
| 5 | Markov-blanket self | conceptual but operationally unclear | strong but untestable | 4/10 | Eliminated |
| 7 | FEP source monitoring | strong conceptual | strong | 6/10 | Defer to v2 of #1 |
| 8 | Dual-system retrieval | conceptual + | weak | 4/10 | Subsume into #1 |
| 10 | Generative replay | weak | weak | 4/10 | Eliminated |

*Pilot signal is conceptual only — no GPU pilots run in this session.

---

## Next Steps (after Gate 1 confirmation)

If user confirms Idea 1:

- [ ] Stage 2 — Implementation
  - [ ] Add `borge/values/self_model.py` with `SelfModel(μ_self, π_self)` class
  - [ ] Extend `borge_memories` schema with `self_relevance_score`
  - [ ] Add embedding step in consolidation (sentence-transformers, lazy import)
  - [ ] Modify `forget_score` to include `× 1/(1 + λ_f · π_self · sr)`
  - [ ] Modify `MemoryRetrieval.recall` to add `w_s · self_similarity`
  - [ ] **Derive π_self update rule from prediction-error variance** (reviewer #1 fix)
- [ ] Stage 3 — Run E1 + E2 + E3 (all CPU; ~hours)
- [ ] Stage 4 — `/auto-review-loop` (deferred without Codex MCP; manual review pass instead)
- [ ] Stage 5 — Write NARRATIVE_REPORT.md

If user wants Idea 3 or 4 instead, branch from here.

---

**Pipeline limitations declared**:
- No Codex MCP → Phase 4 reviewer was in-session, not external GPT-5.4
- No GPU compute → Phase 2 pilots are conceptual not empirical
- arXiv metadata only (ARXIV_DOWNLOAD=false honored)
- 2-3 narrow searches per topic, not exhaustive — concurrent-work risk monitored at submission time

These limit the strength of the "Pilot: POSITIVE" signal in the original pipeline spec; treat the rankings as **paper-level reasoning**, not empirical pilots.
