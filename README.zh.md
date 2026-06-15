<div align="center">

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md) · [Português](README.pt.md)

<img src="assets/title.svg" alt="Learning Multi-Factor Memory" width="860"/>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2606.12945-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/pdf/2606.12945)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CPU only](https://img.shields.io/badge/compute-CPU%20only-orange?style=flat-square)]()
[![No API calls](https://img.shields.io/badge/API%20calls-none-blueviolet?style=flat-square)]()

<br/>

*AI 智能体应该记住什么——又应该遗忘什么？*

</div>

---

## 愿景

任何在数天乃至数周内持续运行的 AI 智能体，都会积累远超上下文窗口容量的交互历史。当前系统通常采用以下两种启发式策略之一：**保留最新内容**或**保留与当前查询最相似的内容**。然而，这两种策略对于"遗忘决策"来说都是错误的——遗忘发生在记忆巩固阶段，此时根本不存在任何未来查询。

人类记忆并非单一因素驱动。数十年的认知心理学研究表明，决定记忆能否抵御遗忘的，是记忆本身的**价值**：其情感权重、目标相关性、可靠程度以及与自我的关联。没有任何单一线索能够主导这一过程——它是多因素的集成效应。

**本项目将这一洞见引入智能体记忆领域。** 我们定义了一个可学习的多因素价值函数，通过单一可解释标量统一驾驭三类记忆操作：编码深度、遗忘风险与检索排序。权重并非手动设定，而是从下游目标中学习得到。整个系统在笔记本电脑 CPU 上运行，无需任何 API 调用。

---

## 核心思想

### 价值函数

每条存储记忆 $m$ 获得一个评分：

$$V(m) = \sum_{i=1}^{7} w_i \, f_i(m)$$

七个可解释因素，每一个都是人类记忆保留的决定因子：

| # | 因素 | 认知依据 |
|---|------|---------|
| 1 | **情感强度** | 唤醒水平调节记忆巩固（McGaugh 2000） |
| 2 | **目标相关性** | 价值导向记忆（Castel 2008） |
| 3 | **价值对齐** | 加工层次理论（Craik & Lockhart 1972） |
| 4 | **自我/用户相关性** | 自我参照效应（Rogers 1977） |
| 5 | **任务效用** | 适应性记忆（Anderson 1991） |
| 6 | **可靠性** | 信息来源启发式（用户陈述 > 模型陈述） |
| 7 | **使用历史** | 需求概率检索（Anderson & Milson 1989） |

单一标量 $V(m)$ 控制三类操作：

```
encode  →  depth tier ∝ V(m)
forget  →  drop lowest V(m) under keep-budget κ
retrieve →  rank by V(m) + query match
```

### 权重学习

编码 → 遗忘 → 检索 → 回答的完整流程不可微分，因此权重 $\mathbf{w}$ 通过**无梯度优化器**（随机重启爬山法）学习，目标是在固定记忆预算下最大化黄金证据保留率。训练过程无需任何 LLM 调用。

### Oracle 模式与盲测模式的区分

一项关键的方法论贡献：标准记忆保留基准测试会将目标相关性与**留存的评测问题**进行对比——这相当于一个能预知未来查询的 oracle。此类评测的分数可饱和至约 0.98，衡量的是检索能力，而非遗忘质量。我们在**盲测模式**下进行评估：巩固策略从不接触评测问题，与真实系统的运行方式保持一致。

```
oracle regime (unfair):  goal-only → 0.979   ← measures retrieval
blind  regime (honest):  goal-only → 0.286   ← actual forgetting quality
```

---

## 效果验证——LongMemEval 基准测试

479 个真实多轮对话场景，保留比例 κ = 0.30，盲测模式：

<div align="center">

| 策略 | 黄金证据保留率 | 与学习方法的差距 |
|------|:------------:|:-----------:|
| **学习多因素模型（本文）** | **0.770 ± 0.011** | — |
| 均匀权重 | 0.657 | −0.113 |
| 最佳单因素（自我相关性） | 0.518 | −0.252 |
| 近期性基线 | 0.368 | −0.402 |

*每项差距的 95% 自助法置信区间均严格大于零（20 次重采样 50/50 分割）。*

</div>

**学习所得的权重具有可解释性**——可靠性（0.64）、情感强度（0.55）和自我/用户相关性（0.23）占据主导地位；查询时的目标相似度被正确地下调权重（0.00），因为它在记忆巩固阶段不可获得。

**基于相同因素的神经网络 MLP 与线性模型持平（+0.003 ± 0.013）**——这证实了各因素的组合近似满足加性关系，可解释的线性价值函数并非一种妥协。

---

## 代码仓库结构

```
borge/memory/value.py        ← value function V(m) + default learned weights
borge/memory/value_net.py    ← MLP ablation model
borge/eval/                  ← LongMemEval annotator + retention metric
borge/audit/                 ← memory hygiene audit CLI
borge/affective/             ← emotional intensity factor extractor
borge/values/                ← SBert embedder, value system, soul parser
experiments/                 ← all paper experiments (CPU-only, no API)
results/                     ← cached experiment outputs
figures/                     ← paper figures (PDF)
paper/                       ← LaTeX source + compiled PDF
packaging/borge-audit/       ← standalone audit wheel
assets/                      ← readme assets
```

---

## 安装

```bash
git clone https://github.com/zhibao-dev/Learning-Multi-Factor-Memory.git
cd Learning-Multi-Factor-Memory

# Full dev install (experiments + audit + tests)
pip install -e ".[dev]"

# Audit CLI only
pip install -e ".[audit]"
```

> **无需 GPU。** 向量嵌入使用本地 sentence-transformer 模型（首次运行时下载至 HuggingFace 缓存）。所有实验均在 CPU 上运行，耗时数分钟。

---

## 复现论文结果

```bash
# Table 2 headline — blind forgetting on LongMemEval
python experiments/lme_blind_forgetting.py

# Per-case bootstrap CI (Figure 3)
python experiments/lme_bootstrap_ci.py

# Keep-fraction sweep κ ∈ {0.1 … 0.9} (Figure 2)
python experiments/lme_keepfrac_sweep.py

# Synthetic confound study — Table 1
python experiments/e2_mood_self_factorial.py

# MLP ablation
python experiments/lme_blind_forgetting.py --model mlp
```

所有结果均已缓存至 `results/` 目录——无需原始 LongMemEval 数据即可重新运行脚本。

---

## 记忆审计 CLI

`borge-audit` 可在任意智能体记忆存储中发现冗余、矛盾、重复和过期条目。它完全在本地运行——记忆数据始终不会离开您的设备。**只读模式与预演模式：不会删除任何内容。**

```bash
# JSON dump (any agent: Mem0, Zep, pgvector, custom)
borge-audit memories.json

# Markdown dump (heading-split)
borge-audit memories.md

# Keep 30 % (more aggressive)
borge-audit memories.json --budget 0.3

# With optional LLM judge for contradiction precision
borge-audit memories.json --llm-endpoint http://localhost:11434/v1 --llm-model qwen3:8b
```

输出：`memories.audit.md`（人类可读报告）+ `memories.audit.forget.json`（可逆遗忘列表）。

---

## 直接使用价值函数

```python
from borge.memory.value import default_memory_value, memory_factors

mv = default_memory_value()          # learned weights from the LongMemEval blind fit

# A memory row carries the factor inputs (valence/arousal → emotion,
# reliability, self-relevance, retrieval_count → usage, ...).
reliable_fact = {
    "emotional_valence": 0.3, "emotional_arousal": 0.6,
    "reliability": 1.0, "self_relevance_score": 0.8, "retrieval_count": 4,
}
chatter = {
    "emotional_valence": 0.0, "emotional_arousal": 0.1,
    "reliability": 0.4, "self_relevance_score": 0.1, "retrieval_count": 0,
}

v_fact    = mv.value(memory_factors(reliable_fact))   # high  → keep
v_chatter = mv.value(memory_factors(chatter))         # low   → forget candidate
print(round(v_fact, 3), round(v_chatter, 3))
```

---

## 引用

```bibtex
@article{chen2026multifactor,
  title   = {Learning What to Remember: A Cognitively Grounded
             Multi-Factor Value Model for Agentic Memory},
  author  = {Chen, Zhibao and Cheng, Qian},
  journal = {arXiv preprint arXiv:2606.12945},
  year    = {2026},
  url     = {https://arxiv.org/pdf/2606.12945}
}
```

---

## 许可证

MIT © 2026 Zhibao Chen, Qian Cheng
