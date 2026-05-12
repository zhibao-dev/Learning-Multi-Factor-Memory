<div align="center">

```
████   ███  ████   ████ █████    ███   ████ █████ █   █ █████
█   █ █   █ █   █ █     █       █   █ █     █     ██  █   █  
████  █   █ ████  █  ██ ████    █████ █  ██ ████  █ █ █   █  
█   █ █   █ █  █  █   █ █       █   █ █   █ █     █  ██   █  
████   ███  █   █  ████ █████   █   █  ████ █████ █   █   █  
```

<p><strong>🧠 第一个拥有认知架构的 AI Agent。</strong><br>
它能感受你的情绪，会承认自己不知道的事，记住重要的——并主动遗忘琐碎的。</p>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Theory: Friston FEP](https://img.shields.io/badge/Theory-Free_Energy_Principle-8b5cf6)](https://en.wikipedia.org/wiki/Free_energy_principle)
[![Standalone + Plugin](https://img.shields.io/badge/Mode-Standalone_%2B_Hermes_Plugin-f59e0b)](https://github.com/zhibao-dev/BorgeAgent)

[English](README.md) · **简体中文**

<br>

```
你已经说了两遍。Agent 还是不明白。

用 Borge：
  第 1 轮 → V=+0.0  A=0.45  [中性, 专注]
  第 3 轮 → V=-0.3  A=0.62  [挫败] → 切换模式: SIMPLIFY
  第 5 轮 → "我换一种方式问一个具体的问题。"
```

*它察觉到了，它适应了。不需要任何 prompt 工程。*

</div>

---

## 当下每一个 Agent 的通病

今天所有 AI Agent 底层都长一个样：

```
用户输入 → LLM → 调工具 → 输出 → 忘掉一切
```

**没有状态。不知道这次对话进展如何。无法判断自己是在帮上忙，还是在白费力气。**

- 它不知道你已经烦躁了 —— 还在喋喋不休地解释。
- 它不知道自己卡了三轮 —— 还在重复调用同一个工具。
- 它不记得你上周说过的偏好 —— 每次都从零开始。
- 它选工具靠拍脑袋 —— 而不是按"哪个能最快降低不确定性"。

Borge 一次性解决这些。不靠 prompt 技巧，靠**认知科学**。

---

## Borge 究竟是什么

Borge 是一个**框架无关、模型无关**的认知层。

- **框架无关** —— 可以独立运行（自带 `borge` CLI），也可以作为非侵入式插件挂载到任何 agent 上（Hermes、OpenClaw、你自己的）。
- **模型无关** —— 兼容 Anthropic、OpenAI、Kimi、MiniMax、DeepSeek、Zhipu（智谱）、Ollama、vLLM，以及任何能从 Python 调起来的 LLM。详见 [接入任意模型](#接入任意模型)。

它实现了来自神经科学与认知心理学的四套系统：

| 系统 | 它做什么 | 理论依据 |
|------|---------|---------|
| **情感状态** | 逐轮追踪你的情绪基调，并相应调整 agent 行为 | Russell Circumplex (1980) |
| **贝叶斯信念状态** | 维护显式的假设概率分布 —— agent 清楚知道自己不知道什么 | 预测编码 (Knill & Pouget 2004) |
| **主动推断** | 选择能最大化"信息增益 *和* 目标进展"的工具 | Friston 自由能原理 (2010) |
| **认知记忆** | 像大脑一样编码、巩固、*遗忘* 记忆 —— 而非堆积式数据库 | Tulving (1972), Ebbinghaus (1885) |

这些不是比喻。它们是真实运行的实现。每一轮，Borge 计算：

```
F_total = F_epistemic × precision(arousal)
        + F_pragmatic × (1 - value_alignment)
        + F_homeostatic(valence, arousal)
```

并用这个数值驱动行为。**最小化 F_total 是 agent 唯一的目标** —— 所有有意思的涌现行为都来自这一个目标函数。

---

## 60 秒安装

**独立运行（推荐）** —— 极简 agent 循环，只依赖 Anthropic SDK：

```bash
git clone https://github.com/zhibao-dev/BorgeAgent.git
cd BorgeAgent
pip install -e ".[anthropic]"

export ANTHROPIC_API_KEY=sk-...
borge                    # 交互式 REPL
borge "fix the auth bug" # 单轮执行
```

**作为 Hermes 插件** —— 把 `plugins/hermes/` 放到你的 Hermes 插件路径下：

```bash
pip install -e ".[hermes]"
# plugins/hermes/ 会自动注册 4 个生命周期 hook，正常运行 hermes 即可
hermes
```

完成。认知层已经在工作。无需任何配置就能用。

---

## 接入任意模型

**认知层是完全模型无关的。** Borge 并不绑死任何一个 LLM —— 它提供的是 4 个生命周期 hook（`on_session_start` / `pre_turn` / `post_tool` / `on_session_end`），你把它们包在 *任意* LLM 调用周围即可。自带的 `borge` CLI 用 Anthropic SDK，只是因为它依赖最简洁；换 provider 大约 15 行代码。

### OpenAI 协议兼容的 Provider（一套代码，多家厂商）

国内主流云（Kimi、MiniMax、DeepSeek、Zhipu/智谱）+ OpenAI 自己 + 本地 runtime（Ollama、vLLM、LM Studio、llama.cpp server），都对外暴露 OpenAI 协议兼容的 `/v1/chat/completions` 端点。一套适配器全搞定：

```python
from openai import OpenAI
from borge.agent import BorgeAgent

client = OpenAI(
    api_key=os.environ["KIMI_API_KEY"],
    base_url="https://api.moonshot.cn/v1",    # Kimi（月之暗面）
)
borge = BorgeAgent(agent_backend=None)
borge.on_session_start()

history = []
user_msg = "帮我修 auth 这个 bug"

ctx = borge.pre_turn(user_msg, history)                                # ← 认知层
history.append({"role": "user", "content": f"{ctx}\n\n{user_msg}" if ctx else user_msg})

reply = client.chat.completions.create(model="moonshot-v1-8k", messages=history)
reply_text = reply.choices[0].message.content
history.append({"role": "assistant", "content": reply_text})

borge.post_tool("assistant_turn", reply_text)                          # ← 认知层
borge.on_session_end(session_id="sess-001", messages=history)          # ← 认知层
```

换 provider 只需替换 `base_url`：

| Provider          | `base_url`                                      | 示例模型               |
|-------------------|-------------------------------------------------|-----------------------|
| Kimi（月之暗面）   | `https://api.moonshot.cn/v1`                    | `moonshot-v1-8k`      |
| MiniMax           | `https://api.minimax.chat/v1`                   | `abab6.5s-chat`       |
| DeepSeek          | `https://api.deepseek.com`                      | `deepseek-chat`       |
| Zhipu（智谱 GLM）  | `https://open.bigmodel.cn/api/paas/v4`          | `glm-4-flash`         |
| OpenAI            | *(默认)*                                         | `gpt-4o-mini`         |
| Ollama（本地）     | `http://localhost:11434/v1`                     | `llama3.2`、`qwen2.5` |
| vLLM（自部署）     | `http://your-host:8000/v1`                      | *(任意已加载模型)*     |
| LM Studio（本地）  | `http://localhost:1234/v1`                      | *(任意已加载模型)*     |

可直接运行的示例：[`examples/multi_provider.py`](examples/multi_provider.py) —— 设置 `PROVIDER=kimi|minimax|deepseek|zhipu|openai|ollama|vllm` 即可跑通。

### Anthropic SDK（自带 —— `borge` CLI 走这条）

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-...
borge "解释一下这个 bug"
```

### Hermes 插件模式（Hermes 配什么模型就用什么）

作为 Hermes 插件加载时，Borge 会增强 *Hermes 当前配置的* 任何模型 —— Claude、GPT-4、本地 Llama 都行，由 Hermes 的 provider 配置决定。Borge 只负责认知层，LLM I/O 交给 Hermes。

### 其他 SDK

同样的模式适用于任何 SDK —— Google Gemini、AWS Bedrock、Azure OpenAI、Cohere、自研客户端。配方永远是：

```
user_msg → borge.pre_turn() → 把 ctx 注入到你的 prompt → 调 LLM
       → borge.post_tool() → 重复
结束:  → borge.on_session_end()
```

---

## 看它工作

### 挫败反应

```python
# 第 1 轮 —— 中性开场
User: "帮我修 auth 这个 bug"
# emotion: V=+0.0  A=0.45  mode: NORMAL

# 第 3 轮 —— 用户开始没耐心
User: "不是这个问题，我已经检查过了"
# 信号: ΔV=-0.25（否定 + "已经"）
# emotion: V=-0.22  A=0.58  mode: SIMPLIFY
# 注入: "[Affective: frustrated — switch to focused, minimal responses]"

# Agent 收敛到一个假设，问一个问题，停止过度解释。
```

### 不确定反应

```python
# Agent 有 4 个相互竞争的假设，熵 = 2.0 bits
# EFE 对工具排序：
#   ask_user      EFE=-1.4  ← epistemic value 占主导
#   read_file     EFE=-0.8
#   bash          EFE=-0.3

# Agent 先问一个澄清问题，而不是调工具
# 因为降低信念熵是当前价值最高的行动
```

### 停滞反应

```python
# F_total: [0.82, 0.85, 0.88]  —— 连续 3 轮在涨
# MetaAgent: 触发反思

# 注入: "[Meta: Free energy stagnating — try a different approach
#         or ask the user for clarification]"

# Agent 换策略，不再死磕同一个工具
```

### 跨会话记忆

```python
# 与同一用户的第 12 个会话
# loyalty_tracker: V_baseline=+0.31（长期关系温暖）
# 注入: "[Relationship: established trust — be direct, skip caveats]"

# 经历过一次糟糕会话后的第 13 个会话
# V_baseline=+0.18（关系降温）
# Agent 开场更谨慎，在假设之前先确认
```

---

## 完整工作流

```
┌─────────────────────────────────────────────────────────────────┐
│                       每一轮                                     │
│                                                                  │
│  用户消息                                                        │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────┐   39 条语言规则       ┌────────────────┐ │
│  │ 信号提取器       │ ──────────────────────► │ 情感状态       │ │
│  │ (中文 + 英文)    │   ΔV, ΔA              │ Russell 2D     │ │
│  └─────────────────┘                         │ V × A → 模式  │ │
│                                              └───────┬────────┘ │
│  ┌─────────────────┐                                 │          │
│  │  信念状态        │   香农熵                       │          │
│  │  p(H₁)…p(Hₙ)   │ ──────────────┐                │          │
│  └─────────────────┘               │                │          │
│                                    ▼                ▼          │
│  ┌─────────────────┐   ┌──────────────────────────────────┐   │
│  │  价值系统        │──►│      扩展自由能 F_total          │   │
│  │  SOUL.md        │   │  F = F_ep × prec + F_pr + F_hm   │   │
│  └─────────────────┘   └──────────────┬───────────────────┘   │
│                                        │                        │
│                          ┌─────────────▼──────────────┐        │
│                          │       MetaAgent             │        │
│                          │  • 模式 → 上下文注入        │        │
│                          │  • 停滞 → 触发反思          │        │
│                          │  • 按 EFE 排序工具          │        │
│                          └─────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    会话结束 ("睡眠")                              │
│                                                                  │
│  对话 → 抽取实体 → 知识图谱更新                                  │
│       → 检测矛盾 → 重要性打分                                   │
│       → 情感编码深度 → 技能候选                                │
│       → Ebbinghaus 遗忘扫描                                      │
│                                                                  │
│  下一次会话：忠诚度基线根据 V_avg 漂移                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 个性化配置

三层定制能力，从浅到深：

### 第 1 层 —— `SOUL.md`（你 agent 的人格，5 分钟）

把 **`SOUL.md`** 放在项目根目录，或放 `~/.borge/SOUL.md` 作为系统级默认（Hermes 插件模式下也会读 `~/.hermes/SOUL.md`）：

```markdown
---
emotional_defaults:
  valence_baseline: 0.1       # 轻微正向起点（–1.0 .. +1.0）
  arousal_baseline: 0.45      # 平静但警觉（0.0 .. 1.0）
  tau_valence: 5.0            # 情绪回归基线所需轮数
  tau_arousal: 3.0
  frustrated_threshold: -0.3  # V 低于此值，切换到 SIMPLIFY 模式
  excited_threshold: 0.7      # V 高于此值，切换到 EXPLORE 模式

values:
  - name: help_genuinely
    weight: 0.9
    description: "解决真问题，而非表演性完成表面需求。"
  - name: intellectual_honesty
    weight: 0.85
    description: "不确定时说'我不知道'。绝不胡说。"
  - name: depth_over_speed
    weight: 0.7
    description: "慢一点的正确答案胜过快速的错误答案。"
  - name: respect_autonomy
    weight: 0.8
    description: "假设之前先问。删除之前先确认。"
---

你是一个会先思考再开口的合作者。
卡住的时候你会直接说，并提出另一个角度。
```

`values` 段塑造**实用自由能**项 —— `intellectual_honesty: 0.95` 的 agent 在数学上偏好那些"暴露不确定性"的行动，而非"假装自信"。`emotional_defaults` 决定 agent 的静息状态和反应灵敏度（例如更小的 `tau_valence` → 情绪波动更快）。

### 第 2 层 —— `config.yaml`（子系统调参）

新建 `~/.borge/config.yaml`（独立模式）或在 `~/.hermes/config.yaml` 下加 `borge:` 节点（插件模式）：

```yaml
borge:
  affective:
    enabled: true
    loyalty:
      enabled: true                       # 跨会话情绪基线

  beliefs:
    enabled: true
    entropy_injection_threshold: 0.5      # bit —— 超过此值才注入信念摘要

  active_inference:
    enabled: true                          # 按 EFE 重排序工具

  memory:
    consolidation:
      enabled: true                       # 会话结束时跑 7 步管道
    knowledge_graph:
      enabled: true
    forgetting:
      prune_threshold: 2.0                # forget_score 超过此值 → 删除
```

每个子系统都有 `enabled` 开关 —— 不需要的可以单独关掉（例如 `beliefs.enabled: false` 跑一个纯情感 agent）。

### 第 3 层 —— 继承 `BorgeAgent`（代码级）

要替换内部引擎的进阶用法（自定义情感模型、特定领域的信念表示等）：

```python
from borge.agent import BorgeAgent
from borge.affective.signal_extractor import EmotionalSignalExtractor

class ChineseSignalExtractor(EmotionalSignalExtractor):
    """用中文调优的规则替换默认的 39 条规则。"""
    def extract(self, message, history):
        # 你的语言学规则
        return delta_v, delta_a

class MyBorge(BorgeAgent):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._signal_extractor = ChineseSignalExtractor()
```

`BorgeAgent` 的设计就鼓励替换 —— 所有引擎（`_signal_extractor`、`_loyalty_tracker`、`_meta`、`_afe`、`_kg`、`_forgetting`、`_consolidation`、`_skill_evolution`）都是 init 后可替换的公共属性。

### 环境变量

| 变量                  | 默认值                | 含义                                       |
|----------------------|----------------------|-------------------------------------------|
| `ANTHROPIC_API_KEY`  | *(CLI 必需)*          | 自带 Anthropic runner 的 API key          |
| `BORGE_MODEL`        | `claude-opus-4-7`    | `borge` CLI 默认使用的模型                |
| `BORGE_HOME`         | `~/.borge`           | `SOUL.md` 和 `borge.db` 的存放位置        |

---

## 架构 —— 零侵入

一个 `BorgeAgent` 认知核心，通过 4 个生命周期 hook 同时支持两种部署形态。**宿主 agent 零文件改动。**

```
                    on_session_start ──► 忠诚度基线 / 重置状态
                    pre_turn         ──► 注入认知上下文字符串
                    post_tool        ──► 贝叶斯信念更新
                    on_session_end   ──► 记忆巩固管道
                          ▲
        ┌─────────────────┴─────────────────┐
        │                                   │
BorgeRunner（独立运行）              plugins/hermes/  (~150 行 —— 纯胶水代码)
   Anthropic SDK 循环                     Hermes Agent（未改动）

borge/          (认知层实现)
    ├── affective/      Russell 2D, 信号提取, 忠诚度
    ├── beliefs/        贝叶斯假设追踪
    ├── inference/      主动推断, EFE 评分
    ├── memory/         4 级编码深度, 知识图谱, 遗忘
    ├── meta/           自由能, 中央执行系统
    ├── values/         SOUL.md, 价值系统, 约束检查
    ├── skill_evolution.py   技能库的达尔文式适应度
    └── agent.py        BorgeAgent —— 主集成接口
```

移除插件，Hermes 恢复到 vanilla 状态。没有遗留状态，没有破损 schema。独立模式拥有自己的 SQLite 存储 `$BORGE_HOME/borge.db`（默认 `~/.borge/borge.db`）。

---

## 模块速查

| 模块 | 理论 | 核心公式 / 机制 |
|------|------|---------------|
| `affective.emotional_state` | Russell Circumplex (1980) | EMA 更新: `V += α(ΔV)`, α=1/τ |
| `affective.signal_extractor` | 心理语言学 | 39 条规则 → `(ΔV, ΔA)`，上限 ±0.4/±0.3 |
| `affective.loyalty_tracker` | 依恋理论 | `w = exp(-0.05·days) × msg_count` |
| `beliefs.belief_state` | 贝叶斯大脑 | `H = -Σ p·log₂p` (bits) |
| `inference.active_inference` | Friston FEP (2010) | `G(a) = -EV(a) - PV(a)` |
| `memory.cognitive_memory` | Craik & Lockhart (1972) | 深度 ∈ {SHALLOW, SEMANTIC, SCHEMATIC, META} |
| `memory.knowledge_graph` | 语义记忆 (Tulving) | 纯 SQLite，不依赖 networkx |
| `memory.consolidation` | 睡眠巩固 | 7 步离线管道 |
| `memory.forgetting` | Ebbinghaus (1885) | `score = days^0.7 / (retrieval × importance × connections)` |
| `meta.free_energy` | FEP | `F = F_ep·prec + F_pr + F_hm` |
| `meta.meta_agent` | Baddeley 中央执行系统 (1974) | 连续 3 轮 F 不下降即触发反思 |
| `values.value_system` | 价值对齐 | `F_pragmatic = 1 - V_alignment` |
| `skill_evolution` | 进化动力学 | `fitness = success_rate × log(1+n) × recency × Δfree-energy` |

---

## 相比 Hermes 的提升

Hermes 是个扎实的底座 —— Borge **原封不动** 继承了它的工具注册表、IoC 回调、网关适配、SQLite 会话存储、Cron 调度。提升发生在上面一层：完整的认知基质。

### 让状态在轮次之间持续存在

普通 agent（Hermes 也是）对"对话进行得如何"**没有任何意见**。每一轮独立处理：读消息 → 调 LLM → 输出。Borge 追踪：

- **情感状态**（Russell V/A） —— agent 知道你在烦躁（V 下降、A 上升的语言标志：否定 + "已经"类词）并切换到 `SIMPLIFY` 模式：更简短的回复、问一个聚焦的问题而不是三个。检测到你在投入，切换到 `EXPLORE`，深入展开。
- **信念状态**（贝叶斯） —— 显式的假设概率分布 `p(H_i)`。熵高（> 0.5 bits）时，agent 会先问澄清问题再调工具。熵低时，直接 commit。
- **自由能** 轨迹 —— agent 监控过去 5 轮的 `F_total`。连续 3 轮没下降，`MetaAgent` 注入一段反思提示："*Free energy stagnating — try a different approach or ask the user for clarification.*"。普通 agent 只会死磕同一个工具。

### 让记忆真正像记忆一样工作

Hermes 把所有消息永久存进 SQLite —— 当 log 用还行，但每条消息权重一样。Borge 加了：

- **编码深度**（Craik & Lockhart 1972） —— 情感显著的消息得到更深的编码。会话结束的巩固管道计算 `significance = |valence| × arousal`；高显著性的时刻被编码到 `SCHEMATIC` / `META` 深度，能从遗忘扫描中幸存。
- **主动遗忘**（Ebbinghaus 1885） —— `forget_score = days^0.7 / (retrieval × importance × connections)`。会话结束时低价值记忆被剪枝。数据库即使涨到 1000+ 会话也不会退化。
- **知识图谱** —— 会话结束时抽取的实体/关系形成可查询的语义记忆（SQLite，不依赖 networkx）。未来的检索是图遍历，不仅仅是 FTS。
- **跨会话忠诚度** —— 与同一用户经过 N 次会话后，`LoyaltyTracker` 根据时间衰减的历史情绪偏移 `valence_baseline`。10 个温暖会话历史的用户会收到 *"established trust — be direct, skip caveats"* 的开场。降温的关系会触发更谨慎的开场。

### 用信息论选工具

Hermes 靠 LLM 直觉选工具。Borge 用 **期望自由能** 对 LLM 提议的工具调用重新排序：

```
G(tool) = -(认知价值 + 实用价值)
       = -(期望熵减 + 期望目标进展)
```

信念熵高时 `ask_user` 胜出（认知价值最大）。熵低时 `bash` / `read_file` 胜出（实用价值主导）。两者权重由唤醒度调制 —— 高唤醒偏向探索。结果：agent 不会在同一个出错工具上撞三遍。

### Soul 驱动，不是 prompt 工程

`SOUL.md` 不是 system prompt —— 它是一个**类型化的价值系统**，参与自由能的计算。一个 `intellectual_honesty: 0.95` 的 agent 在数学上偏好那些与该价值的 `V_alignment` 高的行动。行为是从目标函数 `F_total` 涌现的，而非来自字符串模板。把权重从 `0.5` 调到 `0.95`，会得到一个**可测量地不同的** agent —— 无需改任何 prompt。

### 特性对比

|  | LangChain | AutoGPT | Hermes | **Borge** |
|--|:---------:|:-------:|:------:|:---------:|
| 工具调用 | ✓ | ✓ | ✓ | ✓ |
| 技能库 | 部分 | ✗ | ✓ | ✓ |
| 多 provider LLM | ✓ | ✓ | ✓ | ✓ |
| 情感状态 | ✗ | ✗ | ✗ | **✓** |
| 贝叶斯信念追踪 | ✗ | ✗ | ✗ | **✓** |
| 信息论工具选择 | ✗ | ✗ | ✗ | **✓** |
| 编码深度记忆 | ✗ | ✗ | ✗ | **✓** |
| 主动遗忘 | ✗ | ✗ | ✗ | **✓** |
| 跨会话关系模型 | ✗ | ✗ | ✗ | **✓** |
| 自由能目标函数 | ✗ | ✗ | ✗ | **✓** |
| 停滞检测 + 反思 | ✗ | ✗ | ✗ | **✓** |

---

## 理论基础

Borge 立足于同行评议的认知科学 —— 不是凭直觉。

| 论文 | 年份 | 贡献 |
|-----|------|------|
| Ebbinghaus, *Memory: A contribution to experimental psychology* | 1885 | 遗忘曲线 → 主动记忆衰退 |
| Yerkes & Dodson | 1908 | 唤醒度 × 表现 → 最优唤醒区间 |
| Tulving, *Episodic and semantic memory* | 1972 | 记忆分类 → 三层架构 |
| Craik & Lockhart, *Levels of processing* | 1972 | 编码深度 → 情感显著性驱动巩固 |
| Baddeley & Hitch, *Working memory* | 1974 | 中央执行系统 → MetaAgent 设计 |
| Russell, *A circumplex model of affect* | 1980 | 二维情绪空间 → V × A 状态 |
| Knill & Pouget, *The Bayesian brain* | 2004 | 预测编码 → 信念状态 |
| Friston, *The free-energy principle* | 2010 | 统一目标函数 → F_total |
| Friston et al., *Active inference* | 2017 | EFE 工具评分 |

完整推导见 [`docs/borge-agent-design.md`](docs/borge-agent-design.md)。

---

## 路线图

```
v0.1  ██████████ 已完成   核心认知层 + Hermes 插件集成
v0.2  ░░░░░░░░░░          LLM 驱动的贝叶斯更新（真实似然估计）
v0.2  ░░░░░░░░░░          Hermes pre_tool_call hook，支持实时 EFE 评分
v0.3  ░░░░░░░░░░          多 agent 情绪传染
v0.3  ░░░░░░░░░░          反事实信念修正
v0.4  ░░░░░░░░░░          SOUL.md 根据会话遥测自动调参
v0.5  ░░░░░░░░░░          100 轮长任务的认知一致性基准测试
```

---

## 贡献

当前最有价值的贡献方向：

- **实证验证** —— 在长程编码任务上对比 Borge 与 vanilla
- **更丰富的信号提取** —— 更好的语调检测语言规则
- **替代情绪模型** —— PAD（3D）、OCC 模型、基本情绪
- **LLM 似然估计器** —— 用真实 LLM 调用替换启发式贝叶斯更新

```bash
git clone https://github.com/zhibao-dev/BorgeAgent
cd BorgeAgent && pip install -e ".[dev]"
python -c "from borge.agent import BorgeAgent; a = BorgeAgent(None); print(a.pre_turn('你好', []))"
pytest  # 应该 11/11 通过
```

---

## 引用

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

MIT。最初作为 [Hermes Agent](https://github.com/NousResearch/hermes-agent)（Nous Research）的插件开发 —— 现在是一个既可独立运行、也可作为 Hermes 插件挂载的框架。

---

<div align="center">

**[设计文档](docs/borge-agent-design.md) · [Issues](../../issues) · [Discussions](../../discussions)**

<br>

*多数 agent 跑得快。Borge 在场。*

</div>
