# borge audit MVP — Design

> Branch: `multi-factor-eval-business` (商业化层；不并回研究分支 main/self-FEP/multi-factor-eval)
> Date: 2026-05-29
> Status: design validated (brainstorm Sections 1–4), ready for writing-plans

## 一句话

`borge audit` —— 一个你在**付费审计**中自己跑的内部 CLI：客户交一份记忆 dump，你回一份审计报告（可遗忘清单 + 矛盾/污染候选 + 省 $ + 留存分）。客户什么都不装，数据不出他们的基础设施。服务先行验证付费意愿，验证后再产品化（self-hosted MCP + license-key）。

## 商业定位（来自策略讨论）

- **楔子 = 记忆卫生层（遗忘 + 抗污染），坐在任何记忆存储之上**，不是又一个记忆框架。
- **交付 = run-in-their-infra**：记忆是 agent 最敏感数据，绝不做"把数据传给我"的 hosted API。
- **变现 = 卖能力不卖高频调用**：审计服务 → license-key → 企业；不给高频记忆操作按量计费（与"遗忘=减少记忆"反向）。
- MVP = 验证步骤（卖 3 个手动审计），不是产品化步骤。

## 范围 + 输入输出契约（Section 1）

**输入** —— 一个标准 JSON：
```json
[{"id": "...", "text": "...", "timestamp": "ISO8601",
  "role": "user|assistant|system", "metadata": {...}}]
```
每客户的存储 schema 不同 → 在付费交付里手写 ~20 行 adapter。MVP 带标准 JSON loader + 一个示例 adapter。无实时连接器。

**输出** —— markdown（可选 PDF）报告，4 个产物：
1. 臃肿/可遗忘清单（分级 安全→激进）
2. 矛盾/污染候选（供人工 review）
3. 去重 + 过期
4. 头条数字（存储缩减 % / 月省 $ / 预算内留存分）

**硬约束**：API-free（本地 SBert + 本地 NLI）；数据不出 infra；**绝不自动删**（只出建议 + 可回滚脚本）。

## Pipeline（Section 2）

**复用现有（零改动）：**
- `values/self_model.py::SBertEmbedder` —— embedding 基座
- `memory/value.py::MemoryValue` + `memory_factors` + `default_memory_value` —— 价值→遗忘排序
- `affective/signal_extractor.py::EmotionalSignalExtractor` —— emotion 因子
- blind 因子逻辑（无未来 query）：goal=cos(记忆, 会话/全局 topic 质心)；self=cos(记忆, μ_user 质心)；reliability=role 启发；value_alignment=cos(SOUL) 或 0；usage 从 metadata 或 0

**新建（`borge/audit/`）：**
1. `ingest.py` —— 标准 JSON loader + 校验
2. `contradiction.py` —— ① embedding 预筛同主题对（避免 N²）② 本地 NLI cross-encoder（`sentence-transformers` `CrossEncoder`，如 `cross-encoder/nli-deberta-v3-small`）打 contradiction 分 ③ 排序候选对。纯本地。
3. `hygiene.py` —— 近重复簇（cos 阈值）+ 按龄过期 + "被更新记忆取代"
4. `savings.py` —— forget-list token 量（tiktoken 或 char/4）× 假定检索频率 × 单价；假设显式
5. `report.py` —— markdown 拼装 + 头条 + 可回滚脚本（forget-id JSON）
6. CLI：`borge audit <dump.json> --soul <soul.md> --budget 0.3 -o report.md`

**关键**：矛盾模块是唯一新 ML 依赖（一个本地 NLI 模型）。预筛把成本从 N² 降到每条只比同主题少数几条。

## 报告（Section 3）

1. **执行摘要（首半页=付费时刻）**：存储缩减 X% / 月省 $Y / N 条污染候选 / 留存 Z%。细分客户换 lead（垂直 agent 看 $，陪伴看污染例子）。
2. **臃肿/遗忘**：分级清单 + 留存权衡曲线 + top 样例。
3. **污染（视觉冲击）**：排序矛盾对（A vs B、NLI 分、判哪条过期）+ 重复簇 + 过期；给具体例子。
4. **安全/dry-run**：绝不自动删；分级（人选档）；可回滚脚本；可选 replay（客户给 query 日志 → "套安全档，近 30 天成功召回 X% 仍命中"）。
5. **方法附录（诚实）**：怎么算的；污染=候选非自动消解；遗忘=paper2 价值模型；$ 假设全列。

## 与 paper2 的关系（诚实边界）

- **有论文背书**：遗忘/价值排序 + 留存分 = paper2 核心（`MemoryValue`，full-479 blind 0.770 vs recency 0.368，84 测试）。
- **超出 paper2（产品驱动，无研究验证）**：矛盾检测 + 去重 + 过期。paper2 只标低价值，不标矛盾。卖的时候别把污染检测说成"论文证明的"。
- 矛盾信号**不并入 V**（保 paper2 7 因子纯度）；以后验证有用 → paper3 的第 8 因子。

## 砍掉（YAGNI，Section 4）

实时连接器 / runtime 接线 / SaaS 面板 / 计费 / 自动删 / license-gate / 在线学权重 / 多语言 NLI / PDF 美化 / 矛盾并入 V。

## 诚实风险（Section 4）

1. NLI 在记忆文本上会误判（短/口语/实体中心 ≠ MNLI 句对）→ 缓解：human-review 候选、真实 dump 调阈值。
2. $ 估算依赖假定检索频率 → 显式列假设、用真实 query 日志校准。
3. 没真实 dump 测不准 → 第一个客户 = 共同开发（折扣换 dump）。
4. paper2 价值模型在客户领域迁移未验证 → 说"受 paper2 启发"非"保证 0.770"。
5. NLI ~400MB、CPU 慢 → MVP 限 dump ≤2 万条或分批。

## 验证路径

技术已验证（论文+测试）；**市场未验证（0 用户）**。下一步不是写更多代码，是：(1) 点名 10 个目标团队（陪伴 + 垂直 agent）；(2) 10 次对话听"你们怎么决定 agent 忘什么"；(3) 卖 3 个手动审计（`borge audit` 跑他们 dump → 报告）。3 个 yes = 验证 → 再产品化（MCP + license）。

## 测试策略（MVP）

- ingest：标准 JSON 往返 + 坏数据降级。
- contradiction：植入已知矛盾对的小 fixture → NLI 标出；无矛盾对不误报（阈值）。
- hygiene：植入重复/过期 → 标出。
- savings：确定性 token 估算 + 假设输出。
- report：跑通端到端，markdown 含 5 节 + 可回滚脚本。
- 全程 API-free、确定性、可在 CI 跑（NLI 模型本地缓存）。
