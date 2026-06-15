<div align="center">

<img src="assets/title.svg" alt="Learning Multi-Factor Memory" width="860"/>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2606.12945-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/pdf/2606.12945)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CPU only](https://img.shields.io/badge/compute-CPU%20only-orange?style=flat-square)]()
[![No API calls](https://img.shields.io/badge/API%20calls-none-blueviolet?style=flat-square)]()

<br/>

*AIエージェントは何を記憶し、何を忘れるべきか？*

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

---

## ビジョン

数日から数週間にわたって稼働するAIエージェントは、いかなるコンテキストウィンドウをも超える膨大なインタラクション履歴を蓄積する。現在のシステムはそれに対し、二つのヒューリスティックのいずれかで応答する：**直近のものを保持する**か、**現在のクエリに最も類似したものを保持する**かだ。しかし、どちらも忘却の意思決定として誤りである。忘却は統合（consolidation）の段階で行われ、将来のクエリが存在するより前に決定される。

人間の記憶は単一要因で支配されていない。認知心理学の数十年にわたる研究が示すように、忘却を生き残るものを決めるのは、その項目の**価値（value）**——感情的な重み、目標との関連性、信頼性、自己との結びつき——である。単一の手がかりが支配することはなく、それはアンサンブルなのだ。

**本プロジェクトはこの洞察をエージェント型メモリに持ち込む。** 我々は、符号化の深度・忘却リスク・検索ランクという三つのメモリ操作すべてを、単一の解釈可能なスカラーで制御する、学習済みの多要因価値関数を定義する。重みは手動で設定するのではなく、下流の目的関数から学習される。そしてシステム全体は、APIコールなしにラップトップのCPU上で動作する。

---

## コアアイデア

### 価値関数

保存されたメモリ $m$ それぞれにスコアを付与する：

$$V(m) = \sum_{i=1}^{7} w_i \, f_i(m)$$

七つの解釈可能な要因は、それぞれ人間の記憶保持の決定要因である：

| # | 要因 | 認知科学的根拠 |
|---|------|--------------|
| 1 | **感情強度** | 覚醒が統合を調節する（McGaugh 2000） |
| 2 | **目標関連性** | 価値志向的記憶（Castel 2008） |
| 3 | **価値整合性** | 処理水準説（Craik & Lockhart 1972） |
| 4 | **自己／ユーザー関連性** | 自己参照効果（Rogers 1977） |
| 5 | **タスク有用性** | 適応的記憶（Anderson 1991） |
| 6 | **信頼性** | 情報源ヒューリスティック（ユーザー発話 > モデル発話） |
| 7 | **使用履歴** | 必要確率検索（Anderson & Milson 1989） |

スカラー $V(m)$ 一つで三つの操作を制御する：

```
encode  →  depth tier ∝ V(m)
forget  →  drop lowest V(m) under keep-budget κ
retrieve →  rank by V(m) + query match
```

### 重みの学習

符号化 → 忘却 → 検索 → 応答のパイプラインは微分不可能であるため、重み $\mathbf{w}$ は**勾配不要の最適化器**（ランダム再起動ヒルクライム）によって学習され、固定メモリバジェット下でゴールエビデンスの保持率を最大化する。学習中にLLMの呼び出しは不要である。

### オラクル／ブラインドの区別

方法論上の重要な貢献として、標準的な保持ベンチマークは目標関連性を**保留された評価質問**に対してスコアリングする——これは将来のクエリを先読みするオラクルである。この設定では~0.98に飽和し、忘却ではなく検索を測定していることになる。我々は**ブラインド体制**で評価する：統合ポリシーは評価質問を一切参照せず、実システムの動作に即している。

```
oracle regime (unfair):  goal-only → 0.979   ← measures retrieval
blind  regime (honest):  goal-only → 0.286   ← actual forgetting quality
```

---

## 有効性 — LongMemEvalベンチマーク

479件の実際のマルチセッションチャットケース、保持割合 κ = 0.30、ブラインド体制：

<div align="center">

| ポリシー | ゴールド保持率 | vs. 学習済み |
|---------|:------------:|:-----------:|
| **学習済み多要因（本手法）** | **0.770 ± 0.011** | — |
| 均一重み | 0.657 | −0.113 |
| 最良単一要因（自己関連性） | 0.518 | −0.252 |
| 直近ベースライン | 0.368 | −0.402 |

*各差の95%ブートストラップCIはすべて厳密にゼロを上回る（50/50分割による20回リサンプリング）。*

</div>

**学習済み重みは解釈可能である**——信頼性（0.64）、感情強度（0.55）、自己／ユーザー関連性（0.23）が支配的であり、クエリ時の目標類似度は統合時に利用不可能であるため、正しく低く重み付けされている（0.00）。

**同一要因上のニューラルMLPは線形モデルと同等の性能を示す（+0.003 ± 0.013）**——これは要因がほぼ加算的に結合することを確認し、解釈可能な線形価値関数が妥協の産物ではないことを示す。

---

## リポジトリ構成

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

## インストール

```bash
git clone https://github.com/zhibao-dev/Learning-Multi-Factor-Memory.git
cd Learning-Multi-Factor-Memory

# Full dev install (experiments + audit + tests)
pip install -e ".[dev]"

# Audit CLI only
pip install -e ".[audit]"
```

> **GPUは不要。** 埋め込みにはローカルのsentence-transformerを使用する（HuggingFaceキャッシュへ一度だけダウンロード）。すべての実験はCPU上で数分以内に完了する。

---

## 論文結果の再現

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

すべての結果は `results/` にキャッシュされており、LongMemEvalの生データなしにスクリプトを再実行できる。

---

## メモリ監査CLI

`borge-audit` は任意のエージェントメモリストアにおける肥大化、矛盾、重複、陳腐化したエントリを検出する。完全にローカルで動作し、メモリデータが外部に送出されることはない。**読み取り専用かつドライラン：何も削除されない。**

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

出力：`memories.audit.md`（人間が読めるレポート）＋ `memories.audit.forget.json`（可逆的な忘却リスト）。

---

## 価値関数を直接使用する

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

## ライセンス

MIT © 2026 Zhibao Chen, Qian Cheng
