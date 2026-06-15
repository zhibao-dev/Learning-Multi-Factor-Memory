<div align="center">

<img src="assets/title.svg" alt="Learning Multi-Factor Memory" width="860"/>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2606.12945-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/pdf/2606.12945)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CPU only](https://img.shields.io/badge/compute-CPU%20only-orange?style=flat-square)]()
[![No API calls](https://img.shields.io/badge/API%20calls-none-blueviolet?style=flat-square)]()

<br/>

*O que um agente de IA deve lembrar — e o que deve esquecer?*

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

---

## Visão Geral

Todo agente de IA implantado por dias ou semanas acumula um histórico de interações muito maior do que qualquer janela de contexto comporta. Os sistemas atuais respondem a isso com uma de duas heurísticas: **manter o mais recente** ou **manter o mais similar à consulta atual**. Ambas são inadequadas para a decisão de esquecimento, que ocorre no momento da consolidação — antes que qualquer consulta futura exista.

A memória humana não é regida por um único fator. Décadas de psicologia cognitiva demonstram que o que sobrevive ao esquecimento é determinado pelo **valor** de um item: seu peso emocional, sua relevância para os objetivos, sua confiabilidade, sua conexão com o self. Nenhuma pista isolada predomina — trata-se de um conjunto.

**Este projeto traz essa perspectiva para a memória agêntica.** Definimos uma função de valor aprendida e multifatorial que governa as três operações de memória — profundidade de codificação, risco de esquecimento e ranqueamento na recuperação — por meio de um único escalar interpretável. Os pesos não são definidos manualmente; eles são aprendidos a partir de um objetivo downstream. E todo o sistema roda em um CPU de laptop, sem chamadas a APIs externas.

---

## Ideia Central

### A Função de Valor

Cada memória armazenada $m$ recebe uma pontuação:

$$V(m) = \sum_{i=1}^{7} w_i \, f_i(m)$$

Sete fatores interpretáveis, cada um determinante da retenção humana:

| # | Fator | Fundamentação cognitiva |
|---|-------|------------------------|
| 1 | **Intensidade emocional** | A ativação fisiológica modula a consolidação (McGaugh 2000) |
| 2 | **Relevância para objetivos** | Memorização orientada por valor (Castel 2008) |
| 3 | **Alinhamento de valores** | Níveis de processamento (Craik & Lockhart 1972) |
| 4 | **Relevância para o self/usuário** | Efeito de autorreferência (Rogers 1977) |
| 5 | **Utilidade para a tarefa** | Memória adaptativa (Anderson 1991) |
| 6 | **Confiabilidade** | Heurística de proveniência (declarado pelo usuário > declarado pelo modelo) |
| 7 | **Histórico de uso** | Recuperação por probabilidade de necessidade (Anderson & Milson 1989) |

Um único escalar $V(m)$ controla três operações:

```
encode  →  depth tier ∝ V(m)
forget  →  drop lowest V(m) under keep-budget κ
retrieve →  rank by V(m) + query match
```

### Aprendizado dos Pesos

O pipeline codificar → esquecer → recuperar → responder é não-diferenciável, portanto os pesos $\mathbf{w}$ são aprendidos por um **otimizador sem gradiente** (hill-climb com reinicializações aleatórias) que maximiza a retenção de evidências-ouro sob um orçamento de memória fixo. Nenhuma chamada a LLMs é necessária durante o treinamento.

### A Distinção Oráculo / Cego

Uma contribuição metodológica central: os benchmarks de retenção padrão pontuam a relevância para objetivos em relação à **pergunta de avaliação retida** — um oráculo que espia a consulta futura. Isso satura em ~0,98 e mede recuperação, não esquecimento. Avaliamos no **regime cego**: a política de consolidação nunca vê a pergunta de avaliação, reproduzindo o funcionamento de sistemas reais.

```
oracle regime (unfair):  goal-only → 0.979   ← measures retrieval
blind  regime (honest):  goal-only → 0.286   ← actual forgetting quality
```

---

## Eficácia — Benchmark LongMemEval

479 casos reais de conversas multi-sessão, fração de retenção κ = 0,30, regime cego:

<div align="center">

| Política | Retenção de evidências-ouro | vs. aprendida |
|----------|:---------------------------:|:-------------:|
| **Multifatorial aprendida (nossa)** | **0,770 ± 0,011** | — |
| Pesos uniformes | 0,657 | −0,113 |
| Melhor fator único (relevância para o self) | 0,518 | −0,252 |
| Linha de base por recência | 0,368 | −0,402 |

*O IC de bootstrap de 95 % de cada diferença está estritamente acima de zero (20 divisões reamostradas 50/50).*

</div>

**Os pesos aprendidos são interpretáveis** — confiabilidade (0,64), intensidade emocional (0,55) e relevância para o self/usuário (0,23) dominam; a similaridade de objetivos no momento da consulta é corretamente desponderada (0,00) por não estar disponível no momento da consolidação.

**Um MLP neural sobre os mesmos fatores empata com o modelo linear (+0,003 ± 0,013)** — confirmando que os fatores se combinam de forma quase aditiva e que o valor linear interpretável não é um compromisso.

---

## Estrutura do Repositório

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

## Instalação

```bash
git clone https://github.com/zhibao-dev/Learning-Multi-Factor-Memory.git
cd Learning-Multi-Factor-Memory

# Full dev install (experiments + audit + tests)
pip install -e ".[dev]"

# Audit CLI only
pip install -e ".[audit]"
```

> **GPU não é necessário.** Os embeddings utilizam um sentence-transformer local (baixado uma vez para o cache do HuggingFace). Todos os experimentos rodam em CPU em questão de minutos.

---

## Reproduzir os Resultados do Artigo

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

Todos os resultados estão armazenados em cache em `results/` — os scripts podem ser reexecutados sem os dados brutos do LongMemEval.

---

## CLI de Auditoria de Memória

O `borge-audit` identifica inchaço, contradições, duplicatas e entradas desatualizadas em qualquer armazenamento de memória agêntica. Funciona inteiramente de forma local — nenhum dado de memória sai da sua máquina. **Somente leitura e modo dry-run: nada é excluído.**

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

Saída: `memories.audit.md` (relatório legível por humanos) + `memories.audit.forget.json` (lista de esquecimento reversível).

---

## Usar a Função de Valor Diretamente

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

## Citação

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

## Licença

MIT © 2026 Zhibao Chen, Qian Cheng
