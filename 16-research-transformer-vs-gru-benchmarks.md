---
title: "Research: Transformer vs GRU Empirical Benchmarks on Event Sequences"
source: multiple
author: various
created: 2026-04-08
type: research-compilation
tags: [event-embedding, transformer, gru, benchmark]
---

## Sources
| # | URL | Type | Author | Key Finding |
|---|-----|------|--------|-------------|
| 1 | https://arxiv.org/html/2410.03399v1 | Paper (KDD 2025) | On-Point-RND (Sberbank AI) | GRU-based methods occupy top 3 spots across 7 event sequence datasets; Transformer ranks 5th |
| 2 | https://github.com/On-Point-RND/EBES | Code repo | On-Point-RND | 10 datasets, 9 models; GRU dominance confirmed with reproducible code |
| 3 | https://github.com/pytorch-lifestream/ptls-experiments | Code repo + results | Sberbank AI / pytorch-lifestream | coles_transformer wins on Age (0.646) and Scoring (0.7968); GRU wins on Gender (0.882) and Churn (0.841) — mixed results depending on dataset |
| 4 | https://arxiv.org/pdf/1808.09781 | Paper (ICDM 2018) | Kang & McAuley (UCSD) | SASRec (Transformer) is 11x faster than Caser, 17x faster than GRU4Rec+ on GPU; outperforms GRU4Rec on all recommendation benchmarks |
| 5 | https://arxiv.org/html/2406.03733v1 | Paper (2024) | Chang Yu et al. | Transformer achieves 0.998 F1 on credit card fraud (284K transactions), but comparison is vs ML baselines not RNNs |

## Key Findings

### 1. EBES Benchmark (KDD 2025): GRU dominates event sequences overall
The largest systematic benchmark for event sequences (7 real-world + 1 synthetic dataset, 9 models). **Overall ranking:**

| Rank | Model | Architecture |
|------|-------|-------------|
| 1 | CoLES | GRU + contrastive pretraining |
| 2 | GRU (vanilla) | GRU supervised |
| 3 | MLEM | GRU + hybrid contrastive/generative |
| 4 | Mamba | State-space model |
| 5 | Transformer | Self-attention |
| 6 | mTAND | Multi-time attention |
| 7 | MLP | Feedforward baseline |

### 2. EBES dataset-level numbers (AUC-ROC or metric per dataset)
GRU consistently edges out Transformer on every dataset:

| Dataset | Domain | Seq Length (mean±std) | GRU | Transformer | Delta |
|---------|--------|----------------------|-----|-------------|-------|
| Age | Banking/demographics | 881±125 | 0.626±0.004 | 0.621±0.006 | +0.005 GRU |
| MBD | Process mining | 21±435 | 0.827±0.001 | 0.821±0.002 | +0.006 GRU |
| MIMIC-III | Healthcare ICU | 58±93 | 0.901±0.002 | 0.894±0.002 | +0.007 GRU |
| Pendulum | Synthetic | 32±9 | 0.896±0.010 | 0.891±0.015 | +0.005 GRU |
| PhysioNet2012 | Healthcare | 75±23 | 0.846±0.004 | 0.838±0.008 | +0.008 GRU |
| Retail | E-commerce | 114±103 | 0.543±0.002 | 0.536±0.006 | +0.007 GRU |
| Taobao | E-commerce | 280±387 | 0.713±0.004 | 0.692±0.013 | +0.021 GRU |

**Key observation:** Differences are small (0.5-2.1 points) but consistent. The largest gap is Taobao (+2.1 points for GRU), a dataset with high variance in sequence length (280±387). Taobao also shows distribution shift between val and test sets.

### 3. CoLES with GRU vs Transformer encoder (PyTorch-Lifestream)
From ptls-experiments, unsupervised embeddings + LightGBM downstream:

| Dataset | Best GRU method | Score | Best Transformer method | Score | Winner |
|---------|----------------|-------|------------------------|-------|--------|
| Gender | mles2_embeddings (GRU) | 0.882 AUROC | — | — | GRU |
| Age group | — | 0.629 baseline | coles_transformer | 0.646 Acc | **Transformer** |
| Churn (Rosbank) | mles_embeddings (GRU) | 0.841 AUROC | — | — | GRU |
| Scoring (Alpha) | — | 0.779 baseline | coles_transformer | 0.797 AUROC | **Transformer** |

**Key observation:** coles_transformer wins on Age and Scoring datasets — both are classification tasks with many event types and longer sequences. GRU wins on Gender and Churn — simpler binary tasks. This suggests transformers may benefit from richer label spaces and more diverse event vocabularies.

### 4. SASRec vs GRU4Rec (Sequential Recommendation)
From Kang & McAuley (2018), the foundational comparison:

- SASRec outperforms GRU4Rec and GRU4Rec+ on **all benchmark datasets** (Amazon Beauty, Amazon Games, MovieLens-1M, Steam)
- SASRec is **11x faster** than Caser and **17x faster** than GRU4Rec+ during training on GPU (parallel self-attention vs sequential RNN)
- On **sparse datasets**, SASRec focuses on recent items (behaves like Markov chain)
- On **dense datasets**, SASRec captures long-range dependencies (exploits full attention)
- Correlation of ~0.9 between SASRec and GRU4Rec relative accuracy changes across datasets, suggesting they respond to similar data properties

**This is the one domain where Transformers clearly win over GRU for event sequences.** Sequential recommendation has: (a) large item vocabularies (thousands), (b) relatively short sequences (10-200 interactions), (c) strong long-range skip patterns (user returns to old preferences).

### 5. When does Transformer beat GRU? Conditions synthesis

| Condition | Favors Transformer | Favors GRU |
|-----------|-------------------|------------|
| Sequence length | Very long (>500) with skip patterns | Short to medium, or irregular timing |
| Event vocabulary size | Large (1000s of event types) | Smaller vocabularies |
| Task type | Next-item prediction, multi-class | Binary classification, regression |
| Temporal regularity | Regular intervals | Irregular, variable inter-event gaps |
| Training data size | Large datasets (parallelism advantage) | Smaller datasets (fewer params) |
| Training speed on GPU | Much faster (parallel attention) | Sequential, slower on GPU |
| Inference latency | Higher (full sequence attention) | Lower (incremental hidden state) |
| Pretraining strategy | Benefits from contrastive learning | Benefits more from contrastive learning (CoLES) |
| Robustness to shuffled order | More robust (position embeddings) | Degrades more with shuffled input |
| Distribution shift | More sensitive | More robust |

### 6. The paradox: Transformers are faster to train but less accurate on event data
EBES shows GRU wins on accuracy. SASRec shows Transformer wins on training speed (17x). This creates a practical tradeoff:
- **If compute budget is the constraint**: Transformer (faster iteration, GPU parallelism)
- **If accuracy on financial/medical event data is the constraint**: GRU + CoLES pretraining
- **If sequential recommendation**: Transformer (both faster AND more accurate)

### 7. Credit card fraud detection
The transformer-based model achieved 0.998 F1 on the European credit card dataset (284,804 transactions, 492 fraudulent). However, this comparison was against ML baselines (SVM, KNN, Decision Tree), not LSTM/GRU. No rigorous transformer-vs-RNN fraud benchmark was found with matched experimental conditions.

## Suggested Wiki Pages
- [[Transformer vs GRU for Event Sequences]] — main comparison page with the EBES table and conditions matrix
- [[EBES Benchmark]] — dedicated page for the KDD 2025 benchmark (datasets, methodology, full results)
- [[CoLES Encoder Architecture]] — update existing page with transformer encoder variant results
- [[SASRec vs GRU4Rec]] — sequential recommendation specific comparison
- [[When to Choose Transformer vs RNN]] — practical decision guide with the conditions table

## Code Snippets

### EBES benchmark reproduction
```bash
# Install EBES
pip install ebes

# Run GRU on Age dataset
python main -d age -m gru -e correlation -s best

# Run Transformer on Age dataset
python main -d age -m transformer -e correlation -s best

# Run CoLES (top performer) on Age dataset
python main -d age -m coles -e correlation -s best
```

### PyTorch-Lifestream: CoLES with Transformer encoder
```python
import pytorch_lifestream as ptls

# CoLES with GRU encoder (default)
coles_gru = ptls.frames.coles.CoLESModule(
    seq_encoder=ptls.nn.RnnSeqEncoder(
        trx_encoder=trx_encoder,
        hidden_size=256,
        type='gru',
    ),
    head=ptls.nn.Head(use_norm_encoder=True),
)

# CoLES with Transformer encoder
coles_transformer = ptls.frames.coles.CoLESModule(
    seq_encoder=ptls.nn.TransformerSeqEncoder(
        trx_encoder=trx_encoder,
        n_heads=4,
        dim_hidden=256,
        n_layers=4,
    ),
    head=ptls.nn.Head(use_norm_encoder=True),
)
```

## Uncertain Claims

1. **SASRec 17x speed claim** — From the original 2018 paper. Hardware has changed significantly; modern GRU implementations with cuDNN may narrow this gap. The 17x figure should be treated as directional, not current. `confidence: medium`

2. **Taobao distribution shift explanation** — EBES notes distribution shift on Taobao but does not explicitly attribute the GRU advantage to this shift. The 2.1-point gap could be due to other dataset properties. `confidence: medium`

3. **Transformer wins on "large event vocabularies"** — Inferred from SASRec success (large item catalogs) vs EBES results (smaller event type counts). Not directly tested as an isolated variable. `confidence: low`

4. **CoLES+Transformer winning on Age and Scoring in ptls-experiments** — The results.md does not present GRU and Transformer side-by-side for these datasets. The "coles_transformer" entries appear as top performers but direct GRU CoLES numbers on those same datasets were not extracted. The comparison is against baselines, not explicit GRU CoLES. `confidence: low`

5. **Credit card fraud F1 of 0.998** — Suspiciously high. The dataset is extremely imbalanced (0.172% fraud rate) and the paper compares against weak baselines. This number should not be taken as representative of transformer capability on fraud detection in production. `confidence: low`
