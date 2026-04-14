---
source: Web research (arXiv, GitHub, blog posts, benchmarks)
author: Various (15+ sources)
date: 2026-04-07
type: research
---

# Research: Transformer Architectures for Tabular & Sequential Event Data

## Sources Consulted

### Primary Papers
1. **FT-Transformer** — Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021. arXiv:2106.11959. https://arxiv.org/abs/2106.11959
2. **TabTransformer** — Huang et al., "TabTransformer: Tabular Data Modeling Using Contextual Embeddings", 2020. arXiv:2012.06678. https://arxiv.org/abs/2012.06678
3. **SASRec** — Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018. arXiv:1808.09781. https://arxiv.org/abs/1808.09781
4. **BERT4Rec** — Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer", CIKM 2019. arXiv:1904.06690. https://arxiv.org/abs/1904.06690
5. **SASRec vs BERT4Rec** — Klenitskiy & Vasilev, "Turning Dross Into Gold Loss: Is BERT4Rec really better than SASRec?", RecSys 2023. arXiv:2309.07602. https://github.com/antklen/sasrec-bert4rec-recsys23
6. **HT-Transformer** — "HT-Transformer: Event Sequences Classification by Accumulating Prefix Information with History Tokens", 2025. arXiv:2508.01474. https://arxiv.org/html/2508.01474v1

### Benchmarks & Surveys
7. **Large-scale tabular benchmark** — 2024 comprehensive benchmark, 20 models across 111 datasets. arXiv:2408.14817. https://arxiv.org/html/2408.14817v1
8. **McElfresh et al.** — "When Do Neural Nets Outperform Boosted Trees on Tabular Data?", NeurIPS 2023. arXiv:2305.02997.
9. **Sebastian Raschka** — "A Short Chronology of Deep Learning for Tabular Data", 2022. https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html
10. **BERT4Rec Replicability** — "A Systematic Review and Replicability Study of BERT4Rec", RecSys 2022. arXiv:2207.07483.

### Implementations
11. **lucidrains/tab-transformer-pytorch** — PyTorch TabTransformer + FT-Transformer. https://github.com/lucidrains/tab-transformer-pytorch
12. **yandex-research/rtdl-revisiting-models** — Official FT-Transformer code (NeurIPS 2021). https://github.com/yandex-research/rtdl-revisiting-models
13. **pmixer/SASRec.pytorch** — PyTorch SASRec. https://github.com/pmixer/SASRec.pytorch
14. **FeiSun/BERT4Rec** — Original BERT4Rec TensorFlow. https://github.com/FeiSun/BERT4Rec
15. **RecBole SASRec** — Modular PyTorch SASRec in RecBole library. https://recbole.io/

---

## FT-Transformer (Feature Tokenizer + Transformer)

### Architecture
Two-stage design: Feature Tokenizer → Transformer.

**Feature Tokenizer** converts every feature into a d-dimensional token:
- Numerical feature x_j: T_j = W_j * x_j + b_j (where W_j ∈ ℝ^d, b_j ∈ ℝ^d — a linear projection per feature)
- Categorical feature x_j: T_j = e_j[x_j] + b_j (embedding lookup + bias)
- A learnable [CLS] token is prepended to the sequence of feature tokens

**Key difference from TabTransformer**: ALL features (numerical + categorical) become tokens and pass through the Transformer. TabTransformer only passes categorical features through the Transformer.

**Transformer backbone**: Standard multi-head self-attention + FFN. Typical config: depth=6, heads=8, dim=32-192, attn_dropout=0.1, ff_dropout=0.1.

**Output**: CLS token → linear projection → prediction.

### Performance
- Outperforms ResNet and other DL baselines on most tabular tasks
- Beats GBDT on 7/11 datasets in original paper
- BUT: 2024 benchmark (111 datasets, 20 models) found FT-Transformer achieved 0 "best" results; CatBoost won 19, LightGBM 15
- "No universally superior solution" — GBDT still dominates medium datasets
- FT-Transformer's advantage: more universal across dataset types than other DL models

### PyTorch Code (lucidrains)
```python
from tab_transformer_pytorch import FTTransformer

model = FTTransformer(
    categories=(10, 5, 6, 5, 8),  # cardinality per categorical feature
    num_continuous=10,
    dim=32,          # token embedding dimension
    dim_out=1,       # output dimension
    depth=6,         # transformer layers
    heads=8,         # attention heads
    attn_dropout=0.1,
    ff_dropout=0.1
)

x_categ = torch.randint(0, 5, (1, 5))   # [batch, num_cat_features]
x_numer = torch.randn(1, 10)            # [batch, num_cont_features]
pred = model(x_categ, x_numer)          # [batch, dim_out]
```

---

## TabTransformer

### Architecture
- Categorical features → column embedding + feature embedding → stacked Transformer layers → contextual embeddings
- Numerical features → BYPASS Transformer, concatenated with contextual cat embeddings after
- Combined → MLP → prediction

**Key insight**: Only categorical features benefit from inter-feature attention. Numerical features are appended directly.

### Semi-supervised Pre-training
Two procedures:
1. **MLM (Masked Language Model)**: Mask some categorical features, predict them from context
2. **RTD (Replaced Token Detection)**: Replace some embeddings with random ones, detect which were replaced

Pre-training improves performance by 2.1% mean AUC in semi-supervised settings.

### Performance
- Outperforms MLP baselines by ≥1.0% mean AUC across 15 datasets
- Matches tree-based ensembles (XGBoost, LightGBM) in supervised setting
- Robust to missing and noisy data
- Better interpretability than standard DL

### PyTorch Code (lucidrains)
```python
from tab_transformer_pytorch import TabTransformer

model = TabTransformer(
    categories=(10, 5, 6, 5, 8),
    num_continuous=10,
    dim=32,
    dim_out=1,
    depth=6,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    mlp_hidden_mults=(4, 2),
    mlp_act=nn.ReLU(),
)
```

---

## SASRec (Self-Attentive Sequential Recommendation)

### Architecture
Transformer decoder for sequential item prediction:

1. **Input**: item_embedding(x_i) + position_embedding(i) for i=1..n
2. **Self-attention**: Scaled dot-product with CAUSAL masking (can only attend to past items)
3. **Point-wise FFN**: Two-layer FFN with ReLU
4. **Residual connections + LayerNorm** at each sub-layer
5. **Stacking**: Typically 2 self-attention blocks
6. **Prediction**: Dot product of final hidden state with all item embeddings

### Key Design Choices
- **Causal masking**: Future items are masked out (like GPT, not like BERT)
- **Learned positional embeddings** (not sinusoidal)
- **Shared item embeddings** between embedding layer and prediction layer
- Hidden dim d=50, max seq length 200 (ML-1m) or 50 (sparse datasets)
- **Binary cross-entropy loss** with negative sampling (1 negative per positive in original)

### Performance (original paper)
- 6.9% Hit Rate and 9.6% NDCG improvement over strongest baseline (average)
- Outperforms GRU4Rec, GRU4Rec+, Caser, MC, FPMC
- Order of magnitude faster than CNN/RNN alternatives (parallelizable)

### RecSys 2023 Benchmark (with corrected loss)
MovieLens-1m:
- SASRec (original BCE): HR@10=0.2500, NDCG@10=0.1341
- BERT4Rec (CE loss): HR@10=0.2843, NDCG@10=0.1537
- **SASRec+ (CE loss)**: HR@10=0.3152, NDCG@10=0.1821 ← **BEST**

MovieLens-20m:
- SASRec+: HR@10=0.2983, NDCG@10=0.1833
- BERT4Rec: HR@10=0.2816, NDCG@10=0.1703

SASRec+ is 2.6× faster to train than BERT4Rec.

### PyTorch (RecBole)
```python
# Key components from RecBole SASRec
class SASRec(nn.Module):
    def __init__(self, n_items, hidden_size=64, n_layers=2, n_heads=2,
                 inner_size=256, hidden_dropout=0.5, attn_dropout=0.5,
                 max_seq_length=50, loss_type='CE'):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=n_layers, n_heads=n_heads,
            hidden_size=hidden_size, inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout,
            attn_dropout_prob=attn_dropout,
        )
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)
    
    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1))
        position_emb = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        # Causal attention mask
        attention_mask = get_attention_mask(item_seq, bidirectional=False)
        output = self.trm_encoder(input_emb, attention_mask)
        # Gather last item's representation for each sequence
        output = gather_indexes(output, item_seq_len - 1)
        return output  # [B, hidden_size]
```

---

## BERT4Rec (Bidirectional Encoder for Sequential Recommendation)

### Architecture
Transformer encoder (bidirectional) with masked item prediction:

1. **Input**: item_embedding(x_i) + position_embedding(i) for i=1..n
2. **Self-attention**: Standard bidirectional (NO causal mask — sees all positions)
3. **Feed-forward**: Two-layer FFN with GELU activation
4. **Residual + LayerNorm + Dropout** at each sub-layer
5. **Stacking**: 2 layers (default config)
6. **Prediction**: Softmax over all items at masked positions

### Cloze Task (Masked Item Prediction)
- Randomly mask proportion ρ of items (typically ρ=0.2 for training)
- Replace masked items with special [MASK] token
- Model predicts original item at masked positions
- Loss: cross-entropy over masked positions only
- Generates up to C(n,k) training samples per sequence (vs n for autoregressive)

### Inference
- Append [MASK] token at end of sequence: [item1, item2, ..., itemN, MASK]
- Model predicts probability distribution over all items for the MASK position
- Top-k items become recommendations

### Key Differences from SASRec
| Aspect | SASRec | BERT4Rec |
|--------|--------|----------|
| Attention | Causal (left-to-right) | Bidirectional |
| Training | Next-item prediction | Masked item prediction (Cloze) |
| Loss | BCE with negative sampling (original) | Cross-entropy over all items |
| Activation | ReLU | GELU |
| Inference | Natural (predict next) | Append [MASK] token |

### Default Hyperparameters (ML-1m)
- hidden_size: 64
- num_layers: 2
- num_heads: 2
- intermediate_size: 256
- max_seq_length: 200
- mask_prob: 0.2 (variable)
- dropout: 0.2
- learning_rate: 1e-4
- optimizer: Adam with warmup

### Key Finding (RecSys 2023)
**BERT4Rec's advantage was due to its loss function, not its architecture.** When SASRec is trained with the same cross-entropy loss (SASRec+), it significantly outperforms BERT4Rec on most datasets AND trains 2.6× faster. The bidirectional attention provides little practical benefit for sequential recommendation.

---

## HT-Transformer (Event Sequences with History Tokens)

### Architecture (2025)
- Processes heterogeneous event sequences with categorical + numerical features
- Categorical features: trainable embedding per unique value
- Numerical features: incorporated directly into event embeddings
- **History tokens**: Injected into transformer as information bottlenecks
- Sparse attention: events attend to history tokens and nearby events, not all events
- Time-based positional encoding from timestamp distributions

### Key Innovation
History tokens accumulate prefix information like RNN hidden states but within a transformer framework. Extracts fixed-size embeddings without CLS token or mean pooling.

### Performance
- Churn prediction: 83.76 ROC AUC (after fine-tuning)
- MIMIC-III: 92.97 ROC AUC
- Taobao: 87.29 ROC AUC
- State-of-the-art on 4/5 benchmarks

---

## Comprehensive Benchmark Summary

### Tabular Data (111 datasets, 2024)
| Model | # Best Results | Avg Rank |
|-------|---------------|----------|
| AutoGluon (ensemble) | 39 | — |
| CatBoost | 19 | — |
| LightGBM | 15 | — |
| AutoGluon-DL | 11 | — |
| ResNet | 10 | — |
| XGBoost | 5 | — |
| MLP | 6 | — |
| FT-Transformer | 0 | 10.7 |
| TabNet | 0 | 13.1 |

### Sequential Recommendation (RecSys 2023)
| Model | ML-1m HR@10 | ML-1m NDCG@10 | Train Time |
|-------|-------------|---------------|------------|
| SASRec+ (CE loss) | 0.3152 | 0.1821 | 540s |
| BERT4Rec | 0.2843 | 0.1537 | 1409s |
| SASRec (BCE loss) | 0.2500 | 0.1341 | — |

### Event Sequence Classification (EBES KDD 2025)
GRU outperforms Transformer across 10 event sequence datasets (already documented in wiki).

---

## Key Takeaways for Event Embedding Design

1. **For tabular data (static features)**: GBDT still wins on average. FT-Transformer is the best DL option if you need neural nets (e.g., for end-to-end training, embeddings, or multi-modal fusion).

2. **For sequential events**: SASRec-style causal transformer is preferred over BERT4Rec (faster, better with CE loss). But GRU baseline is hard to beat on event sequences specifically (EBES benchmark).

3. **Feature tokenization matters**: FT-Transformer's approach of tokenizing EVERY feature (numerical + categorical) into tokens is strictly more expressive than TabTransformer's categorical-only approach.

4. **Loss function matters more than architecture**: SASRec vs BERT4Rec debate was resolved — the loss function (CE vs BCE) explained most of the performance gap, not bidirectional vs unidirectional attention.

5. **For mixed tabular+sequential**: Consider a two-stage approach:
   - Event-level features: Use FT-Transformer-style tokenization (linear projection for numerical, embedding for categorical)
   - Sequence encoding: Use GRU (robust default) or causal Transformer (if parallelism needed)
   - This is essentially what the existing EventEncoder + SequenceEncoder pattern already does
