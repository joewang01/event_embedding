---
source: Multiple (5+ sources)
author: Various (Gorishniy et al., Yandex Research, community tutorials)
date: 2026-04-07
type: research
---

# Research: Numerical Feature Embeddings for Tabular Deep Learning

## Research Query
"numerical feature embedding piecewise linear encoding periodic activation quantile binning deep learning tabular"

Context: Encoding per-event numerical features (dollar amounts, property assessment values, home median values, counts, scores) into embeddings. Current approach is BatchNorm + Linear projection. Need to research whether more sophisticated numerical encoding helps.

---

## Source 1: Gorishniy et al. "On Embeddings for Numerical Features in Tabular Deep Learning" (NeurIPS 2022)

**URL**: https://arxiv.org/abs/2203.05556
**Official Code**: https://github.com/yandex-research/rtdl-num-embeddings
**PyPI**: `pip install rtdl_num_embeddings`

### Key Thesis
Embeddings for numerical features are an underexplored degree of freedom in tabular deep learning. Two novel embedding approaches — piecewise linear encoding (PLE) and periodic activations — can lead to significant performance boosts compared to conventional linear layers and ReLU activations.

### Mathematical Formulations

#### Piecewise Linear Encoding (PLE)

For a numerical feature x with T bins defined by edges b_0, b_1, ..., b_T:

```
PLE(x) = [e_1, ..., e_T] in R^T

where e_t = {
  0,                            if x < b_{t-1} AND t > 1
  1,                            if x >= b_t AND t < T
  (x - b_{t-1})/(b_t - b_{t-1}),  otherwise (linear interpolation within bin)
}
```

Example (x=45, range [0,100], 4 bins with edges [0, 25, 50, 75, 100]):
- Bin 1 [0,25]: x=45 >= 25, so e_1 = 1
- Bin 2 [25,50]: x=45 is within, so e_2 = (45-25)/(50-25) = 0.8
- Bin 3 [50,75]: x=45 < 50, so e_3 = 0
- Bin 4 [75,100]: x=45 < 75, so e_4 = 0
- Result: PLE(45) = [1, 0.8, 0, 0]

This is a "soft one-hot" or thermometer-like encoding — values below the current bin are 1, above are 0, and the active bin gets a linear interpolation.

#### Periodic Embedding

```
f(x) = ReLU(Linear(concat[sin(v), cos(v)]))

where v = [2*pi*c_1*x, ..., 2*pi*c_k*x]
```

The c_i coefficients are trainable, initialized from N(0, sigma). sigma is a critical hyperparameter that requires validation set tuning.

### Notation System

| Name | Composition | Description |
|------|-------------|-------------|
| L | Linear | Simple linear projection |
| LR | ReLU(Linear) | Linear + ReLU |
| Q | PLE_quantile | Quantile-based piecewise linear encoding |
| Q-L | Linear(PLE_quantile) | PLE with quantile bins + linear |
| Q-LR | ReLU(Linear(PLE_quantile)) | PLE with quantile bins + linear + ReLU |
| T | PLE_tree | Target-aware (tree-based) piecewise linear encoding |
| T-L | Linear(PLE_tree) | PLE with tree bins + linear |
| T-LR | ReLU(Linear(PLE_tree)) | PLE with tree bins + linear + ReLU |
| P | Periodic | Periodic activation only |
| PL | Linear(Periodic) | Periodic + linear |
| PLR | ReLU(Linear(Periodic)) | Periodic + linear + ReLU |

### Bin Computation Strategies

**Quantile-Based (Unsupervised)**:
Bins determined by uniformly chosen empirical quantiles of individual feature distributions:
b_t = Q_{t/T}({x_ij for j in J_train})

**Target-Aware (Supervised)**:
Uses C4.5-style discretization: builds a decision tree using only one feature and the target, then treats leaf boundaries as bin edges. Tuned parameters: max_leaves, min_samples_leaf, min_impurity_decrease.

### Benchmark Results (Table 6 — Average Ranks, 11 datasets)

| Method | Avg Rank (std) |
|--------|---------------|
| MLP (baseline) | 8.5 (2.6) |
| MLP-PLR | **3.0 (2.4)** |
| MLP-Q-LR | 5.1 (1.7) |
| ResNet (baseline) | 6.7 (3.3) |
| ResNet-PLR | **3.2 (1.3)** |
| Transformer-L (baseline) | 5.9 (2.2) |
| Transformer-PLR | 3.9 (2.5) |
| Transformer-T-LR | **3.7 (2.2)** |
| CatBoost | 3.6 (2.9) |
| XGBoost | 4.6 (2.7) |

### Key Findings

1. **PLR (periodic) is the overall best encoding** — MLP-PLR achieves rank 3.0, beating CatBoost (3.6)
2. **PLE (piecewise linear) is simpler and nearly as good** — Q-LR at 5.1, T-LR at 3.7 for Transformer
3. **Embeddings matter more than backbone** — After proper embeddings, MLP matches Transformer
4. **All backbones benefit** — Not just Transformers; MLP, ResNet all improve dramatically
5. **Initialization sigma is critical** for periodic embeddings — must tune on validation set
6. **Adding ReLU after the linear layer consistently helps** (LR > L)

---

## Source 2: rtdl-num-embeddings Python Library

**URL**: https://github.com/yandex-research/rtdl-num-embeddings
**PyPI**: `pip install rtdl_num_embeddings`

### Available Modules

1. **LinearEmbeddings** — Simple linear projection per feature
2. **LinearReLUEmbeddings** — ReLU(Linear(x)) per feature
3. **PiecewiseLinearEncoding** — Fixed (non-trainable) PLE transformation
4. **PiecewiseLinearEmbeddings** — PLE + trainable linear embedding per bin
5. **PeriodicEmbeddings** — ReLU(Linear(CosSin(2*pi*Linear(x))))

### Code Examples

```python
from rtdl_num_embeddings import (
    PiecewiseLinearEncoding, PiecewiseLinearEmbeddings,
    PeriodicEmbeddings, compute_bins
)
import torch
import torch.nn as nn

# --- Compute bin edges from training data ---
X_train = torch.randn(10000, n_features)  # training numerical features

# Quantile-based bins (unsupervised)
bins = compute_bins(X_train)

# Target-aware tree-based bins (supervised)
bins = compute_bins(
    X_train,
    tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
    y=Y_train,
    regression=True,
)

# --- Option A: PiecewiseLinearEncoding (fixed, non-trainable) ---
total_n_bins = sum(len(b) - 1 for b in bins)
model_a = nn.Sequential(
    PiecewiseLinearEncoding(bins),
    MLP(d_in=total_n_bins, ...)
)

# --- Option B: PiecewiseLinearEmbeddings (trainable) ---
d_embedding = 8  # or 12
model_b = nn.Sequential(
    PiecewiseLinearEmbeddings(bins, d_embedding, activation=False, version='B'),
    nn.Flatten(),
    MLP(d_in=n_features * d_embedding, ...)
)

# --- Option C: PeriodicEmbeddings ---
d_embedding = 24
model_c = nn.Sequential(
    PeriodicEmbeddings(n_features, d_embedding, lite=False),
    nn.Flatten(),
    MLP(d_in=n_features * d_embedding, ...)
)
```

### Complete Working Example (California Housing)

From the official example notebook:

```python
import sklearn.preprocessing
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

# Step 1: Quantile transform preprocessing (IMPORTANT!)
noise = np.random.default_rng(0).normal(0.0, 1e-5, X_cont_train.shape).astype(np.float32)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
).fit(X_cont_train + noise)

# Apply to all splits
for part in data:
    data[part]['x_cont'] = preprocessing.transform(data[part]['x_cont'])

# Step 2: Compute bins from PREPROCESSED training data
bins = compute_bins(data['train']['x_cont'])

# Step 3: Create embeddings
d_embedding = 8
cont_embeddings = PiecewiseLinearEmbeddings(
    bins, d_embedding, activation=False, version='B'
)

# Step 4: Use in model
class Model(nn.Module):
    def __init__(self, n_features, bins, d_embedding, mlp_kwargs):
        super().__init__()
        self.cont_embeddings = PiecewiseLinearEmbeddings(
            bins, d_embedding, activation=False, version='B'
        )
        d_num = n_features * d_embedding
        self.backbone = MLP(d_in=d_num, **mlp_kwargs)

    def forward(self, x_cont):
        x = self.cont_embeddings(x_cont).flatten(1)
        return self.backbone(x)
```

### Key Implementation Detail
The official example applies **QuantileTransformer BEFORE computing bins**. This is the recommended pipeline:
1. QuantileTransformer(output_distribution='normal') on raw features
2. compute_bins() on the quantile-transformed features
3. PiecewiseLinearEmbeddings on the quantile-transformed features

---

## Source 3: TabM (Gorishniy et al., ICLR 2025)

**URL**: https://arxiv.org/abs/2410.24210
**Code**: https://github.com/yandex-research/tabm

### Key Finding on Numerical Embeddings
TabM (parameter-efficient MLP ensemble) explicitly recommends PiecewiseLinearEmbeddings as the default numerical encoding:

> "By default, we recommend using the piecewise-linear embeddings (Gorishniy et al., 2022)."

TabM uses:
- PiecewiseLinearEmbeddings (marked as "dagger" variant, TabM†)
- PeriodicEmbeddings as alternative (marked as "double-dagger" variant, TabM‡)

This confirms PLE as the production-ready default even in 2025 SOTA tabular models.

---

## Source 4: Yandex Research Blog — Embeddings for Numerical Features

**URL**: https://research.yandex.com/blog/embeddings-for-numerical-features-in-tabular-deep-learning

### Benchmark Summary (from blog)

| Backbone | Embedding | Avg Rank (std) |
|----------|-----------|----------------|
| MLP (baseline) | None | 8.5 (2.6) |
| MLP | PLR | **3.0 (2.4)** |
| MLP | PLE (Q-LR) | 5.1 (1.7) |
| Transformer | PLR | 3.9 (2.5) |
| CatBoost | — | 3.6 (2.9) |

### Practical Recommendations from Blog
1. Periodic embeddings (PLR) outperform piecewise linear substantially
2. Careful frequency initialization is critical for periodic scheme
3. Adding ReLU after embedding consistently improves performance
4. Simple MLPs with embeddings match heavy Transformer architectures

---

## Source 5: Mambular Tutorial — FT-Transformer + PLE

**URL**: https://medium.com/tabular-deep-learning/mambular-tabular-deep-learning-series-2-ft-transformer-and-piecewise-linear-encodings-371cb54dd399

### Benchmark (California Housing MSE)

| Model | MSE |
|-------|-----|
| XGBoost | ~0.165 |
| MLP (standardization) | 0.190 |
| FT-Transformer (linear embeddings) | 0.217 |
| FT-Transformer (PLR embeddings) | 0.190 |
| **FT-Transformer (PLE, 64 bins)** | **0.173** |

### Mambular Library Code
```python
from mambular.models import FTTransformerRegressor

model = FTTransformerRegressor(
    numerical_preprocessing="ple",
    n_bins=64,
    d_model=64,
)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val, max_epochs=200, lr=1e-03)
```

---

## Source 6: Normalization Strategy — QuantileTransformer

### Gorishniy et al. Recommended Pipeline
The rtdl codebase consistently uses sklearn's QuantileTransformer as the preprocessing step BEFORE any embedding:

```python
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
)
```

### Why QuantileTransformer > BatchNorm for Tabular DL
1. **Handles skew**: Dollar amounts, counts are typically log-normal. QuantileTransformer maps them to Gaussian without needing to know the distribution family.
2. **Outlier robustness**: Compresses extreme values more than standard scaling.
3. **Non-parametric**: Doesn't assume any distribution shape.
4. **Deterministic at inference**: Unlike BatchNorm which depends on running statistics and batch composition.
5. **Works per-feature**: Each feature gets its own quantile mapping.

### When to Use Log Transform vs QuantileTransformer
- **Log transform**: When you KNOW the feature is log-normal (dollar amounts, property values). Simple, interpretable, preserves multiplicative relationships.
- **QuantileTransformer**: When distribution is unknown or complex. More robust but loses interpretability.
- **Recommendation**: QuantileTransformer is the safe default. Log transform is a good addition for known-skewed features (can compose: log1p → QuantileTransformer).

### BatchNorm Limitations for Tabular Data
1. Batch-dependent: Statistics change across batches, unstable for small batches.
2. Poor with skew: BatchNorm normalizes to zero-mean unit-variance, but doesn't fix heavy tails.
3. Leaks information: Batch statistics in eval mode depend on entire training set distribution.
4. Not feature-specific: Applies same normalization logic to all features regardless of distribution.

---

## Source 7: PyTorch-Lifestream Numerical Handling

**URL**: https://github.com/pytorch-lifestream/pytorch-lifestream

PyTorch-Lifestream's TrxEncoder handles numerical features with:
- `identity`: Pass-through (no transform)
- `log`: Log transform
- Standard normalization options

As of 2025, ptls does NOT implement PLE or periodic embeddings for numerical features. It uses the simpler BatchNorm + identity/log approach. The Gorishniy embeddings would be a direct upgrade path.

---

## Synthesis: Recommendations for Event Embedding System

### Current State
The EventEncoder uses: `BatchNorm → Linear(n_num, 32)` — all numerical features are projected to a shared 32-dim space.

### Upgrade Path (by priority)

#### Priority 1: Replace BatchNorm with QuantileTransformer (preprocessing)
- Apply sklearn QuantileTransformer to training data before model sees it
- Use output_distribution='normal'
- Handles dollar amounts, counts, scores automatically
- Zero model changes needed

#### Priority 2: Add log1p for known-skewed features
- Property assessment values, home median values, dollar amounts → log1p before QuantileTransformer
- Counts → log1p is standard

#### Priority 3: Replace Linear projection with PiecewiseLinearEmbeddings
- Per-feature embedding: each numerical feature gets its own d-dimensional representation
- Use compute_bins() on quantile-transformed training data
- d_embedding=8-12 is typical
- This gives each feature its own bin-based representation instead of a shared linear projection

#### Priority 4 (Optional): Try PeriodicEmbeddings
- If PLE doesn't give enough lift, try PLR (periodic + linear + ReLU)
- Requires tuning sigma (frequency initialization scale)
- More parameters but potentially better for complex feature distributions

### Important: Interaction with Sequence Encoder

The Gorishniy benchmarks are on TABULAR data (one row = one prediction). For EVENT SEQUENCES:
- The numerical encoding happens per-event, then feeds into a GRU/Transformer sequence encoder
- The sequence encoder adds another layer of representation learning
- The per-event encoding improvement may be smaller when a powerful sequence encoder follows
- But it still helps: better per-event representations give the sequence encoder better inputs

### Recommended Configuration

```python
# BEFORE training: preprocess with QuantileTransformer
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
X_num_train = qt.fit_transform(X_num_train_raw)

# IN model: use PiecewiseLinearEmbeddings
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

bins = compute_bins(torch.tensor(X_num_train))
d_embedding = 8

class EventEncoder(nn.Module):
    def __init__(self, cat_features, num_features, bins, d_embedding=8):
        super().__init__()
        self.cat_embeddings = ...  # unchanged
        self.num_embeddings = PiecewiseLinearEmbeddings(
            bins, d_embedding, activation=False, version='B'
        )
        n_num = len(num_features)
        cat_dim = sum(ed for _, (_, ed) in cat_features.items())
        num_dim = n_num * d_embedding
        self.output_dim = cat_dim + num_dim + time_dim

    def forward(self, batch):
        parts = []
        # Categorical
        for name, emb in self.cat_embeddings.items():
            parts.append(emb(batch[name]))
        # Numerical — PLE per feature
        nums = torch.stack([batch[n] for n in self.num_features], dim=-1)
        B, T, F = nums.shape
        num_emb = self.num_embeddings(nums.reshape(B*T, F))  # [B*T, F, d]
        num_emb = num_emb.reshape(B, T, -1)                  # [B, T, F*d]
        parts.append(num_emb)
        # Time encoding (unchanged)
        ...
        return torch.cat(parts, dim=-1)
```
