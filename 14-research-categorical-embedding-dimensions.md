---
source: Multiple (see below)
author: Various
date: 2026-04-07
type: research
---

# Research: Categorical Embedding Dimension Rules, High Cardinality, and Rare Category Strategies

## Research Objective

Understand optimal embedding strategies for categorical variables in event data: dimension sizing rules, high-cardinality handling (hash embeddings, compositional embeddings), rare/low-frequency category treatment (noise injection, frequency-based regularization), and pre-trained categorical embeddings.

---

## Source 1: Guo & Berkhahn — Entity Embeddings of Categorical Variables (2016)

**URL**: https://arxiv.org/abs/1604.06737
**Code**: https://github.com/entron/entity-embedding-rossmann

The foundational paper on entity embeddings for categorical variables. Applied to Rossmann Store Sales Kaggle competition (3rd place).

### Embedding Dimensions Used

Table 1 from the paper — exact dimensions chosen:

| Feature | Cardinality | Embedding Dimension | Ratio (dim/card) |
|---------|-------------|--------------------|----|
| Store | 1,115 | 10 | 0.009 |
| Day of week | 7 | 6 | 0.86 |
| Day | 31 | 10 | 0.32 |
| Month | 12 | 6 | 0.50 |
| Year | 3 | 2 | 0.67 |
| Promotion | 2 | 1 | 0.50 |
| State | 12 | 6 | 0.50 |

### Dimension Sizing Rule

The authors did NOT provide a formula. Their approach:
- "The more complex the more dimensions. We roughly estimated how many features/aspects one might need to describe the entities."
- Fallback: between 1 and m_i - 1 (where m_i is number of categories)
- This was manual/intuitive selection, not algorithmic

### Architecture
- Two fully connected layers: 1,000 and 500 neurons
- ReLU activation, sigmoid output
- Adam optimizer, 10 epochs, ensemble of 5 networks
- No dropout used (did not help)
- No embedding-specific regularization mentioned

### Key Results
- Entity embeddings matched one-hot on shuffled data (0.070 MAPE)
- Outperformed one-hot on temporal split (0.093 vs 0.101 MAPE)
- Major finding: embeddings learned meaningful geographic relationships (German states mapped close to actual neighbors)
- Embeddings transfer well as features to other models (XGBoost, random forest)

### Rare Categories
Not addressed in the paper. All store IDs had sufficient data in the Rossmann dataset.

---

## Source 2: FastAI Embedding Dimension Rules (Howard)

**URL**: https://docs.fast.ai/tabular.model.html
**Forum**: https://forums.fast.ai/t/embedding-layer-size-rule/50691

### Original Rule (2018)

```python
def emb_sz_rule(n_cat: int) -> int:
    return min(50, (n_cat // 2) + 1)
```

### Current Rule (2019+)

```python
def emb_sz_rule(n_cat: int) -> int:
    return min(600, round(1.6 * n_cat ** 0.56))
```

### Origin

Jeremy Howard: "I just fitted a line to the empirical values in Excel. There's no theory behind it." The 1.6 coefficient and 0.56 exponent came from fitting empirical results from multiple Kaggle competitions.

### Example Outputs

| Cardinality | Old Rule min(50, n/2+1) | New Rule min(600, 1.6*n^0.56) |
|-------------|------------------------|-------------------------------|
| 2 | 2 | 2 |
| 7 | 4 | 5 |
| 12 | 7 | 6 |
| 31 | 16 | 10 |
| 100 | 50 | 21 |
| 1,000 | 50 | 73 |
| 10,000 | 50 | 266 |
| 100,000 | 50 | 600 |
| 1,000,000 | 50 | 600 |

### Cap at 600

Likely correlates with Word2Vec research showing optimal embedding dimensions between 100-1,000.

---

## Source 3: Google/TensorFlow 4th Root Rule

**URL**: https://forums.fast.ai/t/embedding-layer-size-rule/50691 (discussed here)

### Formula

```python
embedding_dim = int(n_categories ** 0.25)
```

Google developers recommend the 4th root of the number of categories.

### Example Outputs

| Cardinality | 4th Root |
|-------------|----------|
| 7 | 1 |
| 12 | 1 |
| 81 | 3 |
| 1,000 | 5 |
| 10,000 | 10 |
| 100,000 | 17 |
| 1,000,000 | 31 |

### Criticism

Jeremy Howard found this produces dimensions "way too low" for practical use. Most practitioners agree this is too aggressive a compression.

---

## Source 4: Common Kaggle Heuristic (min(50, n/2))

**URL**: https://yashuseth.wordpress.com/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/

### Rule

```python
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
```

### PyTorch Implementation

```python
class FeedForwardNN(nn.Module):
    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes,
                 output_size, emb_dropout, lin_layer_dropouts):
        super().__init__()
        
        # Embedding layers
        self.emb_layers = nn.ModuleList([
            nn.Embedding(x, y) for x, y in emb_dims
        ])
        
        # Embedding dropout
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        
        # Batch normalization for continuous features
        self.first_bn_layer = nn.BatchNorm1d(no_of_cont)
        
        # Linear layers
        first_lin_layer_size = sum(y for x, y in emb_dims) + no_of_cont
        self.lin_layers = nn.ModuleList([
            nn.Linear(first_lin_layer_size if i == 0 else lin_layer_sizes[i-1],
                      lin_layer_sizes[i])
            for i in range(len(lin_layer_sizes))
        ])
        
    def forward(self, cont_data, cat_data):
        x = [emb_layer(cat_data[:, i]) 
             for i, emb_layer in enumerate(self.emb_layers)]
        x = torch.cat(x, 1)
        x = self.emb_dropout_layer(x)
        # ... continue with linear layers
```

---

## Source 5: PyTorch-Lifestream NoisyEmbedding

**URL**: https://github.com/dllllb/pytorch-lifestream
**IJCAI Paper**: https://www.ijcai.org/proceedings/2025/1272.pdf

### NoisyEmbedding Class — Full Implementation

```python
class NoisyEmbedding(nn.Embedding):
    """
    Embeddings with additive gaussian noise with mean=0 and user-defined variance.
    
    Args:
        noise_scale (float): when > 0, applies additive noise to embeddings.
            When = 0, forward is equivalent to usual embeddings.
        dropout (float): probability of embedding axis to be dropped.
        spatial_dropout (bool): whether to dropout full dimension of embedding
            in the whole sequence.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, noise_scale=0, dropout=0,
                 spatial_dropout=False):
        if max_norm is not None:
            raise AttributeError("Please don't use embedding normalisation.")
        
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight)
        self.noise = torch.distributions.Normal(0, noise_scale) if noise_scale > 0 else None
        self.scale = noise_scale
        self.spatial_dropout = spatial_dropout
        self.dropout = nn.Dropout2d(dropout) if spatial_dropout else nn.Dropout(dropout)
        
        # Warm up embedding — ensure all rows are initialized
        _ = super().forward(torch.arange(num_embeddings))

    def forward(self, x):
        x = super().forward(x)
        if self.spatial_dropout:
            x = self.dropout(x.permute(0, 2, 1).unsqueeze(3)).squeeze(3).permute(0, 2, 1)
        else:
            x = self.dropout(x)
        if self.training and self.scale > 0:
            x += self.noise.sample((self.weight.shape[1], )).to(self.weight.device)
        return x
```

### Usage in TrxEncoder

```python
noisy_embeddings[emb_name] = NoisyEmbedding(
    num_embeddings=emb_props['in'],
    embedding_dim=emb_props['out'],
    padding_idx=0,
    max_norm=1 if norm_embeddings else None,
    noise_scale=embeddings_noise,  # e.g., 0.003
    dropout=emb_dropout,
    spatial_dropout=spatial_dropout,
)
```

### How It Works

1. Standard embedding lookup: `x = super().forward(x)` — normal nn.Embedding
2. Dropout: either element-wise or spatial (full dimension across sequence)
3. Noise injection (training only): adds Gaussian noise N(0, noise_scale) to all embedding dimensions
4. The noise is sampled ONCE per forward pass (same noise for all tokens in the batch), NOT per-token

### Why Noise Helps Rare Categories

- Acts as implicit regularization — prevents embedding rows from overfitting to their sparse training signal
- Equivalent to adding L2 regularization to the embedding weights (Bishop 1995 — training with noise is equivalent to Tikhonov regularization)
- Rare categories benefit disproportionately: their embedding vectors have more variance from fewer updates, so the added noise acts as a smoothing prior
- Typical noise_scale values: 0.003 to 0.01

---

## Source 6: Compositional Embeddings / Quotient-Remainder Trick (Meta 2019)

**URL**: https://arxiv.org/abs/1909.02107
**Blog**: https://www.shaped.ai/blog/beyond-the-hashing-trick-the-math-of-scaling-to-100m-ids-in-production

### The Problem

Standard embedding table: V categories x D dimensions = V*D parameters.
At 100M categories with 128-dim embeddings → ~50GB VRAM (+ 150GB with Adam optimizer states).

### Quotient-Remainder Trick

Split one large table into two smaller tables:

```python
# Instead of: embedding_table[V, D]
# Use two tables:
table_quotient  = nn.Embedding(B, D)  # B << V
table_remainder = nn.Embedding(B, D)  # B << V

def lookup(category_id):
    q = category_id // B  # quotient
    r = category_id % B   # remainder
    return table_quotient(q) + table_remainder(r)  # or multiply
```

Memory: from O(V * D) to O(2 * B * D) where B = sqrt(V).
Effective address space: B * B = V unique combinations.

### Collision Analysis

- Single hash table (20K rows, 100M IDs): ~5,000 IDs per bucket
- QR factorization (two 10K tables): 100M effective combinations → near-zero collision
- With k hash functions: collision probability = (1 - e^(-N^2/2B))^k

### Production Results (Shaped)

- Reduced VRAM usage by 85%
- Improved recall on tail items by eliminating collision noise
- Key finding: "eliminating collision noise for tail items actually increased recall despite reduced total parameters"

### Tiered Strategy for Production

| Scale | Approach |
|-------|----------|
| < 100K IDs | Standard embedding table |
| Tree models | Target encodings + behavioral aggregations |
| Deep neural | Stateless multi-hash + side information |
| Matrix factorization | Higher-resolution multi-hash |
| Sequential (Transformers) | Full tokenization or semantic IDs |

### Deep Hash Embedding (DHE)

```python
# For each category_id:
hash_vector = [hash_fn_k(category_id) for k in range(num_hashes)]  # e.g., 1024
embedding = DNN(hash_vector)  # MLP: 1024 -> 256 -> D
```

Any category_id (even unseen) gets a deterministic embedding. No cold-start problem.
Trade-off: compute-heavy but memory-efficient.

---

## Source 7: TabTransformer — Pre-trained Contextual Categorical Embeddings (Amazon 2020)

**URL**: https://arxiv.org/abs/2012.06678

### Architecture

1. Each categorical feature gets an embedding (standard lookup)
2. All categorical embeddings pass through a stack of Transformer layers
3. Output: contextual embeddings where each category's representation is informed by all other categories in the same row
4. Continuous features bypass the Transformer (separate pathway)
5. Both pathways merge before final prediction layers

### Unsupervised Pre-training Methods

**Masked Language Modeling (MLM)**:
- Randomly mask a fraction of categorical features in each row
- Train the Transformer to reconstruct masked features from remaining features
- Same principle as BERT, applied to tabular categorical columns

**Replaced Token Detection (RTD)**:
- Replace random categorical features with random values from the same column
- Train binary classifier to detect which features were replaced
- Same principle as ELECTRA, applied to tabular data

### Semi-Supervised Results

- Pre-trained TabTransformer-RTD improved AUC by 1.2-2.1% over competitors at low label counts (50-500 labeled samples)
- Contextual embeddings (post-Transformer) show meaningful semantic clustering in t-SNE (related categories cluster together)
- Non-contextual embeddings (pre-Transformer) show no such structure

### Key Insight for Event Data

Pre-training categorical embeddings from co-occurrence patterns (which categories appear in the same entity) creates representations where semantically similar categories are close in embedding space — BEFORE any supervised training begins.

---

## Source 8: Word2Vec / Item2Vec for Categorical Pre-training

**URL**: https://www.taboola.com/engineering/using-word2vec-for-better-embeddings-of-categorical-features/

### Category2Vec Approach (Taboola)

Adapt Word2Vec to categorical features:
- Each "sentence" = set of categories co-occurring for one entity/user
- E.g., for a user: [inquiry, property_assessment, inquiry, home_value] is a "sentence"
- Train Skip-gram or CBOW to predict co-occurring categories

### Results

- Pre-trained embeddings substantially outperformed random-initialized embedding layers
- Visualization: Category2Vec produces meaningful clusters (e.g., by language, by domain)
- Random initialization produced "messy" results

### When to Use

- High-cardinality categorical features where co-occurrence carries signal
- Transfer learning between tasks
- When domain-meaningful relationships exist between category values

### Application to Event Data

For public records events:
- Construct "sentences" from each entity's event history: [inquiry, inquiry, property_assessment, home_value, new_account]
- Train Word2Vec on these sentences
- Initialize embedding table with pre-trained vectors
- Fine-tune during supervised training

This gives rare categories (e.g., "student_file") a meaningful starting position based on which other event types they co-occur with, rather than random initialization.

---

## Summary of Dimension Sizing Rules

| Source | Formula | Cap | Origin |
|--------|---------|-----|--------|
| Guo & Berkhahn (2016) | Manual estimation | m_i - 1 | Domain intuition |
| Google/TensorFlow | n^0.25 (4th root) | None | Google guidance |
| FastAI old (2018) | min(50, n//2 + 1) | 50 | Empirical |
| FastAI new (2019+) | min(600, round(1.6 * n^0.56)) | 600 | Jeremy Howard Excel fit |
| Kaggle common | min(50, (n+1)//2) | 50 | Community standard |
| PyTorch-Lifestream | User-specified per feature | None | Config-driven |
| Production RecSys | 64-256 fixed | 256 | Hardware constraints |

## Summary of Rare Category Strategies

| Strategy | Source | Mechanism | Overhead |
|----------|--------|-----------|----------|
| NoisyEmbedding | PyTorch-Lifestream | Additive Gaussian noise during training | Minimal |
| AdamAR | ICLR 2026, Alibaba | Frequency-adaptive weight decay | +22% compute |
| FAL | Pinterest 2025 | Frequency-adaptive learning rate | Minimal |
| DFE two-step | Deep Feature Embedding 2024 | Small ID → shared DNN → full embedding | Small DNN |
| SSE | NeurIPS 2019 | Stochastic embedding swapping | Minimal |
| QR compositional | Meta 2019 | Quotient-remainder factorization | 2 tables |
| Category2Vec pre-train | Taboola, others | Word2Vec on co-occurrence | Pre-training step |
| TabTransformer MLM/RTD | Amazon 2020 | Masked/replaced token pre-training | Pre-training step |
