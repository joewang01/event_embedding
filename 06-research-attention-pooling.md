---
source: web research (multiple)
author: Various (8+ sources)
date: 2026-04-07
type: research
---

# Research: Attention Pooling for Embedding Extraction

## Sources Consulted

1. **Benjamin Warner — Tinkering With Attention Pooling** (2022)
   - URL: https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling
   - Type: Implementation-focused blog post
   - Key: Learned Aggregation (Touvron et al.) PyTorch code, performance comparison vs avg pooling

2. **Abhishek Selokar — Beyond CLS: Advanced Pooling Strategies for Vision Transformers** (Medium)
   - URL: https://medium.com/@imabhi1216/beyond-cls-advanced-pooling-strategies-for-vision-transformers-8df1785ec81c
   - Type: Tutorial with PyTorch code
   - Key: Six pooling strategies with code, complete DINOv2 wrapper

3. **Alberto Arrigoni — Paper Review & Code: Set Transformer** (Medium)
   - URL: https://medium.com/@albertoarrigoni/paper-review-code-set-transformer-b9750e5c3fdb
   - Type: Paper review with implementation
   - Key: PMA (Pooling by Multihead Attention) code, ISAB, full architecture

4. **Curt Tigges — The Annotated Perceiver** (Medium)
   - URL: https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb
   - Type: Annotated implementation
   - Key: Perceiver cross-attention pooling, latent array as learnable queries

5. **Zhan et al. — Pooling and Attention: What are Effective Designs for LLM-based Embedding Models?** (arXiv 2409.02727, 2024)
   - URL: https://arxiv.org/html/2409.02727v2
   - Type: Research paper
   - Key: Multi-layer trainable pooling with cross-attention, EOS vs mean vs learnable, Mistral-7B experiments

6. **Ahmed Shikha — Pooling Encoder-only Transformers Embeddings** (Blog, 2024)
   - URL: https://kickitlikeshika.github.io/2024/07/27/pooling-transformer-embeddings.html
   - Type: Tutorial with code
   - Key: Eight pooling strategies (CLS, mean, max, concat, LSTM, weighted-layer, attention), Kaggle competition insights

7. **WaggyLabs — Self Attention and Pooling in the Set Transformer** (2023)
   - URL: https://ktamarov.com/blog/2023/oct/02/self-attention-and-pooling-in-the-set-transformer/
   - Type: Conceptual explanation
   - Key: PMA mathematical derivation of dimensionality reduction

8. **timm (Hugging Face) — attention_pool2d.py**
   - URL: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool2d.py
   - Type: Production PyTorch code
   - Key: RotAttentionPool2d, AttentionPool2d with rotary/absolute positional embeddings

9. **Snowdar/asv-subtools — pooling.py**
   - URL: https://github.com/Snowdar/asv-subtools/blob/master/pytorch/libs/nnet/pooling.py
   - Type: Production PyTorch code (speaker verification)
   - Key: AttentiveStatisticsPooling, MultiHeadAttentionPooling, MQMHASP

---

## Key Findings

### 1. Attention Pooling Core Mechanism

Standard attention: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`

**Attention pooling** replaces the input-derived query with a **learnable query parameter**:
```
Q_cls = q_cls · W^Q
AttentionPool(Q_cls, K, V) = softmax(Q_cls · K^T / sqrt(d_k)) V
```

Where `q_cls` is `nn.Parameter(torch.randn(d_model))` — a single learned vector that acts as a "what should I extract?" query over the full sequence.

### 2. Simplest Attention Pooling (Single-Head)

From Selokar (source 2):
```python
query_vector = nn.Parameter(torch.randn(hidden_dim))
attn_scores = torch.matmul(hidden_states, query_vector)  # [B, N]
attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, N, 1]
pooled = torch.sum(hidden_states * attn_weights, dim=1)  # [B, D]
```

Input: `[B, N, D]` → Output: `[B, D]`

### 3. Learned Aggregation (Multi-Head, with FFN)

From Warner (source 1), based on Touvron et al.:
```python
class AttentionPool2d(nn.Module):
    def __init__(self, ni, bias=True, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(ni)
        self.q = nn.Linear(ni, ni, bias=bias)
        self.vk = nn.Linear(ni, ni*2, bias=bias)
        self.proj = nn.Linear(ni, ni)

    def forward(self, x, cls_q):
        x = self.norm(x.flatten(2).transpose(1,2))
        B, N, C = x.shape
        q = self.q(cls_q.expand(B, -1, -1))
        k, v = self.vk(x).reshape(B, N, 2, C).permute(2, 0, 1, 3).chunk(2, 0)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, C)
        return self.proj(x)

class LearnedAggregation(nn.Module):
    def __init__(self, ni, ffn_expand=3, norm=nn.LayerNorm, act_cls=nn.GELU):
        super().__init__()
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones(ni))
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones(ni))
        self.cls_q = nn.Parameter(torch.zeros(ni))
        self.attn = AttentionPool2d(ni, True, norm)
        self.norm = norm(ni)
        self.ffn = nn.Sequential(
            nn.Linear(ni, int(ni*ffn_expand)),
            act_cls(),
            nn.Linear(int(ni*ffn_expand), ni)
        )
        nn.init.trunc_normal_(self.cls_q, std=0.02)

    def forward(self, x):
        x = self.cls_q + self.gamma_1 * self.attn(x, self.cls_q)
        return x + self.gamma_2 * self.ffn(self.norm(x))
```

Key: uses per-channel `gamma` parameters initialized to 1e-4 for stable training, plus FFN for richer pooling.

### 4. Set Transformer PMA

From Arrigoni (source 3):
```
PMA_k(Z) = MAB(S, rFF(Z))
```
Where S ∈ R^(k×d) are k learnable seed vectors.

Tensor shapes: input `[B, n, d]` → output `[B, k, d]` (k is number of seed queries).

This is the purest form of "multiple learnable queries attend to variable-length sequence".

### 5. Perceiver Cross-Attention Pooling

From Tigges (source 4):
- Latent array: `nn.Parameter(torch.empty(N, d).normal_(std=0.02))`
- Q from latent, K/V from input
- Complexity: O(M×N) instead of O(M²)
- Input: `[B, M, d]` (variable M) → Output: `[B, N, d]` (fixed N latents) → mean pool → `[B, d]`

### 6. Multi-Layer Trainable Pooling (LLM Embeddings)

From Zhan et al. (source 5):
- Combines hidden states from ALL layers (not just last)
- Trainable layer weight matrix captures per-layer importance
- Cross-attention maps variable-size to fixed-size (like Flamingo)
- +4.2% improvement on retrieval vs EOS-only pooling
- BUT no universal winner: task-dependent performance

### 7. Attention Pooling for NLP (Kaggle-Proven)

From Shikha (source 6):
```python
attention_pooling = nn.Sequential(
    nn.Linear(768, 512),
    nn.Tanh(),
    nn.Linear(512, 1),
    nn.Softmax(dim=1)
)
weights = attention_pooling(last_hidden_state)  # [B, T, 1]
context_vector = torch.sum(weights * last_hidden_state, dim=1)  # [B, D]
```

"Showed great performance in several Kaggle NLP competitions."

---

## Comparison Results

### Warner's Experiments (Vision, Small Datasets)

| Method | Petals F1 | Imagenette Acc | ImageWoof Acc |
|--------|-----------|----------------|---------------|
| Average Pooling (baseline) | 0.9179 | 95.77% | 90.38% |
| Learned Aggregation | 0.8602 | 94.36% | 87.14% |
| Learned Agg + Sandwich Norm | 0.8743 | — | 87.67% |
| Concat Attn Pool (best hybrid) | 0.9138 | 95.35% | 90.12% |
| Slim Avg+Attn | 0.9060 | 95.29% | 89.47% |

**Finding**: On small vision datasets, average pooling beat learned aggregation. Hybrid concat approach closed the gap.

### Zhan et al. (LLM Embeddings, Mistral-7B)

| Model | Pooling | Attention | STS | Retrieval | Clustering |
|-------|---------|-----------|-----|-----------|-----------|
| 1 | EOS Token | Causal | 0.8302 | 0.5394 | 0.4503 |
| 2 | Last-Layer Trainable | Causal | +0.0129 | +0.0102 | -0.0076 |
| 3 | Multi-Layers Trainable | Causal | +0.0118 | +0.0135 | -0.0017 |
| 5 | Multi-Layers Trainable | Bidirectional | +0.0166 | +0.0226 | -0.0246 |

**Finding**: Multi-layer trainable pooling + bidirectional attention best for STS/retrieval (+4.2% retrieval), but WORSE for clustering. No universal winner.

### General Consensus Across Sources

| Pooling Method | Best For | Worst For |
|---------------|----------|-----------|
| Mean pooling | Safe default, low-resource, balanced features | Sequences where specific positions dominate |
| CLS token | Models pretrained with CLS objective | Models without CLS pretraining |
| Last hidden state | GRU/LSTM | Transformer (information at all positions) |
| Attention pooling | High-resource, complex tasks, variable importance | Low-resource (overfits with few parameters) |
| Set Transformer PMA | Set-input tasks (variable cardinality) | Fixed-length sequences |
| Perceiver cross-attn | Very long/multimodal input | Small/simple sequences |

---

## Computational Cost

| Method | Extra Parameters | Extra FLOPs | Memory |
|--------|-----------------|-------------|--------|
| Mean/Max pool | 0 | O(N·D) | Negligible |
| CLS token | D params | O(N·D) | Negligible |
| Simple attn pool (1 query) | D + D params | O(N·D) | 1 attn vector |
| Learned Aggregation | ~4·D² (attn + FFN) | O(N·D + D²) | Moderate |
| PMA (k seeds, h heads) | k·D + 3·D² | O(k·N·D) | k attn maps |
| Perceiver cross-attn | N_latent·D + 3·D² | O(M·N·D) | N×M attn map |

For a typical event sequence encoder (D=128, N=500 events):
- Simple attention pooling adds ~256 parameters (negligible)
- Learned Aggregation adds ~65K parameters
- PMA with k=1 adds ~49K parameters
