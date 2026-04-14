---
source: Web research (15+ sources)
author: Various
date: 2026-04-07
type: research
---

# Research: Long-Tail Event Type Distributions in Sequence Embeddings

## Research Focus
Handling power-law / long-tail distributions in event types within sequences. When some event types (e.g., inquiries) appear for most entities while others (e.g., student file records) are rare, the model faces severe INPUT-side imbalance — different from label imbalance.

## Key Sources

### Source 1: Adaptive Regularization for Large-Scale Sparse Feature Embedding Models (ICLR 2026)
- **URL**: https://arxiv.org/html/2511.06374
- **Key contribution**: AdamAR — adaptive regularization that assigns frequency-dependent weight decay to embedding rows
- **Core formula**: λ^k_ij = min(1, α × I^k_ij) where I is update interval (proxy for inverse frequency)
- **Mechanism**: Sparse/rare features get STRONGER regularization to prevent overfitting; frequent features get WEAKER regularization to preserve fitting capacity
- **One-epoch problem**: Low-frequency embedding rows are the primary cause of multi-epoch overfitting — unconstrained norms grow, increasing Rademacher complexity
- **Results**: AdamAR on iPinYou: 0.7566 AUC vs AdamW 0.7475. Maintains performance across epochs (0.7566→0.7724) while baseline Adam drops (0.7515→0.7014)
- **Rademacher bound**: Sum of squared embedding norms directly impacts generalization — adaptive regularization controls this inversely to frequency
- **Production**: Deployed at Alibaba
- **Implementation overhead**: ~1P additional memory (one counter per parameter), ~11P vs ~9P multiplications per step

### Source 2: Frequency-Adaptive Learning Rate for Embedding Tables (Pinterest, May 2025)
- **URL**: https://arxiv.org/html/2505.05605v1
- **Key contribution**: FAL — row-wise learning rate scaled by log frequency
- **Formula**: η_T[i] = η_T × log(F_T[i]+1) / max_j log(F_T[j]+1)
- **Mechanism**: Rare items get LOWER learning rates to prevent overfitting (opposite intuition from Adagrad which increases LR for rare items!)
- **Key insight**: Multi-epoch overfitting is caused by infrequent rows. Adagrad's approach of INCREASING LR for rare items actually makes overfitting worse
- **Log vs linear scaling**: Log scaling outperforms linear scaling in experiments
- **Production at Pinterest**: Reduced epoch boundary loss jumps from +1.19% to +0.23%; cumulative AUC gain 0.07-0.20%
- **Implementation**: Online frequency accumulation, no precomputation needed, 3.125% memory overhead
- **Combined with**: Sparse Optimizer (50x higher base LR for embedding tables) + cardinality reduction via binary hashing

### Source 3: Deep Feature Embedding for Tabular Data (DFE, 2024)
- **URL**: https://arxiv.org/html/2408.17162v1
- **Key contribution**: Two-step embedding: entity identification + deep transformation via shared DNN
- **Problem**: "embeddings of infrequent entities are less indexed and trained" in standard lookup tables
- **Solution**: Small identification vector (dim d̂) → shared DNN → full embedding (dim d)
- **Mechanism**: Shared DNN enables "collaborative learning among all categorical entities" — rare categories benefit from patterns learned from frequent ones
- **Architecture**: Identification lookup → feed-forward with residual connections + ExU activation
- **Key property**: "model compression and collaborative learning via deep matrix factorization"
- **Relevant for**: Power-law distributed categorical features where standard embeddings under-train rare values

### Source 4: Stochastic Shared Embeddings (SSE, NeurIPS 2019)
- **URL**: https://arxiv.org/abs/1905.10630
- **Key contribution**: Regularization by stochastically transitioning between embeddings during SGD
- **Mechanism**: During forward pass, embeddings are probabilistically swapped with other embeddings via a transition matrix — forces embedding sharing
- **How it helps rare categories**: Forces rare item embeddings to share statistical information with more frequent items, preventing overfitting to limited examples
- **Two variants**: SSE-SE (no prior info needed, data-driven) and SSE-Graph (uses knowledge graph structure)
- **Theory**: Reduces Rademacher complexity of embedding layers
- **Analogy**: "Dropout for embeddings" — but targeted at the embedding layer specifically
- **Results**: Tested on 6 tasks from recommender systems to BERT/transformers
- **Code**: https://github.com/wuliwei9278/SSE

### Source 5: Cross Decoupling Network for Long-Tail Recommendation (CDN, KDD 2023)
- **URL**: https://arxiv.org/abs/2210.14309
- **Key contribution**: Decouple memorization (head items) from generalization (tail items) via mixture-of-experts
- **Key insight**: Long-tail bias comes from TWO sources: (1) item distribution skew AND (2) different user preference patterns for head vs tail items
- **Architecture**: MoE on item side (memorization vs generalization experts) + bilateral branch network on user side + adapter to aggregate
- **Adapter**: "Softly shifts training attention to tail items"
- **Results**: Improves both tail AND head items (not a tradeoff). Production-tested at Google
- **Relevance**: Applicable to entity-level embeddings where some entities have many events (head) and others have few (tail)

### Source 6: Entity Embedding-based Anomaly Detection for Heterogeneous Categorical Events (2016)
- **URL**: https://ar5iv.labs.arxiv.org/html/1608.07502
- **Key contribution**: APE model — embeds heterogeneous categorical entities into shared latent space
- **Mechanism**: Weighted pairwise interactions (w_ij) between entity types automatically adjust for varying interaction frequencies
- **Scoring**: S_θ(e) = Σ w_ij(v_ai · v_aj) — learned weights distinguish critical from less regular relationships
- **Rare events**: Infrequent event patterns naturally receive low likelihood scores; model learns regularities and flags anomalous combinations
- **Context-dependent noise**: NCE with uniform entity type sampling for realistic negative examples

### Source 7: Multimodal Banking Dataset (MBD, 2024)
- **URL**: https://arxiv.org/html/2409.17587v1
- **Key contribution**: Per-modality separate embedding → concatenation fusion for entity-level representation
- **Dataset**: ~2M banking clients, 950M money transfers, 1B geo events, 5M dialog embeddings
- **Event types**: 56 transaction types, 62 subtypes, up to 40K unique values in some fields
- **Embedding approach**: 24-dimensional embeddings for categorical features, clipping high-cardinality fields
- **Distribution**: 81% of clients have no purchases, 15% have one, 4% have 2+ (extreme imbalance)
- **Fusion strategies tested**: Blending (weighted sum of posteriors), Late Fusion (concatenate embeddings), Early Fusion
- **Aggregation**: hand-crafted statistics (3000 features) + learned pooling (min, max, mean, std from encoder)

### Source 8: Looking Around You — External Information for Event Sequences (2025)
- **URL**: https://arxiv.org/html/2502.10205
- **Key contribution**: External context vectors from co-occurring user sequences enhance individual embeddings
- **Rare type handling**: Rare types benefit indirectly through external context — "categories often less interesting" filtered to "100 most popular codes"
- **Architecture**: (1) encoder → internal embedding, (2) aggregation → external context, (3) concatenation → enhanced representation
- **Entity-level processing**: Individual user sequences receive personalized external context from other users

### Source 9: Deep Hash Embedding (DHE, 2020)
- **URL**: https://arxiv.org/abs/2010.10784
- **Key contribution**: Dynamic embedding computation via hashing — no lookup table needed
- **Power-law problem**: "Categorical features in recommendation systems usually follow highly skewed power-law distributions"
- **Dedicated + hashed**: Top 10% most frequent values get dedicated embeddings; rest use double hashing
- **DHE mechanism**: Each feature → vector of hashed values (1024 hash functions) → DNN → embedding
- **Benefit for rare**: No cold-start problem — any value gets a deterministic embedding via hash functions

### Source 10: Meta Scaling and Shifting Networks for Cold-Start Embedding Warm-Up (SIGIR 2021)
- **URL**: https://arxiv.org/abs/2105.04790
- **Key contribution**: Meta-learning to generate initial embeddings for new/rare items from content features
- **Problem**: "Cold-start items with only limited interactions — hard to train reasonable item ID embeddings"
- **Solution**: Generates scaling and shifting functions per item; addresses fast adaptation + noise reduction
- **Related**: Content-based warm-up, CVAE, adversarial alignment for bootstrapping embeddings

## Synthesis: Techniques for Handling Rare Event Types in Input Sequences

### Technique Category 1: Frequency-Aware Regularization
- **AdamAR** (ICLR 2026): Stronger weight decay for rare embedding rows, weaker for frequent → prevents overfitting of under-trained embeddings
- **FAL** (Pinterest 2025): Lower learning rate for rare rows via log-frequency scaling → prevents multi-epoch overfitting
- **Key insight**: Standard approaches (uniform weight decay, Adagrad boosting rare items) are WRONG for embeddings. Rare items need MORE regularization or SLOWER learning, not less

### Technique Category 2: Embedding Architecture
- **DFE two-step**: Small ID vector → shared DNN → full embedding. Collaborative learning helps rare categories
- **DHE hashing**: Hash-based dynamic embeddings eliminate cold-start entirely
- **Dedicated + hashed split**: Top-K frequent get full embeddings, rest share via hashing

### Technique Category 3: Stochastic Regularization
- **SSE**: Probabilistic embedding sharing during training. Rare items borrow from frequent items
- **Connection to dropout**: SSE is "dropout for embeddings" — reduces Rademacher complexity

### Technique Category 4: Head/Tail Decoupling
- **CDN**: Mixture-of-experts separates memorization (head) from generalization (tail)
- **Bilateral branches**: Different user/entity representations for head vs tail items

### Technique Category 5: External Enhancement
- **Context injection**: External context vectors from co-occurring entities help rare types indirectly
- **Meta-learning warm-up**: Generate initial embeddings for rare items from content features

### Technique Category 6: Type-Level Aggregation
- **Per-type embedding + fusion**: Embed each event type separately, then combine (blending, late fusion, early fusion)
- **Intra-type then inter-type attention**: Aggregate within each type first, then combine across types
- **Type-aware scoring**: Learned weights distinguish critical type interactions from less regular ones

## Practical Recommendations for Public Records Event Embedding

### For Severe Event Type Imbalance (Power-Law Input Distribution):

1. **Do NOT group rare types into UNK** — rare types like "student file records" may be the most informative signals
2. **Use AdamAR or FAL** for the embedding table — give rare event types MORE regularization, not less
3. **Consider DFE two-step embedding** — shared DNN lets rare types benefit from frequent type patterns
4. **Apply SSE** as additional regularization on the event type embedding layer
5. **Use per-type attention pooling** — let the model learn which types matter at inference time
6. **Consider temperature-scaled type sampling during training** — oversample batches with rare types (but NOT the same as label oversampling)
7. **Monitor embedding norms by frequency** — if rare type embedding norms are growing unconstrained, apply norm clipping

### Architecture Pattern:
```
Per-event: [event_type_embedding + feature_embeddings + temporal_encoding] → event_vector
Per-type aggregation: group by type → attention pool within each type → type_summary_vectors
Cross-type fusion: attend over type_summary_vectors → entity_embedding
```

This gives the model explicit type-level representations rather than hoping the sequence encoder discovers type importance on its own.
