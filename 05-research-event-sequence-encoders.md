---
title: "Research: Event Sequence Encoder Architectures in PyTorch"
source: multiple
author: various
published: 2026-04-07
created: 2026-04-07
type: research-compilation
tags:
  - event-embedding
  - sequence-encoding
  - pytorch
  - transformer
  - gru
---

# Research: Event Sequence Encoder Architectures

Compiled from 11 sources during wiki research pass.

## Key Sources

1. **PyTorch-Lifestream** (GitHub, IJCAI 2025) — Production library for event sequence embeddings. TrxEncoder for heterogeneous events.
2. **EBES Benchmark** (KDD 2025) — 10 datasets, finding: GRU dominates Transformer for event sequences
3. **PyTorch Forums** — Transformer with CLS token for arbitrary event sequences
4. **sentence-transformers Pooling.py** — 6 pooling strategies with mask-aware implementations
5. **Time2Vec** — Learned temporal encoding (9.3% improvement over fixed sinusoidal)
6. **HuggingFace Time Series Transformer** — Feature handling patterns for mixed categorical/numerical/temporal

## Key Finding: GRU > Transformer for Events

EBES benchmark across 10 diverse event sequence datasets found GRU consistently outperforms Transformer. This contradicts the common assumption that Transformers always win.

## Design Decision Matrix

- Categorical features: nn.Embedding per feature, concat
- Numerical features: BatchNorm + Linear projection
- Temporal: time differences as features OR Time2Vec
- Pooling for GRU: last hidden state
- Pooling for Transformer: CLS token or mean pooling
