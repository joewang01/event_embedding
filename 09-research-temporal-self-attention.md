---
title: "Research: Temporal Self-Attention for Irregular Event Sequences"
source: multiple
author: various
published: 2026-04-07
created: 2026-04-07
type: research-compilation
tags:
  - event-embedding
  - self-attention
  - transformer
  - temporal-encoding
---

# Research: Temporal Self-Attention for Irregular Event Sequences

Compiled from 11 sources during wiki research pass.

## Key Sources

1. **SAHP** (ICML 2020) — Self-Attentive Hawkes Process. Time intervals as phase shifts in PE.
2. **THP** (ICML 2020) — Transformer Hawkes Process. Sinusoidal encoding of continuous timestamps.
3. **Hawkes-Attention** (2026) — Time modulation directly in Q,K,V via learned neural kernels. Eliminates PE.
4. **mTAN** (ICLR 2021) — Multi-Time Attention Networks. Time itself becomes the query.
5. **ContiFormer** (NeurIPS 2023) — Q,K,V as continuous ODE trajectories.
6. **XTSFormer** (2024) — Feature-dependent cycle-aware time PE + hierarchical attention.
7. **TAA-THP** (2021) — Decoupled temporal + content attention terms.
8. **RoTHP** (2024) — Rotary position embedding for relative time encoding.
9. **EasyTPP** (ICLR 2024) — Unified temporal point process benchmark.
10. **Yang, Mei, Eisner** (ICLR 2022) — Transformer embeddings of irregularly spaced events.
11. **Positional Encoding Survey** (2025) — Comprehensive survey of time PE methods.

## Five Architectural Patterns

1. Additive Time Encoding (THP, SAHP) — simplest
2. Multiplicative Time Modulation (Hawkes-Attention) — most principled
3. Time as Query (mTAN) — best for sparse/irregular
4. Continuous ODE Trajectories (ContiFormer) — most expressive, heaviest
5. Relative Time in Attention Score (RoTHP, XTSFormer) — theoretically correct
