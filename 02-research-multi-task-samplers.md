---
title: "Research: PyTorch Multi-Task Samplers and DataLoader Patterns"
source: multiple
author: various
published: 2026-04-07
created: 2026-04-07
type: research-compilation
tags:
  - event-embedding
  - multi-task-learning
  - pytorch
  - sampling
---

# Research: PyTorch Multi-Task Samplers

Compiled from 7 sources during wiki research pass.

## Key Sources

1. **pytorch-balanced-sampler** (GitHub: khornlund) — Alpha-interpolated class balancing with WeightedFixedBatchSampler
2. **BatchSchedulerSampler** (Gist: bomri) — Round-robin task alternation with auto-resample for ConcatDataset
3. **Sentence Transformers** (HuggingFace) — Production-grade RoundRobinBatchSampler and ProportionalBatchSampler
4. **T5 Temperature-Scaled Mixing** (Raffel et al.) — Formula: rate = size^(1/T), T=2 gives sqrt-proportional
5. **DeepMojiBatchSampler** (Gist: Thomas Wolf, HuggingFace co-founder) — Binary positive/negative balanced batch construction
6. **GumGum Tech Blog** — Simplest zip-based alternating training loop
7. **PyTorch Forum** — Batch sampler for multitask datasets

## Key Pattern: Temperature-Scaled Task Mixing

```python
def calc_mixing_rates(datasets, temperature=2.0, k=1000):
    mixing_rates = []
    for dataset in datasets:
        rate = len(dataset)
        if k:
            rate = min(rate, k)
        if temperature != 1.0:
            rate = rate ** (1.0 / temperature)
        mixing_rates.append(rate)
    return mixing_rates
```

- T=1: proportional to raw dataset size
- T=2: sqrt-proportional (recommended for event embeddings)
- T→∞: equal/uniform mixing
