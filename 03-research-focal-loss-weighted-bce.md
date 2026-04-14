---
title: "Research: Focal Loss and Weighted BCE in PyTorch"
source: multiple
author: various
published: 2026-04-07
created: 2026-04-07
type: research-compilation
tags:
  - event-embedding
  - class-imbalance
  - focal-loss
  - pytorch
---

# Research: Focal Loss and Weighted BCE

Compiled from 11 sources during wiki research pass.

## Key Sources

1. **TorchVision sigmoid_focal_loss** — Canonical implementation
2. **PyTorch BCEWithLogitsLoss docs** — pos_weight math
3. **PyTorch Forums** — Multiple threads on pos_weight gotchas and focal loss implementations
4. **Medium / Codegenes blogs** — Walkthrough implementations
5. **GitHub Issue #6229** — Formula for combining focal loss + pos_weight
6. **pytorch-focalloss PyPI** — Drop-in library

## Critical Gotcha

When combining focal loss with pos_weight, compute p_t from UNWEIGHTED BCE. The focal modulation should reflect true model confidence, not the weighted loss.

## pos_weight Computation

```python
pos_weight = num_negatives / num_positives
# Example: 2% positive rate → pos_weight = 49.0
```

## Gamma Tuning

- gamma=0: standard BCE
- gamma=1: mild focusing
- gamma=2: default (RetinaNet)
- gamma=3-5: very aggressive, risk instability
