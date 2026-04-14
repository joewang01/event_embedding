---
title: "Research: GradNorm, PCGrad, and Multi-Task Gradient Balancing"
source: multiple
author: various
published: 2026-04-07
created: 2026-04-07
type: research-compilation
tags:
  - event-embedding
  - multi-task-learning
  - gradient-balancing
  - pytorch
---

# Research: GradNorm, PCGrad, and Gradient Balancing

Compiled from 10 sources during wiki research pass.

## Key Sources

1. **LucasBoTang/GradNorm** (GitHub, 120 stars) — Clean standalone implementation
2. **WeiChengTseng/Pytorch-PCGrad** (GitHub) — Drop-in optimizer wrapper
3. **codegenes.net** — PCGrad from-scratch tutorial
4. **LibMTL** (GitHub, 2.5k stars, JMLR 2023) — 25 gradient methods in unified API
5. **anzeyimana/Pytorch-PCGrad-GradVac-AMP-GradAccum** — AMP-compatible implementation
6. **CAGrad** (NeurIPS 2021) — Conflict-Averse Gradient Descent

## Key Insight

GradNorm addresses gradient *magnitude* imbalance. PCGrad addresses gradient *direction* conflicts. They solve different problems and can be combined.

## Critical Caveat

Kurin et al. (2022, Google) found that gradient manipulation methods may not beat simple scalarization with comprehensive weight tuning. Gradient methods are most valuable when you have many tasks and cannot afford exhaustive weight sweeps.
