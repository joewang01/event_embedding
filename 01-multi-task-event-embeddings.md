---
title: "Event Data Embedding Methods — Multi-Task Learning for Event Sequences"
source: "https://chatgpt.com/c/69d5d95a-0e5c-83e8-a212-1240bb481968"
author: ChatGPT (GPT-4o)
published: 2026-04-07
created: 2026-04-07
type: conversation
tags:
  - event-embedding
  - multi-task-learning
  - class-imbalance
---
Yes — **using multiple downstream targets to learn the embedding is a strong approach** for your kind of data.

In fact, this is often better than pure self-supervised learning when you already have several meaningful business outcomes. Multi-task learning is designed to learn a **shared representation** that captures structure useful across related tasks, which can improve generalization compared with training separate models task by task. [arXiv+1](https://arxiv.org/pdf/2307.06887?utm_source=chatgpt.com)

For event data, the idea is:

- shared event encoder
- one embedding layer / pooled sequence representation
- multiple prediction heads, one per target

So the model looks like:

`event sequence -> encoder -> shared embedding -> {head1, head2, head3, ...}`

After training, you keep the **shared embedding** as the representation for each entity or sequence. This is a standard and very practical way to produce supervised embeddings from sequence data. [arXiv+1](https://arxiv.org/pdf/2307.06887?utm_source=chatgpt.com)

Why this can work well for you:

- different targets force the embedding to capture **broader signal**
- the model is less likely to overfit to only one narrow objective
- sparse targets can borrow strength from other related targets
- it aligns the embedding directly with downstream business use, instead of hoping a self-supervised task transfers well

That is exactly the kind of inductive transfer multi-task learning is meant to provide. [arXiv+1](https://arxiv.org/html/2402.17016v1?utm_source=chatgpt.com)

A good setup for your case is:

**Input per event**

- event type
- timestamp / time gap / recency
- key categorical fields
- key numeric fields

**Encoder**

- GRU/LSTM if you want a simpler baseline
- Transformer with self-attention if long-range dependencies matter

**Shared representation**

- pooled sequence embedding
- or CLS-style token embedding

**Prediction heads**

- binary classification heads for flags
- regression heads for scores / losses
- maybe ranking head if relevant

Train all heads jointly, then use the shared hidden vector as the embedding. This is especially natural for event sequences, where sequence encoders already produce a latent representation that can be reused across tasks. [arXiv+1](https://arxiv.org/abs/2308.00129?utm_source=chatgpt.com)

The main caution is that **not all targets should be thrown in together**. If some targets are unrelated, noisy, or conflict strongly with the others, they can hurt the shared embedding rather than help it. Recent work on task grouping in multi-task learning highlights this issue directly: task selection and grouping matter. [arXiv](https://arxiv.org/html/2410.12774v1?utm_source=chatgpt.com)

So the best practice is:

- include targets that are **related but not identical**
- avoid highly redundant heads unless they add stability
- exclude targets that are mostly noise
- consider weighting losses so one target does not dominate training

A very practical target mix would be:

- policy ownership flags
- risk flags
- existing model scores
- future event occurrence
- loss / severity style targets

That gives the embedding both **broad behavioral signal** and **business relevance**.

My honest take: for your event-driven data, **multi-target downstream prediction is probably one of the best ways to create embeddings**, especially if self-supervised learning feels weak or too indirect. It is often the safest first strategy.

A simple formula for the loss is:

$$
L = \sum_i w_i L_i
$$

where each $L_i$ is one target’s loss and $w_i$ controls its importance.

Best path:

1. build one event encoder
2. train on 3 to 8 related targets
3. extract the shared embedding
4. evaluate whether that embedding improves each downstream task versus single-target training

If you want, I can turn this into a concrete architecture for your RiskView-style event data, including which targets should go into the first version.

Yes — that is exactly the hard part, and it changes the training design.

When each target has a **very low positive rate** and also a **different valid reference population**, a single naive multi-task batch usually breaks down for two reasons:

1. the batch may contain almost no positives for some heads, so that head gets weak or unstable gradient, and
2. each task really has its own “who is eligible to be negative” set, so using one common negative pool can introduce label noise. Multi-task learning papers consistently note that differences in loss scale and task dynamics can destabilize shared training, which is why methods like GradNorm, DB-MTL, and PCGrad exist in the first place. [arXiv+2](https://arxiv.org/abs/1711.02257?utm_source=chatgpt.com)

So I would **not** treat this as one ordinary multi-label problem with one global sampler.

The better setup is:

## 1\. Keep one shared encoder, but make each target head task-aware

Use:

- a **shared event encoder** to produce the entity embedding
- one **task-specific head** per target
- one **task-specific mask** saying which samples are eligible for that target

So for target $t$, compute loss only on its valid population:

$$
L_t = \frac{1}{|M_t|}\sum_{i \in M_t} \ell(y_{it}, \hat y_{it})
$$

where $M_t$ is the reference population for target $t$.

That is the cleanest way to respect the fact that different targets have different valid negative populations.

## 2\. Do not try to force one universal resampling rule across all targets

Because your targets have different reference groups, a single batch-level class balance rule will distort at least some tasks.

Instead, use one of these two strategies:

### Strategy A: full shared batch + masked per-task loss

Sample a general batch from the union population, then:

- for each head, only score the eligible samples
- use **task-specific positive weighting** or **focal loss** within that head

This is simpler and often the first thing I would try. Focal loss was designed specifically to stop easy negatives from overwhelming rare positives. [arXiv+1](https://arxiv.org/abs/1708.02002?utm_source=chatgpt.com)

### Strategy B: alternating task-specific minibatches

At each step:

- choose one task
- sample from that task’s own reference population
- oversample positives or hard examples for that task
- update the shared encoder plus that task head

This often works better when the tasks have very different eligible populations.

In your case, I think **Strategy B is more natural**.

## 3\. Separate “which task to train now” from “which examples to sample for that task”

That is usually the key design.

For each step:

- sample a task $t$ according to a task schedule
- sample data only from the valid reference group for $t$
- construct a balanced or semi-balanced batch for $t$
- compute $L_t$ and backprop through the shared encoder

This avoids mixing incompatible negative definitions inside one batch.

A very practical task schedule is:

- equal probability by task at first, or
- probability proportional to `sqrt(number of positives)` so ultra-rare tasks are not starved but also do not dominate

## 4\. Use weighting inside each task, not only across tasks

There are two imbalance problems here:

- **within-task class imbalance**
- **between-task optimization imbalance**

You need to handle both.

For within-task imbalance:

- `BCEWithLogitsLoss(pos_weight=...)`
- or **focal loss**
- sometimes LDAM-style margin methods help in imbalanced settings too. [arXiv+1](https://arxiv.org/abs/1708.02002?utm_source=chatgpt.com)

For between-task imbalance:

- **GradNorm** for dynamic task loss balancing
- **DB-MTL** for balancing both losses and gradients
- **PCGrad** if gradients from different tasks conflict strongly [arXiv+2](https://arxiv.org/abs/1711.02257?utm_source=chatgpt.com)

My recommendation:

- start with **task-specific weighted BCE or focal loss**
- then add **GradNorm** only if one head dominates
- add **PCGrad** only if multi-task interference is obvious

## 5\. Build the embedding from the shared encoder, not from the heads

Even if tasks are trained separately in alternating minibatches, the embedding still comes from the shared encoder.

So the architecture is still:

`event sequence -> shared encoder -> embedding -> task head t`

The difference is only in how training batches are scheduled.

## 6\. A strong practical recipe for your case

I would do this:

### Model

- shared Transformer or GRU encoder on event sequence
- pooled entity embedding
- one binary head per target

### Data

For each target $t$:

- define its own eligible reference population
- define positives and negatives only inside that population
- create a per-target sampler

### Training loop

At each step:

1. choose a target $t$
2. sample a minibatch from target $t$ ’s reference population
3. oversample positives to a reasonable ratio, like 1:4, 1:8, or 1:16 instead of trying to match 1:1
4. compute only target $t$ ’s loss
5. update shared encoder + head $t$

Then optionally every few steps:

- run a mixed validation pass across all targets
- adjust task weights or sampling probabilities

This is usually much cleaner than forcing all heads into every batch.

## 7\. Why I would avoid one giant joint batch in your situation

If target A’s valid negatives are people with policy type A, and target B’s valid negatives are people with policy type B, then a common batch may contain:

- many samples invalid for A
- many samples invalid for B
- too few positives for both

That makes gradients noisy and inefficient.

So your problem is really closer to **multi-task learning with task-specific datasets** than to ordinary multi-label learning. That distinction matters. Multi-task literature explicitly treats differing task data distributions as a core training issue. [arXiv+2](https://arxiv.org/html/2109.09138v2?utm_source=chatgpt.com)

## 8\. Best first-choice solution

If I were implementing this from scratch, I would choose:

- **shared encoder**
- **task-specific heads**
- **alternating task-specific minibatches**
- **task-specific eligible population masks**
- **weighted BCE or focal loss inside each task**
- **GradNorm only if task domination appears**

That is the most defensible setup.

## 9\. One more important point: calibration

With heavy resampling, the raw predicted probability is often miscalibrated. Focal loss and oversampling can improve ranking/recall but distort probability interpretation, so evaluate calibration separately and recalibrate if needed on a validation set drawn from the original task distribution. [arXiv+1](https://arxiv.org/abs/1708.02002?utm_source=chatgpt.com)

So the short conclusion is:

**Yes, your concern is real.**  
Because each target has a different reference population, you should probably **not** train with one shared balanced batch across all heads. Use **task-specific sampling and masking**, while keeping a **shared encoder** to learn the embedding.

A compact formula for the full setup is:

$$
L = \sum_t w_t L_t,\quad
L_t = \frac{1}{|M_t|}\sum_{i\in M_t}\ell_t(y_{it}, \hat y_{it})
$$

with $M_t$ as the valid population mask for task $t$, and where the batches for each $L_t$ are sampled from that task’s own reference group.

I can turn this into a concrete PyTorch training loop for your case, including sampler design and how to keep the embedding stable across tasks.