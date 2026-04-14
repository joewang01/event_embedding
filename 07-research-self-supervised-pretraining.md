---
source: multiple (see below)
author: Various (15+ sources)
date: 2026-04-07
type: research
---

# Research: Self-Supervised Pretraining for Event Sequence Embeddings

## Sources

### Primary Papers

1. **CoLES: Contrastive Learning for Event Sequences with Self-Supervision**
   - URL: https://arxiv.org/abs/2002.08232
   - Authors: Dmitrii Babaev, Ivan Kireev, Nikita Ovsov, et al.
   - Published: SIGMOD 2022
   - Key contribution: Adapts contrastive learning to discrete event sequences; deployed at large European financial institution with hundreds of millions of dollars in annual gains

2. **MLEM: Generative and Contrastive Learning as Distinct Modalities for Event Sequences**
   - URL: https://arxiv.org/abs/2401.15935
   - Authors: VityaVitalich et al.
   - Code: https://github.com/VityaVitalich/MLEM
   - Key contribution: Treats contrastive and generative SSL as distinct modalities, aligns them via SigLIP; most universal approach across diverse tasks

3. **Uniting Contrastive and Generative Learning for Event Sequences Models**
   - URL: https://arxiv.org/abs/2408.09995
   - Published: 2024
   - Key contribution: CMLM loss combining CoLES + masked event prediction in latent space; provides explicit loss formulas

4. **EBES: Easy Benchmarking for Event Sequences**
   - URL: https://arxiv.org/abs/2410.03399
   - Code: https://github.com/On-Point-RND/EBES
   - Published: KDD 2025
   - Key contribution: Standardized benchmark across 10 datasets; GRU-based models dominate; CoLES ranks #1 overall

5. **Towards Unified Approaches in Self-Supervised Event Stream Modeling: Progress and Prospects**
   - URL: https://arxiv.org/abs/2502.04899
   - Published: February 2025
   - Key contribution: Comprehensive taxonomy of SSL methods for event streams (predictive + contrastive paradigms)

6. **Self-Supervised Contrastive Pre-Training for Multivariate Point Processes**
   - URL: https://arxiv.org/abs/2402.00987
   - Published: 2024
   - Key contribution: Masks event epochs + inserts "void" epochs; up to 20% improvement on next-event prediction

### Libraries & Implementations

7. **PyTorch-Lifestream**
   - URL: https://github.com/dllllb/pytorch-lifestream
   - Docs: https://pytorch-lifestream.github.io/pytorch-lifestream/
   - Paper: IJCAI 2025
   - Implements: CoLES, CPC, MLM, RTD, NSP, SOP, VICReg, Barlow Twins
   - Install: `pip install pytorch-lifestream`

8. **ptls-experiments (Benchmark Results)**
   - URL: https://github.com/pytorch-lifestream/ptls-experiments
   - Contains: Reproducible experiments on 6 datasets with all SSL methods

---

## Taxonomy of Self-Supervised Methods for Event Sequences

### A. Contrastive Methods

#### CoLES (Contrastive Learning for Event Sequences)
- **Objective**: Pull subsequences from same entity together, push subsequences from different entities apart
- **How it works**: Split each entity's event sequence into K random slices (SampleSlices). Positive pairs = slices from same entity. Negative pairs = slices from different entities.
- **Loss**: Margin-based contrastive loss: L_CoLES = sum d^2(f(x), f(x+)) + sum max{0, rho - d(f(x), f(x-))}^2
- **Default params**: margin=0.5, HardNegativePairSelector(neg_count=5), SampleSlices(split_count=5, cnt_min=25, cnt_max=200)
- **Embedding extraction**: Last hidden state of GRU encoder (or mean pooling for Transformer)
- **Strengths**: Best overall method in EBES benchmark; excels on sequence classification tasks; robust across domains
- **Weaknesses**: Requires sufficiently long sequences for meaningful subsequence splitting

#### CPC (Contrastive Predictive Coding)
- **Objective**: Predict future latent representations from past context
- **How it works**: Encode past events into context vector via autoregressive model; predict future event representations K steps ahead; distinguish correct future from negatives
- **Loss**: InfoNCE loss
- **Strengths**: Good for temporal/sequential tasks (credit scoring); autoregressive nature aligns with prediction tasks
- **Weaknesses**: Inconsistent across datasets; sometimes worse than baseline

#### Barlow Twins / VICReg
- **Objective**: Learn representations with minimal redundancy — alignment + redundancy reduction
- **How it works**: Two augmented views of same sequence; align representations while decorrelating embedding dimensions
- **Strengths**: Avoids explicit negative sampling; consistently strong (#2-3 in benchmarks)
- **Weaknesses**: Less explored for event sequences specifically

### B. Predictive/Generative Methods

#### MLM (Masked Language Model / Masked Event Prediction)
- **Objective**: Reconstruct masked events from context
- **How it works**: Randomly mask 10-15% of events in a sequence; predict masked event embeddings using surrounding context
- **In PyTorch-Lifestream**: Uses contrastive loss in latent space (QuerySoftmaxLoss) rather than per-feature reconstruction
- **Params**: replace_proba=0.1, neg_count=64, loss_temperature=20.0
- **Architecture**: Typically uses Longformer/Transformer encoder (needs bidirectional context)
- **Embedding extraction**: Save pretrained trx_encoder weights, load into downstream GRU/Transformer with differential learning rates
- **Strengths**: Captures local transaction-level patterns; good for temporal point process tasks
- **Weaknesses**: Can underperform CoLES on some tasks; requires careful tuning

#### CMLM (Contrastive Masked Language Model)
- **Objective**: Predict masked event embeddings in latent space using contrastive loss
- **Loss**: L_CMLM = -sum_i log(exp(sim(r_i, r_hat_i)) / (exp(sim(r_i, r_hat_i)) + sum_j exp(sim(r_i, r_hat_j))))
- **Key insight**: Predicting in latent space avoids complexity of multi-feature reconstruction
- **Combined with CoLES**: L_total = L_CoLES + lambda * L_CMLM (lambda typically 0.005-0.05)

#### RTD (Replaced Token Detection — ELECTRA-style)
- **Objective**: Detect which events in a sequence have been replaced with plausible alternatives
- **How it works**: Small generator network produces replacement events; discriminator (main encoder) classifies each event as original vs. replaced
- **Strengths**: Every position provides training signal (not just masked ones); good for anomaly detection / credit scoring
- **Weaknesses**: Inconsistent results; worse than CoLES on most benchmarks

#### NSP (Next Sequence Prediction — BERT-style)
- **Objective**: Predict whether two subsequences follow each other in time
- **Weaknesses**: Generally weak performer; ranked low across benchmarks

#### SOP (Sequences Order Prediction — ALBERT-style)
- **Objective**: Predict the temporal ordering of two subsequences
- **Weaknesses**: Worst performer in nearly all benchmarks; not recommended

#### Next-Event Prediction (Autoregressive / GPT-style)
- **Objective**: Predict next event given history (causal ordering)
- **Strengths**: Natural fit for recommendation; captures temporal dynamics
- **Weaknesses**: Doesn't see future context; less effective for classification

### C. Hybrid Methods

#### MLEM (Multimodal Learning Event Model)
- **Objective**: Align generative and contrastive representations as distinct modalities
- **How it works**: 
  1. Train CoLES encoder first (frozen teacher)
  2. Train generative encoder (next-event prediction) with dual loss:
     - Reconstruction loss (cross-entropy for categorical, MSE for numerical/temporal)
     - Alignment loss (SigLIP) to match contrastive teacher embeddings
  3. Total loss: L = alpha * L_LM + beta * L_align (alpha=1, beta=10)
  4. Only generative encoder used for downstream
- **Why better than naive combination**: Avoids requiring subsequence sampling during generative training
- **Results**: Most universal method; performs at least on par with best single approach on every task
- **Key metrics**: Highest intrinsic dimension (15.86), lowest anisotropy (0.06)

---

## Benchmark Results

### EBES Benchmark (10 datasets, KDD 2025)

Overall ranking (CoLES > GRU > MLEM > Transformer > Mamba):

| Method | MBD (AUC) | Retail (Acc) | Age (Acc) | Taobao (AUC) |
|--------|-----------|-------------|----------|--------------|
| CoLES | 0.826 | 0.553 | 0.634 | 0.713 |
| GRU | 0.827 | 0.543 | 0.626 | 0.713 |
| MLEM | 0.824 | 0.544 | 0.634 | 0.713 |

When pretraining helps: Tasks where target is a characteristic of the observed sequence (Age, Retail)
When pretraining doesn't help much: Tasks where target relates to future events (MBD, Taobao, PhysioNet)

### PyTorch-Lifestream Experiments (6 datasets)

#### Unsupervised Embeddings + LightGBM

**Gender (AUROC)**:
- MLES2 (CoLES v2): 0.882
- MLES (CoLES): 0.881
- Baseline: 0.877
- Barlow Twins: 0.865
- RTD: 0.855
- NSP: 0.852
- CPC: 0.851
- SOP: 0.785

**Age (Accuracy)**:
- CoLES Transformer: 0.646
- MLES2: 0.643
- MLES: 0.640
- Barlow Twins: 0.634
- RTD: 0.631
- Baseline: 0.629
- NSP: 0.621
- CPC: 0.602

**Churn/Rosbank (AUROC)**:
- MLES: 0.841
- Barlow Twins: 0.839
- MLES2: 0.837
- NSP: 0.828
- Baseline: 0.827
- CPC: 0.792
- SOP: 0.780
- RTD: 0.771

**Scoring/Alpha Battle (AUROC)**:
- CoLES Transformer: 0.7968
- MLES: 0.7921
- CPC: 0.7919
- RTD: 0.7910
- Barlow Twins: 0.7878
- Baseline: 0.7792
- NSP: 0.7655
- SOP: 0.7238

#### Supervised Finetuned Encoders + MLP

**Gender (AUROC)**:
- MLES Finetuning: 0.879
- RTD Finetuning: 0.868
- Target Scores (supervised): 0.867
- CPC Finetuning: 0.865

**Age (Accuracy)**:
- CPC Finetuning: 0.625
- MLES Finetuning: 0.624
- RTD Finetuning: 0.622
- Target Scores: 0.620

### MLEM vs Individual Methods (Fine-tuning)

| Dataset | CoLES | Generative | MLEM | Supervised |
|---------|-------|-----------|------|-----------|
| ABank (AUC) | 0.785 | 0.788 | 0.790 | lower |
| Age (Acc) | 0.635 | 0.638 | 0.642 | lower |
| TaoBao (AUC) | 0.690 | 0.695 | 0.695 | lower |

Self-supervised pretraining outperformed supervised baseline across ALL datasets in this study.

### CMLM+CoLES Combined (4 financial datasets)

| Dataset | CoLES | CMLM | CMLM+CoLES (best lambda) | Improvement |
|---------|-------|------|-------------------------|-------------|
| Churn (AUC) | 0.770 | 0.762 | 0.784 | +1.8% |
| Gender (AUC) | 0.856 | 0.845 | 0.856 | tied |
| Age (AUC) | 0.852 | 0.849 | 0.852 | tied |
| DataFusion (AUC) | 0.726 | 0.728 | 0.732 | +0.8% |

CMLM+CoLES (lambda=0.01) achieves best overall ranking: 1.5 global, 1.25 local.

---

## Key Training Recipes

### Recipe 1: CoLES with PyTorch-Lifestream

```python
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule, ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
import pytorch_lightning as pl
import torch
from functools import partial

# 1. Transaction encoder
trx_encoder = TrxEncoder(
    embeddings_noise=0.003,
    numeric_values={"amount": "identity"},
    embeddings={
        "event_date": {"in": 800, "out": 16},
        "event_type": {"in": 250, "out": 16},
    },
)

# 2. Sequence encoder (GRU recommended by EBES benchmark)
seq_encoder = RnnSeqEncoder(
    trx_encoder=trx_encoder,
    hidden_size=256,
    type="gru",
)

# 3. CoLES module
model = CoLESModule(
    seq_encoder=seq_encoder,
    optimizer_partial=partial(torch.optim.Adam, lr=0.001),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9),
)

# 4. Data with subsequence splitting for contrastive pairs
train_dl = PtlsDataModule(
    train_data=ColesDataset(
        MemoryMapDataset(data=train, i_filters=[SeqLenFilter(min_seq_len=25)]),
        splitter=SampleSlices(split_count=5, cnt_min=25, cnt_max=200),
    ),
    train_num_workers=16,
    train_batch_size=256,
)

# 5. Train
trainer = pl.Trainer(max_epochs=12, accelerator="auto")
trainer.fit(model, train_dl)

# 6. Extract embeddings
from ptls.data_load.datasets import inference_data_loader
dl = inference_data_loader(data, num_workers=0, batch_size=256)
embeddings = torch.vstack(trainer.predict(model, dl))
```

### Recipe 2: MLM Pretraining with PyTorch-Lifestream

```python
from ptls.frames.bert import MlmDataset, MLMPretrainModule
from ptls.nn import TrxEncoder, LongformerEncoder, PBLinear, PBL2Norm

# 1. Transaction encoder + projection
trx_encoder = TrxEncoder(
    embeddings_noise=0.003,
    numeric_values={"amount": "identity"},
    embeddings={"event_date": {"in": 800, "out": 16}, "event_type": {"in": 250, "out": 16}},
)
trx_proj = torch.nn.Sequential(trx_encoder, PBLinear(trx_encoder.output_size, 64), PBL2Norm())

# 2. MLM module with Longformer encoder
mlm_module = MLMPretrainModule(
    trx_encoder=trx_proj,
    seq_encoder=LongformerEncoder(
        input_size=64, num_attention_heads=1, intermediate_size=256,
        num_hidden_layers=2, attention_window=32, max_position_embeddings=2000,
    ),
    hidden_size=64,
    loss_temperature=20.0,
    total_steps=30000,
    replace_proba=0.1,  # mask 10% of events
    neg_count=64,
)

# 3. Data
mlm_dm = PtlsDataModule(
    train_data=MlmDataset(MemoryMapDataset(data=train), min_len=100, max_len=128),
    train_batch_size=128,
)

# 4. Train
trainer = pl.Trainer(max_steps=1200)
trainer.fit(mlm_module, mlm_dm)

# 5. Save pretrained weights for downstream transfer
torch.save(mlm_module.trx_encoder.state_dict(), "mlm-pretrained.pt")

# 6. Transfer to downstream task with differential learning rates
from ptls.nn import Head
downstream = SequenceToTarget(
    seq_encoder=RnnSeqEncoder(trx_encoder=trx_proj, input_size=64, hidden_size=64, type="gru"),
    head=Head(input_size=64, use_batch_norm=True, objective="classification", num_classes=4),
    pretrained_lr=0.005,  # lower LR for pretrained layers
    optimizer_partial=partial(torch.optim.Adam, lr=0.015),  # higher LR for head
)
downstream.seq_encoder.trx_encoder[0].load_state_dict(torch.load("mlm-pretrained.pt"))
```

---

## When to Use Self-Supervised vs. Supervised

### Self-supervised pretraining is better when:
- **Limited labels**: Few labeled examples relative to total data (semi-supervised setting)
- **Many downstream tasks**: Same embedding serves multiple use cases
- **Target is a sequence characteristic**: Age prediction, demographic classification, behavioral clustering
- **You want transferable representations**: Pretrain once, use everywhere

### Supervised multi-task is better when:
- **Abundant labels**: Many labeled examples for well-defined tasks
- **Task-specific optimization needed**: You need maximum performance on specific targets
- **Tasks are closely related**: Shared encoder benefits from multi-task regularization
- **Target relates to future events**: Churn prediction, default scoring (pretraining helps less here)

### Hybrid approach (best of both worlds):
1. Pretrain with CoLES or MLEM on all data (labeled + unlabeled)
2. Fine-tune with task-specific heads on labeled data
3. Use differential learning rates (lower for pretrained encoder, higher for heads)

Self-supervised finetuning consistently matches or beats pure supervised training in MLEM benchmarks. The EBES benchmark shows the gap is small but consistent.
