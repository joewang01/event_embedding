---
source: Web research (multiple sources)
author: Various
date: 2026-04-07
type: research
keywords: heterogeneous events, multi-schema, type-specific encoders, cross-type attention, unified embedding space, multi-source records
---

# Research: Heterogeneous Event Types — Multi-Schema Event Sequence Embedding

## Research Question

How to embed event sequences where events come from VERY DIFFERENT sources (inquiry records, property assessments, home median values, student files) — each with completely different schemas — into ONE shared embedding per entity.

## Source 1: HE-LSTM — Heterogeneous Event LSTM (AAAI 2020)

**Paper**: "Learning the Joint Representation of Heterogeneous Temporal Events for Clinical Endpoint Prediction" (arXiv 1803.04837)
**URL**: https://ar5iv.labs.arxiv.org/html/1803.04837

### Architecture

Each clinical event = triple: (type, value, time). Three separate encoding mechanisms:

1. **Event Type Embedding**: One-hot type → embedding matrix → s = C_type × type
2. **Categorical attributes**: One-hot → lookup matrix V_c
3. **Numerical attributes**: Raw values → tanh(V_n × value_n)

Combined: x = s + V_c × value_c + tanh(V_n × value_n)

### Event Gate Innovation

Event gate (j_s,t) controls which events update which memory cells:
- Event Filter (e_s): neural net determining event type relevance → σ(W_em × tanh(W_ms × s + b_m) + b_e)
- Phase Gate (k_t): learned oscillating function for temporal sampling
- Update rule: c_l = j_l ⊙ ~c_l + (1 - j_l) ⊙ c_{l-1}

Allows different LSTM neurons to "specialize" in tracking specific event clusters.

### Results

MIMIC-III: Death prediction 0.9516 AUC (vs 0.9466 LSTM+embeddings), Lab test 0.7987 AUC (vs 0.7231)

---

## Source 2: MEME — Multiple Embedding Model for EHR (2024)

**Paper**: "Emergency Department Decision Support using Clinical Pseudo-notes"
**URL**: https://arxiv.org/html/2402.00160v2
**Code**: https://github.com/Simonlee711/MEME

### Core Idea: Multi-Stream Separate Encoding + Self-Attention Fusion

Six distinct EHR modalities processed SEPARATELY:
- Arrival information (demographics, arrival mode)
- Triage (vitals, complaints)
- Medication reconciliation
- Diagnostic codes (ICD-9/10)
- Measurements (vitals during stay)
- Pyxis medications

### Architecture Steps

1. **Pseudo-Notes Conversion**: Each modality → template-based text serialization (not LLM-generated — avoids hallucination). Missing data gets placeholder sentences.
2. **Independent Encoding**: Each pseudo-note → MedBERT tokenizer → MedBERT encoder → dense vector v_i
3. **Concatenation**: V_concat = Concatenate(v_1, v_2, ..., v_n)
4. **Self-Attention Fusion**: V_attention = SelfAttention(V_concat)
5. **Classification**: V_fc = ReLU(FC(V_attention)) → z = Classifier(V_fc)

### Why This Works for Different Schemas

- Text serialization preserves semantic relationships between categorical values
- Foundation models capture implicit hierarchies in terminology
- Separate encoding streams prevent feature conflicts
- Self-attention learns stream-specific importance weights
- Gracefully handles missing modalities (placeholder text)
- No rigid standardization needed

### Training Config

Batch 64, dropout 0.3, AdamW (lr 5e-5), Tesla V100, PyTorch + HuggingFace

---

## Source 3: Hi-BEHRT — Hierarchical BERT for EHR (2021)

**Paper**: "Hi-BEHRT: Hierarchical Transformer-Based Model for Accurate Prediction of Clinical Events Using Multimodal Longitudinal Electronic Health Records"
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC7615082/

### Handling 8 Modalities with Different Schemas

Modalities: diagnoses (ICD-10), medications (BNF), procedures (OPCS), tests (Read codes), blood pressure, BMI, drinking status, smoking status

### Unification Strategy: Everything → Token Vocabulary

- Each coding system (ICD-10, BNF, OPCS, Read) maps to its own token vocabulary space
- Continuous measures binned into categorical: BP → 5mmHg steps, BMI → 1 kg/m² steps
- All records ordered by timestamp into one long sequence

### Four Embedding Types (summed)

1. **Token embeddings** — from diagnosis/medication/procedure codes
2. **Age embeddings** — patient age at event time
3. **Segmentation embeddings** — alternating 0/1 across visits
4. **Position embeddings** — monotonically increasing

embedding = token_emb + age_emb + segment_emb + position_emb

### Hierarchical Transformer

- **Local**: Transformer processes sliding windows (size=50, stride=30)
- **Global**: Second Transformer summarizes across all segments
- Handles sequences up to 1220 tokens (97% of patients)

### Key Insight

No explicit modality embeddings needed — different coding systems create implicitly separate vocabulary subspaces. The shared transformer learns cross-type interactions through self-attention.

---

## Source 4: DescEmb — Text-Based Code Embedding (CHIL 2022)

**Paper**: "Unifying Heterogeneous Electronic Health Records Systems via Text-Based Code Embedding"
**URL**: https://proceedings.mlr.press/v174/hur22a.html
**Code**: https://github.com/hoon9405/DescEmb

### Core Problem

Different hospital EHR systems use completely different coding systems. Same clinical concept = different codes.

### Solution: Embed Clinical Descriptions, Not Codes

Instead of code → embedding lookup, use: code → text description → language model → embedding.
Enables cross-institution transfer since different codes for the same concept have similar descriptions.

### Architecture

- BERT-based and RNN-based variants
- Pretrained with MLM on clinical descriptions
- Supports transfer learning and pooled learning across institutions
- Outperforms code-based approaches in cross-institutional settings

---

## Source 5: Hierarchical Clinical Event Representation (KDD 2020)

**Paper**: "Learning Hierarchical Representations of Electronic Health Records for Clinical Outcome Prediction"
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC7153073/

### Event Type Encoding

Each event gets: v_t = type_embedding + attribute_encoding
- Handles: chart events, input events, lab events, procedure events, output events
- All embedded as vectors v_t ∈ ℝ^N through summation of type and attribute components

### Hierarchical Architecture

1. **Event Groups**: Temporal segmentation into groups based on time gaps
2. **Event Attention**: MLP generates importance weights within groups
   - q_t^i = w_q × tanh(W_e × v_t^i + W_h × h_{i-1} + b_h)
   - α_t^i = softmax(q_1^i, ..., q_n^i)
   - g_i = Σ(α_t^i × v_t^i)
3. **Group-level GRU**: Processes group representations
4. **Temporal Attention**: Identifies critical clinical phases

---

## Source 6: PyTorch Frame — Multi-Modal Tabular Learning

**URL**: https://arxiv.org/html/2404.00776v2

### TensorFrame: Handling Different Column Types

Groups columns by semantic type: numerical, categorical, multicategorical, timestamp, text_embedding, text_tokenized, embedding

### Four-Stage Pipeline

1. **Materialization**: Raw values → type-specific tensors, compute normalization stats
2. **Encoding**: Each column → F-dimensional vector independently. Within each type, parallel embedding → concat across types → X of shape [N, C, F]
3. **Column-wise Interaction**: Message-passing updates across column embeddings (L layers)
4. **Decoding**: Summarize into row embeddings Z

### Key Insight

All heterogeneous columns converge to unified [N, C, F] tensor — N rows, C columns, F shared embedding dim.

---

## Source 7: PyTorch-Lifestream TrxEncoder

**URL**: https://pytorch-lifestream.github.io/pytorch-lifestream/nn/trx_encoder/

### Configuration

```python
model = TrxEncoder(
    embeddings={
        'mcc_code': {'in': 10, 'out': 6},
        'currency': {'in': 4, 'out': 2},
    },
    numeric_values={'amount': 'identity'},
)
# output_size = 6 + 2 + 1 = 9
```

Categorical features → nn.Embedding → concat with numerical features (scaled) → output vector

---

## Architecture Pattern Synthesis

### Pattern A: Type-Specific Encoders + Projection to Shared Space

Each event type gets its own encoder (handles its unique schema), then projects to common dimension:

```
Inquiry events  → InquiryEncoder(cat+num)  → [B, d_inq]  → Linear → [B, D]
Property events → PropertyEncoder(cat+num) → [B, d_prop] → Linear → [B, D]
Student events  → StudentEncoder(cat+num)  → [B, d_stud] → Linear → [B, D]
```

Then interleave in temporal order and feed to shared sequence encoder.

### Pattern B: Unified Token Vocabulary + Type Embedding (BEHRT-style)

Map all event fields to one big token vocabulary. Add a learned type embedding:

token = event_field_emb + type_emb + position_emb + time_emb

### Pattern C: Multi-Stream Encoding + Cross-Attention Fusion (MEME-style)

Process each event type's sequence independently through separate encoders, then fuse with self-attention:

```
seq_inquiry  → Encoder_A → emb_A
seq_property → Encoder_B → emb_B
seq_student  → Encoder_C → emb_C
fused = SelfAttention(stack(emb_A, emb_B, emb_C))
```

### Pattern D: Per-Event Encode + Shared Sequence (HE-LSTM style)

Encode each event with type_emb + attribute_encoding, gate the LSTM updates by event type.
