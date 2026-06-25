# ALBEF + Momentum ID Memory Bank — Architecture

## Overview

This architecture replaces RaSa's FIFO queue with a full-dataset memory bank. Every training sample has a persistent slot in the bank, EMA-updated each time the sample appears in a batch. Contrastive loss is computed against the entire dataset at every step.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ALBEF + Momentum ID Memory Bank                          │
└─────────────────────────────────────────────────────────────────────────────────┘

                    TRAINING FORWARD PASS
                    =====================

    Image                                         Text
      │                                             │
      ▼                                             ▼
┌──────────────┐                           ┌──────────────────┐
│ ViT-B/16     │                           │ BERT (text mode) │
│ Image Encoder│                           │ Text Encoder     │
└──────┬───────┘                           └────────┬─────────┘
       │                                            │
       │ image_embeds [B, 577, 768]                 │ text_embeds [B, L, 768]
       │                                            │
       ├─── CLS token ──► vision_proj ──► image_feat [B, 256] (normalized)
       │                                            │
       │                   text_embeds[:,0,:] ──► text_proj ──► text_feat [B, 256] (normalized)
       │                                            │
       │                                            │
       ▼                                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MEMORY BANK (Full Dataset)                       │
│                                                                     │
│   image_bank: [N, 256]   ◄── EMA update at idx positions            │
│   text_bank:  [N, 256]   ◄── EMA update at idx positions            │
│   id_bank:    [N]        (person IDs for each slot)                 │
│                                                                     │
│   N = total training samples (e.g., 34,674 for ICFG-PEDES)         │
│                                                                     │
│   Update rule (per sample at position idx):                         │
│     bank[idx] = α × bank[idx] + (1 - α) × new_feat                 │
│     α = 0.9 (momentum)                                             │
│                                                                     │
│   Contrastive Loss (computed against ALL N samples):                │
│     sim_i2t = image_feat @ text_bank.T / τ       [B, N]            │
│     sim_t2i = text_feat  @ image_bank.T / τ      [B, N]            │
│     sim_i2i = image_feat @ image_bank.T / τ      [B, N]            │
│     sim_t2t = text_feat  @ text_bank.T / τ       [B, N]            │
│                                                                     │
│     targets = same_person_id soft labels                            │
│     loss_cl = avg(CE(sim_i2t, targets), CE(sim_t2i, targets),       │
│                   CE(sim_i2i, targets), CE(sim_t2t, targets))       │
└─────────────────────────────────────────────────────────────────────┘
       │                                            │
       │                                            │
       ▼                                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CROSS-ENCODER (BERT fusion mode)                  │
│                                                                     │
│   Input: text_embeds + image_embeds (cross-attention)               │
│                                                                     │
│   Positive pairs ──► ITM head ──► loss_pitm (match/no-match)        │
│   Negative pairs (hard negatives mined from memory bank sims)       │
│                                                                     │
│   Positive output ──► PRD head ──► loss_prd (replaced caption?)     │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MOMENTUM ENCODER (for MLM/MRTD only)                    │
│                                                                     │
│   ViT_m (momentum copy of ViT) ──► image_embeds_m                   │
│   BERT_m (momentum copy of BERT) ──► soft prediction labels         │
│                                                                     │
│   Param update: θ_m = 0.995 × θ_m + 0.005 × θ                      │
│                                                                     │
│   MLM: mask text tokens, use momentum soft labels ──► loss_mlm      │
│   MRTD: replace tokens via momentum, detect them ──► loss_mrtd      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Loss Functions

| Loss | Weight | Description |
|------|--------|-------------|
| `loss_cl` | 0.5 | Global contrastive loss via memory bank (i2t, t2i, i2i, t2t) |
| `loss_pitm` | 1.0 | Probabilistic Image-Text Matching (hard negative mining) |
| `loss_mlm` | 1.0 | Masked Language Modeling with momentum soft labels |
| `loss_prd` | 0.5 | Positive Relation Detection (was caption replaced?) |
| `loss_mrtd` | 0.5 | Momentum-based Replaced Token Detection |

**Total loss:**
```
L = 0.5×L_cl + 1.0×L_pitm + 1.0×L_mlm + 0.5×L_prd + 0.5×L_mrtd
```

---

## Memory Bank Initialization

Before training begins, all training samples are forwarded through the encoders once (no gradients) to populate the memory bank with meaningful initial representations.

```
for each (image, text) in training_set:
    image_feat = normalize(vision_proj(ViT(image)[:, 0, :]))
    text_feat  = normalize(text_proj(BERT(text)[:, 0, :]))
    bank.image_bank[idx] = image_feat
    bank.text_bank[idx]  = text_feat
    bank.id_bank[idx]    = person_id
```

---

## Key Difference from Original RaSa

| Aspect | Original RaSa | Memory Bank Version |
|--------|--------------|---------------------|
| Storage | FIFO queue [65536, 256] | Fixed bank [N, 256] per sample |
| Update | Push new, pop oldest | EMA at specific position |
| Coverage | Only recent batches | Entire dataset always |
| Positives | Same ID found in queue | All same-ID samples in bank |
| Negatives | Queue entries | All different-ID samples |
| Momentum encoder for CL | Yes (produces queue features) | No (bank handles consistency) |
| Momentum encoder for MLM | Yes | Yes (unchanged) |

---

## Memory Requirements

| Component | Size |
|-----------|------|
| image_bank [34674, 256] | ~34 MB |
| text_bank [34674, 256] | ~34 MB |
| id_bank [34674] | ~0.1 MB |
| **Total memory bank** | **~68 MB** |

Negligible compared to model parameters (~3GB) and activations (~12GB).

---

## Hyperparameters

| Param | Value | Description |
|-------|-------|-------------|
| `num_train_samples` | 34674 | Number of slots in memory bank |
| `bank_momentum` | 0.9 | EMA coefficient (90% old + 10% new) |
| `temp` | 0.07 | Temperature for contrastive loss |
| `momentum` | 0.995 | Momentum encoder update rate |
| `embed_dim` | 256 | Feature dimension in bank |
