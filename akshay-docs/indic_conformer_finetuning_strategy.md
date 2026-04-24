# Indic Conformer Fine-tuning Strategy & Research Context

This document serves as the central context for fine-tuning the **AI4Bharat Indic Conformer (600M)** model. It synthesizes insights from the **IndicVoices** research paper, lessons learned from initial fine-tuning regressions, and the optimized strategy for stable adaptation.

---

## 📄 Research Context: IndicVoices (arXiv:2403.01926)

The foundation of this project is the **IndicVoices** dataset and the resulting **IndicASR** models (specifically the Indic Conformer architecture).

### Key Insights from the Paper:
- **Scale & Diversity:** IndicVoices provides ~7.3k hours of speech across 22 Indian languages, featuring 16k+ speakers from 145 districts. It captures a mix of read, extempore, and conversational speech.
- **Architecture:** The model is a **Hybrid CTC-RNNT Conformer**.
  - **CTC Branch:** Provides alignment and fast greedy decoding.
  - **RNNT Branch:** Provides better sequence modeling and handles deletion/insertion errors more robustly.
- **Inclusive Design:** The dataset is built to be inclusive of rural dialects and demographic variations, making the base model a "generalist" that understands the broad phonetic space of Indian languages.

> [!NOTE]
> The base model’s strength lies in its **general acoustic knowledge**. Fine-tuning must preserve this "broad phonetic understanding" while specializing in a specific domain or language.

---

## 📉 Failure Analysis: The "WER Regression" Case Study

During initial fine-tuning attempts on a small Hindi dataset (~1.6k–4k samples), the Word Error Rate (WER) regressed significantly from **~15% to ~54%**.

### Root Causes Identified:
1. **Catastrophic Forgetting (Primary):** By training *only* on the Hindi subset, the model lost its multilingual signal and general acoustic stability.
2. **Over-specialization on Small Data:** With only ~1600 samples, the model learned dataset-specific noise and speaker idiosyncrasies rather than general language features.
3. **Improper Warmup:** A warmup of 1000/1800 steps (~55% of the total run) delayed learning too long, followed by sudden, unstable updates.
4. **Visibility Gap:** Using `--disable-validation` hid the degradation until the run was complete.

---

## 🚀 Optimized Fine-tuning Strategy

To stabilize the model and achieve positive delta improvements, follow this multi-phase approach.

### 1. Data Strategy (The "Multilingual Anchor")
**Never fine-tune on a single language.**
- **Distribution:** Maintain a **50/50 mix** (50% Target Language (Hindi) / 50% General Multilingual Data).
- **Sampling:** Use temperature sampling ($\alpha \approx 0.5$) only to balance the *other* languages, keeping the target language at a fixed high priority.

### 2. Model Freezing (Architectural Guardrails)
Lower layers of a Conformer capture universal speech features (edges, phonemes), while upper layers capture language-specific context.
- **Recommendation:** Freeze **70–80% of lower encoder layers**.
- **Focus:** Train only the top layers and the decoder.

### 3. Hyperparameter Configuration
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Learning Rate** | `1e-6` | Extremely conservative to prevent weight "explosion" on small data. |
| **Warmup Steps** | `50–100` | Rapidly move out of the initial state without overshooting. |
| **Effective Batch Size** | `8` | Balanced via `batch_size=2` and `grad_accum=4`. |
| **Gradient Clipping** | `1.0` | Prevents unstable updates from outliers. |

### 4. Training Control
- **Cycles:** For 4000 samples, `steps_per_epoch` is ~500.
- **Initial Run:** Start with `max_steps = 200–300` (less than one full epoch) to verify stability.
- **Validation:** Always monitor two sets:
  - **Set A (Target):** To check intended improvement.
  - **Set B (Multilingual):** To ensure no regression in general knowledge.

---

## 🧠 Mental Model for ASR Adaptation

### "Steering, Not Rebuilding"
Fine-tuning is a **steering** exercise. You are nudging a highly sophisticated, multi-lingual engine to favor a specific dialect or domain. If you push too hard (high LR, too many steps, no anchor data), you break the engine.

### Data Importance Hierarchy:
1. **Data Distribution:** (Multilingual anchor is the #1 stability factor).
2. **Learning Rate:** (Low is safe).
3. **Training Duration:** (Stop early and often).
4. **Layer Freezing:** (Protects the foundations).

---

## 🛠 Workflow Implementation

### Phase 1: Stability Check
- 4k samples (50% HI / 50% Multi)
- 80% layers frozen
- LR 1e-6, 300 steps.

### Phase 2: Refinement
- If stable, increase steps to 600.
- Reduce freezing to 60% if the model is too rigid.

### Phase 3: Scaling
- Gradually add more languages or domain-specific data.
- Apply temperature sampling across all categories.

---

> [!TIP]
> **One-line Summary:** Don't make the model smarter — make it adapt safely.
