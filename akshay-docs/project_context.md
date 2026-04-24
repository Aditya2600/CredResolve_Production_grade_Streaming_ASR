# 🎯 Project Context: Fine-tuning AI4Bharat Indic Conformer

## 🧠 Problem Statement
*   **Goal**: Fine-tuning a pretrained **Indic Conformer (multilingual ASR)** model (600M parameters).
*   **Base Model**: Trained on ~800+ hours of multilingual data.
*   **Dataset**: Current small dataset (1.6k–4k recordings).
*   **Objective**:
    *   Improve **Hindi ASR performance**.
    *   Expand to **multilingual support** (22+ Indian languages).

## 🚨 Root Causes Identified
### 1. Catastrophic Forgetting
*   Fine-tuned on **only Hindi dataset**.
*   Model lost multilingual + general acoustic knowledge.
*   Outcome: WER degraded from **~15% → ~54%** after fine-tuning.

### 2. Normalization Issues
*   Mismatches between training and validation normalization.
*   Formatting inconsistencies in transcripts.

## 🛠️ Corrected Pipeline (Turn-Level Segmentation)
To ensure high data quality and avoid model instability (WER 50%+), we use the following corrected pipeline:

1.  **Parse CSV**: Extract audio URL + transcript JSON from `Result_7.csv`, `Result_11.csv`, `Result_23.csv`, `Result_28.csv`.
2.  **Download Audio**: Fetch source recordings.
3.  **Extract Speaker Turns**: Segment by turns (Role mapping: Agent vs Borrower).
4.  **Speaker Filtering**: Focus on one speaker at a time (Agent OR Borrower).
5. **Segment Audio Per Turn**: Use **Sequential Alignment** or forced alignment (NOT VAD splitting).
6. **VAD Trimming**: Use VAD only as a cleaner to remove silence and trim edges.
7. **Filter Duration**: Only keep segments between **1.0s and 20.0s**.
8. **Text Normalization**: Normalize punctuation, nuktas, spacing, and remove noise words.
9. **Final Manifest**: NeMo-compatible JSON manifest.

> [!WARNING]
> ### ⚠️ BIGGEST WARNING
> **Wrong audio-text alignment = model destruction.**
> This can cause WER 50%+, unstable outputs, and vowel dropping. ASR performance depends more on alignment than model architecture.

## 📋 Segment Rules
- **min_duration**: 1.0 sec
- **max_duration**: 15–20 sec
- **Reject**: `< 1 sec`, `empty text`.

## 📋 Target Languages
Supported languages:
- Assamese (as), Bengali (bn), Bodo (brx), Dogri (doi), Gujarati (gu), Hindi (hi), Kannada (kn), Konkani (kok/gom), Kashmiri (ks), Maithili (mai), Malayalam (ml), Marathi (mr), Manipuri (mni), Nepali (ne), Oriya (or), Punjabi (pa), Sanskrit (sa), Santali (sat), Sindhi (sd), Tamil (ta), Telugu (te), Urdu (ur).
