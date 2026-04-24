# Methodology: Sarvam-Assisted Timestamped Alignment

This document outlines the "Gold Standard" strategy for segmenting conversational Indic ASR data using **Sarvam (Saaras:v3)** for temporal alignment while preserving **Human Ground Truth** for training labels.

## 1. The Core Principle
We separate the **"Where"** from the **"What"**:
- **Sarvam (`saaras:v3`)**: Used exclusively to find the start and end timestamps of speech segments and identify speakers.
- **Human Transcript**: Used exclusively as the source of truth for the text labels.

**Crucial Rule**: We NEVER train on Sarvam-generated text. We only use Sarvam to find the "time-boxes" for our ground truth text.

---

## 2. Alignment Pipeline

### Step 1: Pre-Processing
- Convert raw call recordings to 16kHz Mono WAV.
- Load the interaction transcripts (JSON format) from the source inventory.

### Step 2: Temporal Mapping (Sarvam)
We send the audio to the Sarvam API/Service.
- **Input**: Full call audio.
- **Output**: A JSON list of segments with `start`, `end`, `speaker_label`, and `raw_transcript`.
  ```json
  {
    "start": 2.15,
    "end": 8.40,
    "speaker": "A",
    "text": "(Discarded)" 
  }
  ```

### Step 3: Text Reconciliation
We map our **Human Ground Truth Turns** to the **Sarvam Timestamps**:
1. Take the sequence of Human turns (Turn 1, Turn 2...).
2. Take the sequence of Sarvam time-segments (Segment 1, Segment 2...).
3. Align them using a **Best-Fit Window**:
   - We look for the time-segment that most likely matches our turn text.
   - We verify the speaker role (Agent/User) matches the Sarvam label.

### Step 4: Audio Segmentation
- Cut the original audio according to the Sarvam `start` and `end` times.
- Buffer the edges (e.g., +200ms) to ensure no words are clipped.

### Step 5: Data Filtering & Quality Control
We keep only "High-Confidence" segments:
- **Duration**: Target 1–15 seconds segments.
- **Role Consistency**: Ensure the speaker identified by Sarvam matches our ground truth role.
- **Overlap Check**: Discard segments with significant crosstalk to maintain training data purity.

---

## 3. Why This Wins
1. **Precision**: Sarvam's diarization is highly accurate at finding word boundaries.
2. **Safety**: No "feedback loops" because the training labels come from humans, not the AI.
3. **Efficiency**: Resolves the "Gateway Overload" issue by moving to a batch alignment approach (or using a more robust endpoint).

## 4. Final Training Format
The output is a standard NeMo manifest (`train_manifest.jsonl`):
```json
{
  "audio_filepath": "/path/to/segment.wav",
  "text": "नमस्ते, मैं राधा एसएमएफजी इंडिया क्रेडिट से बोल रही हूँ",
  "duration": 5.25,
  "lang": "hi",
  "original_call_id": "call_123",
  "speaker_role": "agent"
}
```
