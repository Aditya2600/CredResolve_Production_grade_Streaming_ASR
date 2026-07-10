# Diarization Pipeline

> **Scope:** Offline speaker diarization and speakerization pipeline. This document details the process of taking normalized ASR inventories and audio, executing offline speaker diarization via NVIDIA NeMo, annotating acoustic segments with speaker labels, and performing word-level speaker assignment using downstream ASR worker timestamps.

---

## 1. Architecture Overview

Diarization is used to split audio into speaker turns. It is executed **offline** to segment and label training data rather than in the live, real-time WebSocket inference path. 

The pipeline combines a VAD engine (Silero VAD), a deep-learning speaker diarization engine (NVIDIA NeMo ClusteringDiarizer), and an optional ASR alignment engine (GPU worker) to produce speakerized manifests.

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. INPUT INVENTORY (data_inventory.csv) & AUDIO FILES                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. AUDIO PREPROCESSING (tools/diarization/audio.py)                    │
│    - Select mono, channel, or downmixed stereo channels                │
│    - Pad channels to equal length, downmix, and resample to 16 kHz     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. NeMo DIARIZER (tools/diarization/providers/nemo_telephony.py)       │
│    - Construct ClusteringDiarizer configuration                       │
│    - VAD (MarblerNet) + Speaker Embeddings (TitaNet)                   │
│    - Multi-Scale Diarization Decoder (MSDD) + Spectral Clustering      │
│    - Output RTTM file -> Parse & mark turns + overlaps                 │
└────────────────┬──────────────────────────────────────┬────────────────┘
                 │                                      │
                 ▼ (If no timestamps needed)            ▼ (If timestamps requested)
┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│ 4A. SEGMENT-LEVEL ATTRIBUTION         │ │ 4B. WORD-LEVEL SPEAKERIZATION│
│    - Intersect ASR segments with RTTM │ │    - Query ASR worker HTTP   │
│    - Assign winner by max overlap     │ │    - Run interval overlap per│
│    - Calculate speaker confidence     │ │      word timestamp          │
└────────────────┬──────────────────────┘ └─────────────┬────────────────┘
                 │                                      │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. OUTPUT DATA PRODUCTS (outputs/results_all/diarization/)             │
│    - data_inventory_diarized.csv/parquet                               │
│    - diarization_turns.jsonl (Raw turns)                               │
│    - speakerized_transcript.jsonl (UTT/Word labels)                    │
│    - pred_rttms/*.rttm (Standard NIST RTTM files)                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Layout

The diarization implementation spans the following primary scripts and files:

*   **Entrypoint Script:** [`tools/diarize_inventory.py`](../tools/diarize_inventory.py) — The main runner CLI.
*   **Orchestrator Script:** [`tools/build_diarized_training_data.py`](../tools/build_diarized_training_data.py) — Filters inventories and handles turn-reusing segmentation runs.
*   **Provider Implementation:** [`tools/diarization/providers/nemo_telephony.py`](../tools/diarization/providers/nemo_telephony.py) — Wraps NeMo's pipeline.
*   **Audio Helpers:** [`tools/diarization/audio.py`](../tools/diarization/audio.py) — Extracts and prepares diarization-compatible mono audio.
*   **Merge Logic:** [`tools/diarization/merge.py`](../tools/diarization/merge.py) — Annotates segments and individual words with speaker labels.
*   **Configuration Schema:** [`tools/diarization/config.py`](../tools/diarization/config.py) — Defines defaults for the VAD and Clustering models.

---

## 3. Audio Selection & Preprocessing

**File:** [`tools/diarization/audio.py`](../tools/diarization/audio.py)

Diarization requires mono audio normalized to a fixed sample rate (typically **16 kHz**). Depending on the `--audio-mode` parameter, the system extracts the source audio using different strategies:

### 3.1 Audio Modes

| Mode | Behavior |
|---|---|
| `auto` | Prefer `wav_audio_path` if available; fallback to `local_audio_path` if it is a WAV; otherwise extract and downmix stereo channels. |
| `mono` | Use the mono version of the source WAV file. |
| `channel0` / `channel1` | Use the explicit split channel WAV files directly. |
| `borrower` / `agent` | Match the lender-audit borrower channel columns and extract the target speaker's channel. |

### 3.2 Stereo Downmixing
When processing dual-channel stereo calls (which are common in telephony datasets), the script downmixes both channels to a single mono stream. This ensures both parties are present in a single track for the speaker diarizer.

```python
# tools/diarization/audio.py L77-87
mono_0 = to_mono(np.asarray(waveform_0, dtype=np.float32))
mono_1 = to_mono(np.asarray(waveform_1, dtype=np.float32))

# Resample channels to 16 kHz
if sample_rate_0 != target_sample_rate:
    mono_0 = resample_linear(mono_0, int(sample_rate_0), int(target_sample_rate))
if sample_rate_1 != target_sample_rate:
    mono_1 = resample_linear(mono_1, int(sample_rate_1), int(target_sample_rate))

# Zero-pad shorter channel to match the longer channel
max_length = max(len(mono_0), len(mono_1))
if len(mono_0) < max_length:
    mono_0 = np.pad(mono_0, (0, max_length - len(mono_0)))
if len(mono_1) < max_length:
    mono_1 = np.pad(mono_1, (0, max_length - len(mono_1)))

# Downmix by averaging
downmixed = ((mono_0 + mono_1) / 2.0).astype(np.float32, copy=False)
```

---

## 4. NeMo Telephony Diarizer Configuration

**File:** [`tools/diarization/providers/nemo_telephony.py`](../tools/diarization/providers/nemo_telephony.py)

The provider implements the `NeMoTelephonyDiarizationProvider` class, which constructs an OmegaConf schema and passes it to NeMo's `ClusteringDiarizer`.

### 4.1 Underlying Neural Models

The diarization configuration orchestrates three separate models:
1.  **VAD (Voice Activity Detection):** `vad_multilingual_marblenet` — A neural VAD trained on multilingual speech.
2.  **Speaker Embeddings:** `titanet_large` — A neural network model that extracts speaker-discriminative embeddings.
3.  **MSDD (Multi-Scale Diarization Decoder):** `diar_msdd_telephonic` — Captures temporal speaker interactions on multiple scales to refine boundaries.

### 4.2 Multi-Scale Embedding Windows
The diarizer extracts speaker embeddings using multiple window sizes to balance temporal resolution with embedding robustness:
*   **Windows:** `(1.5s, 1.25s, 1.0s, 0.75s, 0.5s)`
*   **Hops:** `(0.75s, 0.625s, 0.5s, 0.375s, 0.25s)`

### 4.3 Spectral Clustering & Thresholds
Diarization clustering can estimate the speaker count dynamically or use a fixed number:
*   **Oracle mode:** Controlled by `--speaker-count-mode`. In `fixed` mode, it uses `--fixed-speakers` (e.g., 2). In `estimate` mode, it dynamically infers speakers between `--min-speakers` (default 1) and `--max-speakers` (default 2).
*   **`max_rp_threshold`:** Clustering threshold (default `0.25`). Higher values merge speakers; lower values split them.

---

## 5. Speaker Attribution & Overlap Identification

Once the diarizer generates the raw turns, the pipeline processes the data through two attribution filters: **Overlap Marking** and **Segment/Word Intersection**.

### 5.1 Overlap Marking
`_mark_turn_overlaps` performs an $O(N^2)$ cross-comparison of all turns. If a speaker turn overlaps in time with a turn from a *different* speaker, it is flagged as an overlap.

```python
# tools/diarization/providers/nemo_telephony.py L10-20
def _mark_turn_overlaps(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    marked: list[SpeakerTurn] = []
    for index, turn in enumerate(turns):
        overlap_flag = False
        for other_index, other in enumerate(turns):
            if index == other_index or turn.speaker_label == other.speaker_label:
                continue
            # Check for temporal intersection
            if min(turn.end_sec, other.end_sec) > max(turn.start_sec, other.start_sec):
                overlap_flag = True
                break
```

### 5.2 Segment-Level Speaker Attribution
To map speaker identities back onto ASR segments (from Silero VAD), `annotate_interval_with_speaker` calculates the overlap between a segment time range and the diarizer's speaker turns:

1.  Calculates the total overlap duration (in seconds) for each speaker within the segment bounds.
2.  Assigns the segment to the speaker with the maximum overlap.
3.  Calculates `speaker_confidence` as:
    $$\text{confidence} = \frac{\text{overlap duration with winning speaker}}{\text{total segment duration}}$$
4.  Flags `overlap_flag = True` if more than one speaker is detected in the interval, or if any intersecting turn was flagged as an overlap.

### 5.3 Word-Level Speakerization
When `--timestamp-source` is set to `worker-http`, the script sends the segment audio to the ASR worker's `/v1/transcribe` endpoint with the `X-Timestamp-Type: word` header.

Using the returned word timestamps, `speakerize_words` maps each individual word back to the diarization turns:
```python
# tools/diarization/merge.py L133-164
for index, word in enumerate(words):
    # Shift segment-relative word times to absolute call-level times
    absolute_start = start_sec + float(segment_offset_sec)
    absolute_end = end_sec + float(segment_offset_sec)
    
    # Query interval overlap to determine word speaker
    annotation = annotate_interval_with_speaker(
        turns,
        start_sec=absolute_start,
        end_sec=absolute_end,
    )
```
This enables granular transcripts where speaker identities are assigned at the word level rather than just the segment level.

---

## 6. Pipeline Orchestration

**File:** [`tools/build_diarized_training_data.py`](../tools/build_diarized_training_data.py)

This script acts as the master orchestrator to reuse pre-computed diarization turns. It filters the input dataset, verifies turn coverage, and calls the primary Silero VAD segmenter while injecting the diarization turn file.

```
[Input Inventory]
       │
       ▼
[Filter: require_diarized_status] (Filters out rows not labeled 'diarized_nemo_telephony')
       │
       ▼
[Filter: require_turn_coverage]   (Keeps rows present in diarization_turns.jsonl)
       │
       ▼
[Write Temp CSV]                  (input_inventory_diarized_for_training.csv)
       │
       ▼
[segment_inventory_with_silero]   (Runs segmentation and writes final manifests)
```

---

## 7. Command Line Reference

### 7.1 Running Diarization (`tools/diarize_inventory.py`)
Run NeMo diarization on a normalized inventory:
```bash
python3 tools/diarize_inventory.py \
  --inventory outputs/results_all/data_inventory_normalized.csv \
  --output-dir outputs/results_all/diarization \
  --audio-mode auto \
  --speaker-count-mode estimate \
  --min-speakers 1 \
  --max-speakers 2
```

Key arguments:
*   `--inventory`: Input CSV/Parquet path.
*   `--output-dir`: Output directory for generated turns and status files.
*   `--audio-mode`: Audio channel extraction method (`auto`, `mono`, `borrower`, etc.).
*   `--timestamp-source`: Set to `worker-http` to fetch word timestamps and build speakerized transcripts.
*   `--worker-url`: URL of the ASR worker endpoint (default: `http://localhost:9000/v1/transcribe`).
*   `--clustering-threshold`: Spectral clustering threshold (default: `0.25`).

### 7.2 Segmenting with Pre-computed Turns (`tools/build_diarized_training_data.py`)
To build the final training dataset using a previous diarization run:
```bash
python3 tools/build_diarized_training_data.py \
  --diarization-dir outputs/results_all/diarization \
  --output-dir outputs/results_all/silero_segments_diarized \
  --output-sample-rate 8000
```

---

## 8. Output Schema Reference

### 8.1 Inventory Column Additions
Running `diarize_inventory.py` updates the dataset inventory with the following columns:

| Column | Type | Description |
|---|---|---|
| `diarization_status` | string | `diarized_nemo_telephony`, `diarization_no_speech`, or `diarization_failed` |
| `diarization_provider` | string | Name of the provider (e.g. `nemo_telephony`) |
| `diarization_turns_path` | string | Path to `diarization_turns.jsonl` |
| `diarization_rttm_path` | string | Path to the standard RTTM file for the call |
| `speakerized_transcript_path`| string | Path to `speakerized_transcript.jsonl` |

### 8.2 JSONL File Formats

#### `diarization_turns.jsonl`
Contains raw, unaligned speaker turns for all calls:
```json
{
  "call_id": "call-001",
  "speaker_label": "speaker_0",
  "start_sec": 1.25,
  "end_sec": 4.60,
  "confidence": null,
  "overlap_flag": false,
  "provider": "nemo_telephony",
  "provider_speaker_label": "VAL_0",
  "source_audio_filepath": "/srv/data/call-001.wav"
}
```

#### `speakerized_transcript.jsonl`
Contains ASR segment utterances merged with speaker labels and word-level timestamps:
```json
{
  "segment_id": "call-001-mono-0001",
  "call_id": "call-001",
  "row_id": "1",
  "audio_filepath": "outputs/segments/call-001_seg0001.wav",
  "source_audio_filepath": "data/wav/call-001.wav",
  "lang": "hi",
  "text": "नमस्ते जी",
  "segment_start_sec": 1.20,
  "segment_end_sec": 2.50,
  "speaker_label": "speaker_0",
  "speaker_confidence": 0.95,
  "overlap_flag": false,
  "timestamp_source": "worker_http",
  "words": [
    {
      "word": "नमस्ते",
      "start_sec": 1.25,
      "end_sec": 2.05,
      "word_index": 0,
      "speaker_label": "speaker_0",
      "speaker_confidence": 1.0,
      "overlap_flag": false
    },
    {
      "word": "जी",
      "start_sec": 2.10,
      "end_sec": 2.45,
      "word_index": 1,
      "speaker_label": "speaker_0",
      "speaker_confidence": 1.0,
      "overlap_flag": false
    }
  ]
}
```

---

## 9. Known Limitations

1.  **Python Version Compatibility:** NeMo toolkit packages (`nemo_toolkit[asr]`) are highly pinned to PyTorch and CUDA dependencies, requiring **Python 3.11** locally or execution via `worker/Dockerfile.diarization`.
2.  **No Real-Time Mode:** NeMo's ClusteringDiarizer is designed for offline/batch processing. It consumes the entire file to extract global speaker embeddings and clustering maps, making it unsuitable for live streaming WebSocket pipelines.
3.  **Network Overhead:** On first execution, NeMo's pipeline attempts to pull pretrained weights (`MarblerNet`, `TitaNet`, and `MSDD`) from NGC. For air-gapped or offline GPU clusters, model weights must be pre-cached in target system home directories (`~/.cache/torch` and `~/.cache/huggingface`).
