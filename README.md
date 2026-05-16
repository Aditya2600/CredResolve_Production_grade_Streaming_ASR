# CredResolve – Production-grade Streaming ASR (IndicConformer) over WebSocket

This repo is a production-ready reference implementation for a **streaming-ish** STT service:
- WebSocket (WSS) ingest of raw PCM binary frames, with legacy JSON/base64 audio still available
- VAD / endpointing (WebRTC VAD)
- Worker concurrency caps and request timeouts
- Split architecture:
  - `gateway` (CPU): WebSocket sessions + auth + VAD + final-only `data` events
  - `worker` (GPU): IndicConformer ONNX/TorchScript transcription service

---

## Architecture

Client/Telephony → Nginx → gateway (WS) → worker (HTTP, GPU) → transcripts

---

## Triton Production Serving

For production-style serving, this repo now supports running the HTTP worker as a thin adapter in front of NVIDIA Triton Inference Server.

Bring up the Triton-backed stack with the compose overlay:

```bash
cp .env.example .env
# Required in Triton mode so the worker can validate explicit language requests.
export ASR_SUPPORTED_LANGS=hi,en,bn,ta,te
docker compose -f docker-compose.yml -f docker-compose.triton.yml up --build -d
```

What changes in this mode:
- Triton serves the ASR model from `triton/model_repository/indic_asr/config.pbtxt`.
- The worker keeps the existing `/v1/transcribe` contract, LID flow, and fallback behavior.
- The worker forwards normalized audio plus `language` and `decoder` to Triton over HTTP.
- Triton listens on host ports `8100` (HTTP), `8101` (gRPC), and `8102` (metrics).

Important constraints:
- `ASR_SUPPORTED_LANGS` must be set in Triton mode because the remote model does not expose vocab metadata back to the worker.
- The Triton image is pinned via `TRITON_SERVER_IMAGE` in `.env.example`; adjust it if your fleet standard differs.
- The RNNT path still uses a Triton Python backend for the data-dependent decode loop, but its heavy encoder stage is delegated to the shared `indic_asr_encoder` TensorRT model. That keeps RNNT semantics while still exercising the TensorRT encoder in production-style serving.

---

## Quickstart (Docker Compose on EC2)

```bash
cp .env.example .env
docker system df   # preflight: check free Docker disk space
docker compose up --build -d
docker compose logs -f gateway
```

Frontend UI:
- On the EC2 host, nginx serves the app at `http://localhost`
- The frontend dev server is also exposed directly at `http://localhost:5173`
- Gateway health is available directly at `http://localhost:8000/healthz` and through nginx at `http://localhost/healthz`
- Gateway WebSocket traffic goes through nginx at `ws://localhost/ws/stt`; the direct gateway port is `8000`
- No service listens on host port `8080` unless you create the SSH tunnel below

Remote browser microphone access:
- Browsers will block mic access on `http://<server-ip>` because it is not a secure origin
- From your local machine, create an SSH tunnel to the server's nginx port: `ssh -L 8080:localhost:80 <user>@<server>`
- Then open `http://localhost:8080` in your local browser
- Do not expect `curl http://localhost:8080/healthz` to work from the EC2 shell; use `http://localhost/healthz` or `http://localhost:8000/healthz` there

Logs:
- All services: `docker compose logs -f`
- Gateway only: `docker compose logs -f gateway`
- Worker only: `docker compose logs -f worker`
- Frontend only: `docker compose logs -f frontend`
- Last 100 lines for a service: `docker compose logs --tail=100 -f gateway`
- Browser logs: open DevTools Console for frontend WebSocket/audio tracing (`VITE_DEBUG_LOGS=true` in Docker build by default)
- Increase backend verbosity with `LOG_LEVEL=DEBUG` in `.env` before `docker compose up --build -d`
- Print client-visible final transcripts in gateway logs with `LOG_TRANSCRIPTS=true` (off by default in `.env.example` because transcript text may contain PII)

Smoke test (put a 16kHz mono PCM16 wav at `sample_data/sample_16k_mono.wav`):
```bash
python3 tools/ws_client_send_wav.py \
  --ws ws://localhost/ws/stt \
  --wav sample_data/sample_16k_mono.wav \
  --api-key dev \
  --language-code hi \
  --model credresolve:v1 \
  --mode transcribe \
  --sample-rate 16000 \
  --input-audio-codec pcm_s16le \
  --vad-signals \
  --flush-signal
```

For the legacy JSON/base64 framing, add `--json-base64`. The default PCM path negotiates
`binary_audio=1` and sends audio chunks as WebSocket binary frames.

Supported languages include `hi`, `en`, `bn`, `ta`, etc. (Check model documentation for full list).

The public STT contract is now Sarvam-like:
- auth comes from `Api-Subscription-Key: <token>` or browser `Sec-WebSocket-Protocol: token,<token>`
- session config comes from query params such as `language-code`, `model`, `mode`, `sample_rate`, and `input_audio_codec`
- clients send raw PCM WebSocket binary frames when `binary_audio=1`, then `{"type":"flush"}`
- legacy JSON/base64 audio envelopes are still accepted when `binary_audio` is unset or false
- the server emits final-only `type:"data"` transcript messages and optional `type:"vad"` signals

See [docs/websocket_auth_migration.md](docs/websocket_auth_migration.md) for the full before/after contract and payload examples.

---

## Offline Segmentation With Optional Diarization

The repo includes an offline segmentation tool for ASR training data at `tools/segment_inventory_with_silero.py`.

Default behavior stays the same:
- audio -> Silero VAD -> heuristic transcript grouping -> `manifest.jsonl`

Optional diarization-aware behavior can now be enabled for mono telephony, where one channel may still contain both agent and user:
- audio -> Silero VAD -> NeMo telephony diarization -> speaker-aware boundary refinement -> final segmentation -> manifest

Important limitation:
- if transcript utterances are only role-ordered and do not have timestamps, the tool does **not** pretend to have word-level or utterance-level speaker alignment
- diarization is used only to refine acoustic segment boundaries and to attach segment-level metadata such as dominant speaker, overlap, and mixed-speaker flags
- transcript grouping still uses the existing order-based heuristic and is marked in the manifest as `transcript_alignment_method: "role_order_heuristic"`

Example VAD-only run:

```bash
python3 tools/segment_inventory_with_silero.py \
  --inventory outputs/results_all/data_inventory_normalized.csv \
  --output-dir outputs/results_all/silero_segments
```

Example diarization-aware run:

```bash
python3 tools/segment_inventory_with_silero.py \
  --inventory outputs/results_all/data_inventory_normalized.csv \
  --output-dir outputs/results_all/silero_segments_diarized \
  --enable-diarization \
  --diarization-audio-mode auto \
  --diarization-max-speakers 2
```

Manifest additions are additive and backward-compatible. New fields include:
- `speaker_label`
- `speaker_confidence`
- `overlap_flag`
- `diarization_used`
- `diarization_source`
- `diarization_provider`
- `diarization_boundary_refined`
- `diarization_speaker_count`
- `diarization_mixed_speaker_flag`
- `speaker_alignment_method`
- `transcript_alignment_method`

If you already have diarization turns from a separate pass, you can reuse them without running NeMo inline:

```bash
python3 tools/segment_inventory_with_silero.py \
  --inventory outputs/results_all/data_inventory_normalized.csv \
  --output-dir outputs/results_all/silero_segments_labeled \
  --diarization-turns outputs/results_all/diarization/diarization_turns.jsonl
```

When inline diarization is enabled, the tool also writes:
- `diarization_turns.jsonl`
- `pred_rttms/*.rttm`
- inventory-level diarization status columns in `data_inventory_segmented.csv`

---

## Language Auto Mode (LID)

- LID runs only when requested language is `auto` (or empty).
- LID is CPU-only and feature-flagged (`ASR_ENABLE_LID=false` by default).
- The worker can run a primary/fallback LID chain. By default for the new config surface:
  - primary: `onecxi/vakgyata-small`
  - fallback: `speechbrain/lang-id-voxlingua107-ecapa`
- If LID is disabled/unavailable/unmappable, worker falls back to `ASR_DEFAULT_LANGUAGE`.
- Backward compatibility: if the new chain env vars are unset but the legacy single-provider
  `ASR_LID_MODEL_SOURCE` / `ASR_LID_MODEL_DIR` settings are explicitly set, they continue to work.
- Worker `/v1/transcribe` responses include:
  - `text`
  - `language`
  - `language_source` (`client`, `auto_default`, `lid_detected`, `lid_cached`, `lid_fallback_default`)
- The public websocket contract does not expose language metadata on `type:"data"` messages.
- The worker still resolves `language` and `language_source` internally for routing, logging, and evaluation.
- **Language Metadata**:
  - `language`: Resolved language code (e.g. `hi`, `en`, `te`).
  - `language_source`: How the language was determined:
    - `client`: Explicitly requested by user.
    - `lid_detected`: Detect by Language ID model.
    - `lid_cached`: Used cached result from previous utterance in session.
    - `auto_default` / `lid_fallback_*`: Fallback to default language.

Quick offline LID benchmark:

1. Create a manifest with one labeled clip per row, for example:

```json
{"audio_path":"recordings/clip_001.wav","language":"hi"}
{"audio_path":"recordings/clip_002.wav","language":"te"}
```

2. Run the worker with `ASR_ENABLE_LID=true`.
3. Evaluate against the worker directly:

```bash
python tools/eval_lid.py \
  --manifest artifacts/lid_eval_manifest.jsonl \
  --worker-url http://localhost:8001/v1/transcribe \
  --target-sample-rate 16000 \
  --out-summary-json artifacts/lid_eval_summary.json \
  --out-results-jsonl artifacts/lid_eval_results.jsonl
```

The tool sends every row with `x-language: auto`, generates unique session IDs by default so cache reuse does not skew the benchmark, and reports accuracy, macro precision/recall/F1, confusion counts, fallback rate, and latency percentiles.

---

## Experimental Context Biasing

This repo now includes a feature-flagged, worker-side NeMo context-biasing path for domain terms.

Important behavior:

- the public websocket contract does not change
- the existing Triton/ONNX baseline stays the default path
- context biasing applies only when the request language is explicit and a matching phrase file exists
- `language-code=auto` stays on the baseline path in v1

Modes:

- `ASR_CONTEXT_BIASING_MODE=disabled`: baseline only
- `ASR_CONTEXT_BIASING_MODE=shadow`: return baseline transcript, but run NeMo context biasing in the worker for evaluation/logging
- `ASR_CONTEXT_BIASING_MODE=active`: return the context-biased transcript when the second decode succeeds; otherwise fall back to baseline

Phrase file format:

- one file per language, for example `context_biasing/phrases/hi.txt`
- underscore-delimited variants per line
- first token is the canonical term
- remaining tokens are accepted variants or spellings

Example `hi.txt`:

```text
loan id_loan id_लोन आईडी
cred resolve_credresolve_क्रेड रिजॉल्व
```

To run the NeMo-enabled worker image without changing the default stack:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.triton.yml \
  -f docker-compose.context_biasing.yml \
  up --build -d
```

Typical envs:

- `ASR_CONTEXT_BIASING_MODE=shadow`
- `ASR_CONTEXT_BIASING_METHOD=ctc_ws`
- `ASR_CONTEXT_BIASING_NEMO_SOURCE=<your .nemo path or pretrained model name>`
- `ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS=<EncDecCTCModelBPE|EncDecRNNTBPEModel|...>`
- `ASR_CONTEXT_BIASING_PHRASES_DIR=/srv/context_biasing/phrases`

Evaluation workflow:

1. Run a baseline IndicVoices evaluation and save `--out-summary-json` plus `--out-errors-jsonl`.
2. Restart with the candidate configuration and rerun the same evaluation.
3. Compare the two iterations with `tools/compare_indicvoices_iterations.py`.
4. Add `--phrases-file context_biasing/phrases/hi.txt` to score keyword precision/recall/F1 for the same domain lexicon used at decode time.
5. Use the comparison summary JSON to check WER deltas plus client/server latency deltas.

---

## Key Knobs

- `GATEWAY_MAX_INFLIGHT_WORKER`
- `WORKER_MAX_JOBS`
- `WORKER_TIMEOUT_MS`
- `GATEWAY_WS_PING_INTERVAL`
- `GATEWAY_WS_PING_TIMEOUT`
- `PARTIAL_DECODE_INTERVAL_MS`
- `ASR_ENABLE_LID`
- `ASR_LID_PRIMARY_PROVIDER`
- `ASR_LID_PRIMARY_SOURCE`
- `ASR_LID_PRIMARY_MODEL_DIR`
- `ASR_LID_FALLBACK_PROVIDER`
- `ASR_LID_FALLBACK_SOURCE`
- `ASR_LID_FALLBACK_MODEL_DIR`
- `ASR_LID_CONFIDENCE_THRESHOLD`
- `ASR_LID_MODEL_SOURCE`
- `ASR_LID_MODEL_DIR`
- `ASR_LID_CACHE_TTL_SEC`
- `ASR_LID_CACHE_MAX_ENTRIES`
- `ASR_CONTEXT_BIASING_MODE`
- `ASR_CONTEXT_BIASING_METHOD`
- `ASR_CONTEXT_BIASING_NEMO_SOURCE`
- `ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS`
- `ASR_CONTEXT_BIASING_PHRASES_DIR`
- `ASR_CONTEXT_BIASING_TIMEOUT_MS`
- `ASR_CONTEXT_BIASING_DEVICE`
- `ASR_CONTEXT_BIASING_SHADOW_SAMPLE_RATE`
- `ASR_CONTEXT_BIASING_BEAM_THRESHOLD`
- `ASR_CONTEXT_BIASING_CONTEXT_SCORE`
- `ASR_CONTEXT_BIASING_CTC_ALI_TOKEN_WEIGHT`
- `ASR_SUPPORTED_LANGS`
- `ASR_BACKEND`
- `ASR_TRITON_PROTOCOL`
- `TRITON_URL`
- `TRITON_MODEL_NAME`
- `TRITON_MODEL_VERSION`
- `ASR_TRITON_READY_RETRY_ATTEMPTS`

---

## Monitoring & Observability

This project includes a full monitoring stack (Prometheus + Grafana + Node Exporter).

### Quick Start
1.  Start the stack:
    ```bash
    docker compose up -d
    ```
2.  **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000)
    *   **Login**: `admin` / `admin`
    *   **Dashboard**: Go to "Dashboards" > "CredResolve ASR" folder > "CredResolve ASR Dashboard".
    *   *Note: Data sources and dashboards are auto-provisioned.*

3.  **Prometheus**: [http://localhost:9090](http://localhost:9090)
    *   Check targets: [http://localhost:9090/targets](http://localhost:9090/targets)

### Key Metrics
*   **Active Connections**: `asr_ws_connections`
*   **Request Rate**: `rate(asr_worker_requests_total[1m])`
*   **Latency**: `asr_worker_inference_seconds` (Model), `asr_e2e_seconds` (End-to-End)
*   **Errors**: `asr_worker_errors_total`
*   **System**: CPU, RAM, Disk (via Node Exporter)

### Troubleshooting
*   **No Data in Grafana?** Check if Prometheus targets are UP.
*   **Zero connections?** Connect a client! (e.g. Frontend or `tools/ws_client_send_wav.py`).
*   **Disk Full?** Check Node Exporter dashboard.

## Development

### Concurrency Load Test
Simulate N concurrent clients:
```bash
# Setup
pip install websockets

# Run (Small)
python tools/load_ws.py --n 20

# Run (Medium - 50 clients, 30s audio, ramp-up)
python tools/load_ws.py --n 50 --ramp-ms 100 --max-audio-sec 30
```

monitor with:
```bash
./tools/prom_query.sh 'sum(asr_ws_connections)'
```

### Example Benchmark Observation

Observed on April 2, 2026 with:
- `ASR_BACKEND=local`
- `GATEWAY_MAX_INFLIGHT_WORKER=2`
- `WORKER_MAX_JOBS=2`
- `recordings/new_test_recording_16_28_mono_8k.wav` (12s, 8kHz, mono PCM16)

Important: `tools/load_ws.py` reports session duration, which includes:
- real-time audio streaming time
- the configured `--timeout-sec` silence wait after `flush`

| Test Scenario | Clients | `--timeout-sec` | Sessions Established | Successful Completions | Failures | Avg Session Duration | P95 Session Duration | Observation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Single-user baseline | 1 | 2 | 1 | 1 | 0 | 14.47s | - | End-to-end path works reliably for one client |
| Burst concurrency, short post-flush wait | 50 | 2 | 50 | 8 | 42 | 15.50s | 16.16s | All 50 clients connected, but most timed out waiting for a final transcript |
| Burst concurrency, extended post-flush wait | 50 | 15 | 50 | 50 | 0 | 38.45s | 38.82s | All 50 clients completed when enough queue time was allowed |

Takeaways:
- The system supported `50` simultaneous WebSocket client sessions in this test.
- The system did not process `50` transcriptions in parallel under this configuration.
- With gateway and worker concurrency both capped at `2`, most requests were queued before transcription.

### IndicVoices WER Benchmark Tiers

For `ai4bharat/IndicVoices`, config `hindi`:
- `train`: `383004` rows
- `valid`: `4740` rows
- average words per row: `15.47`
- average audio duration per row: `5.55s`
- total audio in `valid`: `7.31` hours

Recommended benchmark sizes:

| Tier | Rows | Approx. Audio | Best Use |
|---|---:|---:|---|
| Smoke | 5 | 28s | Validate setup and websocket path |
| Development | 500 | 46.3 min | Fast iteration and error analysis |
| Confirmation | 1000 | 92.6 min | Compare close changes with more confidence |
| Full valid | 4740 | 7.31 h | Final reporting |

Why `500` first:
- It is already about `10.5%` of the Hindi `valid` split.
- It is usually enough to expose the main failure modes and provide a stable directional WER.
- `1000` nearly doubles runtime and serving cost for much less added debugging value when iterating through websocket + gateway + Triton.

See [docs/indicvoices_eval.md](/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/docs/indicvoices_eval.md) for the full runbook, monitoring queries, and output interpretation.

See [akshay-docs/indic_conformer_finetuning_strategy.md](/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/akshay-docs/indic_conformer_finetuning_strategy.md) for research paper context, failure analysis, and the optimized fine-tuning strategy for 600M models.


```bash
# Gateway
cd gateway
uvicorn app.main:app --port 8000

# Worker
cd worker
python -m app.main
```

#sample

## Export A NeMo ASR Checkpoint To ONNX

The serving worker already consumes an ONNX bundle, but this repo can now export a NeMo ASR checkpoint into a standard `.onnx` artifact for offline conversion and deployment workflows.

Use Python 3.11 for local installs. The worker and export dependencies are pinned around the
`pytorch/pytorch:2.4.1` runtime, so a newer interpreter such as Python 3.13 will fail to resolve
packages like `torch==2.4.1`.

Create a local export environment and install the export-only dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r worker/requirements-export.txt
```

Export from a local `.nemo` checkpoint:

```bash
python tools/export_nemo_asr_to_onnx.py \
  --source /path/to/model.nemo \
  --output artifacts/nemo_asr/model.onnx
```

Export from a named NeMo pretrained model:

```bash
python tools/export_nemo_asr_to_onnx.py \
  --source nvidia/parakeet-ctc-1.1b \
  --model-class EncDecCTCModelBPE \
  --output artifacts/nemo_asr/model.onnx
```

Notes:
- The exporter writes a sibling metadata file at `<output>.metadata.json`.
- Validation runs by default through `onnx.checker`. Use `--no-validate` to skip it.
- `--use-dynamo` is optional. The default path prefers the legacy exporter because it is generally steadier for ASR ONNX export.
- If you install NeMo on a fresh machine, NVIDIA recommends system audio deps such as `ffmpeg` and `libsndfile1`.

## ASR Inventory Builder

`build_asr_inventory.py` turns a raw call-export CSV into a normalized ASR inventory table while preserving the original source columns.

### Expected input

- Input is a call registry CSV, not a transcript manifest.
- Pass the file with `--input`, for example:

```bash
python build_asr_inventory.py --input query_result_2026-04-07T05_59_50.851541136Z.csv
```

- The script reads the CSV with string-safe parsing so IDs, codes, and phone-like fields are not silently retyped.

### How schema mapping works

- The script prepends a normalized schema with:
  `row_id`, `call_id`, `audio_url`, `local_audio_path`, `wav_audio_path`, `channel_0_path`, `channel_1_path`, `borrower_channel`, `channel_assignment_method`, `channel_assignment_confidence`, `channels`, `sample_rate`, `duration_sec`, `audio_structure`, `recommended_next_step`, `lender`, `portfolio`, `borrower_name`, `event_date`, `amount_1`, `amount_2`, `amount_3`, `transcript_source`, `raw_transcript`, `normalized_transcript`, `download_status`, `audio_convert_status`, `segmentation_status`, `alignment_status`, `quality_tier`, `split`, `slice_tags`.
- It auto-maps common call-center columns by heuristic matching, for example:
  - `call_id` from columns such as `call_sid`, `call_id`, `session_id`
  - `audio_url` from columns such as `cr_recording_url`, `recording_url`, `audio_url`
  - `lender` from `lender` or client-style columns
  - `portfolio` from campaign, dialer, group, or portfolio-style columns
  - `borrower_name` from borrower, customer, or name-style columns
  - `event_date` from call time, start time, or date/timestamp columns
  - `amount_1` / `amount_2` / `amount_3` from principal, EMI, due, balance, or other amount-style columns
- If a normalized source-derived field cannot be inferred, it is left null and logged clearly in the run summary.
- If a source column name collides with the normalized schema, the source column is preserved with a `source__...` export name so the Parquet schema stays valid.

### What the script does

- Preserves all source CSV columns in the exported inventory.
- Downloads audio from `audio_url` to `data/raw/{call_id}.mp3`.
- Inspects each downloaded source recording with `ffprobe` and stores:
  - `channels`
  - `sample_rate`
  - `duration_sec`
  - `audio_structure`: `mono`, `stereo`, `multi_channel`, or `unknown`
  - `recommended_next_step`: `split_channels_first`, `diarization_or_role_filter_first`, or `manual_review`
- When a call is stereo, splits it into:
  - `data/channels/{call_id}_ch0.wav`
  - `data/channels/{call_id}_ch1.wav`
- Writes those paths into:
  - `channel_0_path`
  - `channel_1_path`
- Writes borrower-audit metadata into:
  - `borrower_channel`
  - `channel_assignment_method`
  - `channel_assignment_confidence`
- Converts downloaded audio to mono 8 kHz WAV at `data/wav/{call_id}.wav`.
- Initializes transcript fields as null.
- Sets:
  - `download_status`: `pending` before processing, then `done` or `failed`
  - `audio_convert_status`: `pending` before processing, then `done` or `failed`
  - `segmentation_status`: `pending`
  - `alignment_status`: `pending`
  - `quality_tier`: `unreviewed`
  - `split`: `unset`
  - `slice_tags`: `[]`

### Setup and run

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure `ffmpeg` and `ffprobe` are installed and available on `PATH`, then run:

```bash
python build_asr_inventory.py --input query_result_2026-04-07T05_59_50.851541136Z.csv
```

Useful options:

```bash
python build_asr_inventory.py \
  --input query_result_2026-04-07T05_59_50.851541136Z.csv \
  --audit-sample-size 25 \
  --channel-map outputs/borrower_channel_audit_filled.csv \
  --default-borrower-channel ch1 \
  --max-rows 100 \
  --timeout 90 \
  --retries 5 \
  --retry-backoff-seconds 3 \
  --overwrite \
  --log-level DEBUG
```

### Borrower channel audit

- Every run now writes `outputs/borrower_channel_audit.csv` by default.
- The audit CSV contains the split channel paths plus borrower assignment columns that are easy to review manually.
- To audit a sample instead of the full eligible set, pass `--audit-sample-size <N>`.
- To apply reviewed assignments back into the inventory on the next run, pass a CSV with `call_id,borrower_channel` through `--channel-map`.
- Once you confirm a stable pattern, you can apply a global fallback with `--default-borrower-channel ch0` or `--default-borrower-channel ch1`.

### Output files

- `outputs/data_inventory.csv`
- `outputs/data_inventory.parquet`
- `outputs/borrower_channel_audit.csv`
- `data/raw/{call_id}.mp3`
- `data/wav/{call_id}.wav`
- `data/channels/{call_id}_ch0.wav`
- `data/channels/{call_id}_ch1.wav`

Generated inventory artifacts under `outputs/` and raw input exports matching `query_result_*.csv` are intended to stay local working files and are now ignored by git.

The logs also emit a summary with total rows, download success/failure counts, WAV conversion success/failure counts, auto-mapped columns, and normalized fields that could not be inferred.

### Assumptions

- Audio URLs are reachable over HTTP or HTTPS.
- Output file names are derived from `call_id`; if a `call_id` contains filesystem-hostile characters, the path component is sanitized while the `call_id` column keeps the original value.
- `event_date` is normalized to ISO-like text when the source value parses cleanly as a date/time; otherwise the original text is preserved.
- Segmentation, alignment, and transcript generation are intentionally not run in this step.

## Silero VAD Segmentation

`tools/segment_inventory_with_silero.py` turns the normalized inventory into utterance-level training audio plus a NeMo-friendly JSONL manifest.

### What it does

- Reads `outputs/data_inventory.csv` or `.parquet`.
- Selects the right source audio per row:
  - mono WAV by default for `ready_mono_segmentation`
  - borrower channel automatically for `ready_split_channels` when `borrower_channel` is known
- Parses transcript payloads from columns such as `normalized_transcript`, `native_language_transcript`, or `raw_transcript`.
- Runs Silero VAD to find speech regions, aligns those regions to transcript utterances in order, and exports one WAV per aligned segment.
- Writes a NeMo-style manifest plus an updated inventory copy with `segmentation_status=segmented_silero_vad` for successful rows.

### Install

The segmentation script needs `torch` and `silero-vad` in the active environment in addition to the usual audio/data packages.

```bash
pip install torch silero-vad soundfile pandas pyarrow numpy
```

### Run

```bash
python tools/segment_inventory_with_silero.py \
  --inventory outputs/results_all/data_inventory.csv \
  --output-dir outputs/results_all/silero_segments \
  --audio-mode auto \
  --text-column raw_transcript \
  --language-column language \
  --output-sample-rate 8000
```

Useful options:

```bash
python tools/segment_inventory_with_silero.py \
  --inventory outputs/results_all/data_inventory.csv \
  --output-dir outputs/results_all/silero_segments \
  --audio-mode borrower \
  --text-column raw_transcript \
  --language-column language \
  --threshold 0.5 \
  --min-speech-duration-ms 250 \
  --min-silence-duration-ms 150 \
  --speech-pad-ms 60 \
  --max-speech-duration-s 20 \
  --overwrite
```

### Output files

- `outputs/.../silero_segments/audio/{call_id}_..._seg0001.wav`
- `outputs/.../silero_segments/manifest.jsonl`
- `outputs/.../silero_segments/segments.jsonl`
- `outputs/.../silero_segments/data_inventory_segmented.csv`
- `outputs/.../silero_segments/data_inventory_segmented.parquet`
- `outputs/.../silero_segments/summary.json`

### Next steps

- Bucket, filter, or dedupe `manifest.jsonl` before NeMo fine-tuning.
- Add stronger forced alignment when transcript order and pause structure drift too far apart.
- Attach transcript source metadata and normalized text after ASR inference.
- Add alignment once transcript supervision is available.
