# CredResolve ASR Knowledge Transfer Guide

This document is the operational and architectural handoff for the CredResolve streaming ASR project. It is meant to help a new engineer understand how the system works, how to run it, where to debug issues, and where to make common changes safely.

## 1. Project Purpose

This repo provides a production-style, streaming-ish speech-to-text stack for Indic ASR over WebSocket.

The main behavior is:

- clients stream audio to the gateway over WebSocket
- the gateway authenticates, rate-limits, validates, buffers, and segments audio
- finalized utterances are sent to a GPU worker over HTTP
- the worker runs IndicConformer inference and returns final transcript text
- the gateway emits final-only transcript events back to the client

This is not true token-by-token streaming ASR. It is utterance-based transcription with streaming ingestion and VAD-driven segmentation.

## 2. High-Level Architecture

Runtime path:

```text
Client / Telephony / Browser
    -> nginx
    -> gateway (FastAPI + WebSocket, CPU)
    -> worker (FastAPI + model inference, GPU)
    -> transcript response
```

Supporting services:

- `prometheus` + `grafana` + `node-exporter` for monitoring
- `frontend` for browser-based testing and demo usage
- optional `triton` for production-style ASR model serving behind the worker API

Primary deployment wiring lives in `docker-compose.yml`.
Triton-specific production overlay wiring lives in `docker-compose.triton.yml`.

## 3. Service Responsibilities

### Gateway

Location: `gateway/app/`

The gateway is the public entrypoint and owns session management.

Responsibilities:

- WebSocket handshake and auth
- query-param based session config validation
- JSON/base64 audio decoding
- audio format normalization
- VAD-based utterance segmentation
- worker dispatch and backpressure
- final transcript emission to clients

Important files:

- `gateway/app/main.py`: websocket API, buffering, VAD flow, worker dispatch
- `gateway/app/worker_client.py`: HTTP client used to call the worker
- `gateway/app/vad.py`: WebRTC-VAD based segmenter
- `gateway/app/config.py`: gateway runtime knobs

### Worker

Location: `worker/app/`

The worker is a private HTTP service focused on model execution.

Responsibilities:

- ASR model download/load on startup
- supported-language resolution
- optional LID when requested language is `auto`
- optional NeMo context biasing for explicit-language requests with deployment-managed phrase files
- audio preparation and resampling
- inference timeout handling
- threadpool-based inference execution
- soft fallback responses instead of hard process failures
- offline NeMo-to-ONNX export through `tools/export_nemo_asr_to_onnx.py`

Important files:

- `worker/app/main.py`: `/v1/transcribe`, worker concurrency, fallback behavior
- `worker/app/model.py`: model load, language resolution, inference execution
- `worker/app/context_biasing.py`: phrase-file parsing, keyword accounting, and optional NeMo context-biasing runtime
- `worker/app/nemo_export.py`: reusable NeMo ASR export helpers and CLI entrypoint logic
- `worker/app/triton.py`: Triton client-backed worker mode
- `worker/app/lid.py`: language ID providers and fallback chain
- `worker/app/config.py`: worker runtime knobs

## 4. End-to-End Request Flow

### WebSocket flow

1. Client connects to `/ws/stt`.
2. Gateway authenticates using `Api-Subscription-Key` or `Sec-WebSocket-Protocol`.
3. Gateway validates required query params like `language-code`, `sample_rate`, and `input_audio_codec`.
4. Client sends JSON `audio` messages containing base64 payloads.
5. Gateway decodes and normalizes the audio into PCM16.
6. Gateway feeds 20 ms frames into `VADSegmenter`.
7. On `speech_end`, `max_utt`, or explicit `{"type":"flush"}`, gateway finalizes the utterance.
8. Gateway sends the utterance bytes to the worker over HTTP.
9. Worker returns `{text, language, language_source}`.
10. Gateway emits a final `type:"data"` message to the client.

### Worker flow

1. Worker receives raw audio bytes at `POST /v1/transcribe`.
2. It reads request metadata from headers such as:
   - `X-Sample-Rate`
   - `X-Decoder`
   - `X-Language`
   - `X-Mode`
   - `X-Session-Id`
   - `X-Utterance-Id`
3. Worker normalizes the audio to model expectations.
4. Worker resolves the language:
   - direct client language if explicit
   - LID-based resolution when `auto`
   - default-language fallback when LID fails or is disabled
5. Worker runs model inference inside a bounded threadpool.
6. Worker returns transcript text plus language metadata.
7. On timeout or unexpected inference failure, worker returns a soft fallback payload instead of crashing the service.

## 5. What "Streaming-ish" Means Here

This project streams audio in, but it does not stream partial token output back out.

Current behavior:

- ingress is streaming over WebSocket
- segmentation is incremental via VAD
- output is final-only per utterance

This is useful for:

- telephony
- browser microphone capture
- near-real-time utterance transcription

It is not yet built for:

- token-by-token live captions
- incremental partial decoding to the client
- TTS roundtrip voice agents

There is a `gateway/app/tts_helper.py`, but it is currently just a helper module and is not wired into the serving path.

## 6. Repository Map

Top-level directories you will touch most often:

- `gateway/`: public WebSocket service
- `worker/`: GPU inference service
- `triton/`: Triton image and model repository for production serving
- `frontend/`: browser demo/test UI
- `docs/`: project docs
- `tools/`: smoke-test and evaluation helpers
- `sample_data/`: local sample wavs for testing
- `nginx/`: reverse proxy config
- `prometheus/`, `grafana/`: observability stack

Recommended first-read order for a new engineer:

1. `README.md`
2. `docs/websocket_auth_migration.md`
3. `gateway/app/main.py`
4. `gateway/app/worker_client.py`
5. `worker/app/main.py`
6. `worker/app/model.py`
7. `docker-compose.yml`

If you need to produce a standalone ONNX file from a NeMo checkpoint, create a Python 3.11 environment first, install `worker/requirements-export.txt`, and then run `python tools/export_nemo_asr_to_onnx.py --help`.

## 7. Environment and Key Runtime Knobs

Defaults live in `.env.example`.

Most important gateway knobs:

- `GATEWAY_MAX_INFLIGHT_WORKER`
- `WORKER_TIMEOUT_MS`
- `WORKER_URL`
- `VAD_END_SILENCE_MS`
- `VAD_KEEP_SILENCE_MS`
- `VAD_MAX_UTT_MS`
- `WS_API_KEYS`

Most important worker knobs:

- `ASR_MODEL_NAME`
- `ASR_DECODER`
- `ASR_INFERENCE_TIMEOUT_MS`
- `ASR_DEFAULT_LANGUAGE`
- `ASR_SUPPORTED_LANGS`
- `ASR_ENABLE_LID`
- `ASR_LID_PRIMARY_PROVIDER`
- `ASR_LID_PRIMARY_SOURCE`
- `ASR_LID_PRIMARY_MODEL_DIR`
- `ASR_LID_FALLBACK_PROVIDER`
- `ASR_LID_FALLBACK_SOURCE`
- `ASR_LID_FALLBACK_MODEL_DIR`
- `ASR_LID_CONFIDENCE_THRESHOLD`
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
- `ASR_LID_CACHE_TTL_SEC`
- `ASR_LID_CACHE_MAX_ENTRIES`
- `WORKER_MAX_JOBS`
- `HUGGINGFACE_HUB_TOKEN`

Operational rule of thumb:

- tune gateway knobs when the problem is session pressure, throttling, VAD behavior, or downstream backpressure
- tune worker knobs when the problem is model latency, GPU concurrency, timeouts, supported languages, or LID behavior

## 8. Startup and Deployment

### Docker Compose

Standard startup:

```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f gateway
docker compose logs -f worker
```

Triton-backed startup:

```bash
cp .env.example .env
# In Triton mode the worker needs an explicit language allowlist.
ASR_SUPPORTED_LANGS=hi,en,bn,ta,te \
docker compose -f docker-compose.yml -f docker-compose.triton.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.triton.yml logs -f triton
docker compose -f docker-compose.yml -f docker-compose.triton.yml logs -f worker
```

Service ports:

- `80`: nginx
- `5173`: frontend
- `8000`: gateway
- `9000`: worker
- `6379`: redis
- `9090`: prometheus
- `3000`: grafana

### Container expectations

Gateway:

- Python 3.11 slim image
- installs `webrtcvad` native dependency toolchain
- runs a single uvicorn worker

Worker:

- CUDA-enabled PyTorch runtime image
- requires GPU availability for the current model path
- runs a single uvicorn worker and handles concurrency internally

### Health checks

- gateway: `GET /healthz`
- worker: `GET /healthz`

Worker health semantics:

- `ok` means model loaded and ready
- `degraded:<detail>` means the service process is up but the model is not ready

## 9. Local Development Workflow

### Run services directly

Gateway:

```bash
cd gateway
uvicorn app.main:app --port 8000
```

Worker:

```bash
cd worker
python -m app.main
```

### Notes

- direct local runs are useful for faster iteration and log visibility
- compose is better when you need the full stack, including Redis, monitoring, frontend, and nginx
- the worker assumes CUDA availability for successful model init in the current setup

## 10. Smoke Test Workflow

Expected sample:

- `sample_data/sample_16k_mono.wav`

Smoke test command:

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
  --vad-signals
```

What success looks like:

- websocket connects
- optional `vad` events are emitted
- final `type:"data"` response contains transcript text
- gateway and worker logs show matching `session_id` and `utterance_id`

## 11. Observability and Debugging

### Logs

Use:

```bash
docker compose logs -f gateway
docker compose logs -f worker
docker compose logs -f frontend
```

Useful log patterns:

- gateway accepted/rejected websocket session
- VAD events: `speech_start`, `speech_end`, `max_utt`
- worker dispatch and completion logs
- timeout and fallback logs from worker

### Metrics

Relevant dashboards and metrics:

- Grafana at `http://localhost:3000`
- Prometheus at `http://localhost:9090`
- `asr_ws_connections`
- `asr_worker_requests_total`
- `asr_worker_inference_seconds`
- `asr_e2e_seconds`
- `asr_worker_errors_total`

### Correlation strategy

Use these fields to trace a request across services:

- `session_id`
- `utterance_id`
- `request_id`
- `language`
- `language_source`

## 12. Test Coverage and Validation

Current visible test coverage focuses on gateway handshake/VAD behavior and worker LID/inference timeout behavior.

Notable test files:

- `gateway/tests/test_ws_stt_partial_out_guard.py`
- `worker/tests/test_lid_chain.py`
- `worker/tests/test_model_lid_cache.py`

Example test command:

```bash
pytest gateway/tests worker/tests
```

If `pytest` is missing in your environment, install it first in the environment you are using for local runs.

## 13. Common Failure Modes

### 1. Gateway accepts connections but transcripts never arrive

Check:

- worker health status
- worker logs for model init failure
- `WORKER_URL`
- gateway timeout settings versus worker inference time
- whether the session is sending `flush` or enough silence for VAD to end the utterance

### 2. Worker health returns degraded

Likely causes:

- model download failure
- invalid Hugging Face token
- GPU/CUDA provider missing
- ONNX Runtime CUDA session failure

Check:

- `HUGGINGFACE_HUB_TOKEN`
- GPU visibility in the container/host
- worker startup logs

### 3. Sessions are rejected at connect time

Check:

- `WS_API_KEYS`
- `language-code` query param
- allowed `sample_rate`
- allowed `input_audio_codec`

### 4. Worker returns `worker-fallback`

This means the service stayed alive but inference did not complete successfully.

Check worker logs for:

- timeout
- model not ready
- inference error
- unexpected exception

## 14. Language Behavior Notes

There are two important language concepts in the system:

- requested language: what the client asked for
- resolved language: what the worker actually used

When the request uses `language-code=auto`:

- worker may run LID
- worker may use cached LID result
- worker may fall back to `ASR_DEFAULT_LANGUAGE`

Be careful when making product changes that assume the public request language is always the model language.

## 15. Change Guide: Where to Modify What

### Add or change websocket contract

Start in:

- `gateway/app/main.py`
- `docs/websocket_auth_migration.md`
- `tools/ws_client_send_wav.py`
- frontend websocket client code in `frontend/src/lib/`

### Change VAD behavior

Start in:

- `gateway/app/vad.py`
- gateway env defaults in `.env.example`
- gateway config parsing in `gateway/app/config.py`

### Change worker request/response contract

Start in:

- `gateway/app/worker_client.py`
- `worker/app/main.py`

### Change ASR model or language support

Start in:

- `worker/app/config.py`
- `worker/app/model.py`
- worker startup/load path

### Change LID behavior

Start in:

- `worker/app/lid.py`
- `worker/app/model.py`
- `.env.example`
- worker tests under `worker/tests/`

### Change context biasing behavior

Start in:

- `worker/app/context_biasing.py`
- `worker/app/main.py`
- `.env.example`
- `docker-compose.context_biasing.yml`
- `docs/indicvoices_eval.md`

### Add TTS or voice-agent behavior

Current status:

- this repo is STT-first
- `gateway/app/tts_helper.py` exists but is not integrated into the request path

If TTS is added later, decide clearly whether it belongs:

- inside gateway as a post-ASR orchestration step
- as a separate worker/service
- as a new client-facing endpoint

## 16. Known Architectural Constraints

- output is final-only, not partial streaming
- gateway currently hardcodes internal decode dispatch to RNNT for worker calls
- worker soft-falls back instead of surfacing hard errors to the client in many cases
- worker currently expects CUDA availability for successful model readiness
- frontend is a test/demo surface, not the core business logic

These constraints are important when setting expectations with product or operations teams.

## 17. IndicVoices Evaluation Baseline and Iteration Comparison

Current saved full-run Hindi baseline artifact:

- summary: `artifacts/indicvoices_hindi_valid_full_summary.json`
- artifact: `artifacts/indicvoices_hindi_valid_full_errors.jsonl`
- split: `ai4bharat/IndicVoices`, config `hindi`, split `valid`
- samples: `4740`
- processed: `4740`
- failures: `0`
- reference words: `73343`
- substitutions: `8656`
- deletions: `1801`
- insertions: `1423`
- WER: `0.1620`
- WER percent: `16.20`

Important interpretation note:

- The raw baseline above includes `dataset-noise` rows where the reference contains `<unintelligible>`.
- It also includes `infra-failure` rows where the returned hypothesis contains `worker-fallback`.
- The evaluator now reports these separately and also emits a `clean_wer` / `clean_wer_percent` that excludes both buckets from ASR-only model-quality reporting.

This baseline is now available as both a compact summary JSON and the original per-sample JSONL, so it remains usable even if the original console log is unavailable.

To rerun the full benchmark against the current stack and save a structured summary:

```bash
python tools/eval_indicvoices_wer.py \
  --ws ws://localhost/ws/stt \
  --api-key dev \
  --dataset-config hindi \
  --split valid \
  --language hi \
  --limit 4740 \
  --out-tsv artifacts/indicvoices_hindi_valid_full_iter2.tsv \
  --out-summary-json artifacts/indicvoices_hindi_valid_full_iter2_summary.json \
  --out-errors-jsonl artifacts/indicvoices_hindi_valid_full_iter2_errors.jsonl \
  --top-errors 20
```

To mine trainable examples from the saved `valid` error profile without contaminating the benchmark split:

```bash
python tools/mine_indicvoices_train_examples.py \
  --errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --dataset-config hindi \
  --split train \
  --scan-limit 50000 \
  --per-bucket-limit 300 \
  --normal-limit 300 \
  --export-dir artifacts/indicvoices_hindi_train_mined
```

This exports:

- a `summary.json` with bucket counts and anchor phrases
- a `manifest.jsonl` for fine-tuning prep
- local `audio/` wavs for the mined IndicVoices train samples

Optional:

- pass `--phrases-file context_biasing/phrases/hi.txt` to mine entity/domain terms
- pass `--call-manifest-jsonl <path>` to merge labeled call-recording examples into the same manifest

To compare the first and second iterations and mark which samples were wrong in both:

```bash
python tools/compare_indicvoices_iterations.py \
  --first-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --second-errors-jsonl artifacts/indicvoices_hindi_valid_full_iter2_errors.jsonl \
  --out-summary-json artifacts/indicvoices_hindi_valid_comparison_summary.json \
  --out-comparison-jsonl artifacts/indicvoices_hindi_valid_comparison.jsonl \
  --out-repeated-error-indices artifacts/indicvoices_hindi_valid_repeated_error_indices.txt \
  --top-persistent 20
```

The comparison output classifies each sample as one of:

- `error_both`: wrong in first and second iteration
- `fixed_in_second`: wrong in first, clean in second
- `regressed_in_second`: clean in first, wrong in second
- `clean_both`: clean in both
- `missing_in_first` / `missing_in_second`: one iteration artifact is incomplete

To export the repeated-error subset as wavs plus a JSONL manifest:

```bash
python tools/compare_indicvoices_iterations.py \
  --first-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --second-errors-jsonl artifacts/indicvoices_hindi_valid_full_iter2_errors.jsonl \
  --export-repeated-errors-dir artifacts/indicvoices_hindi_valid_repeated_errors \
  --dataset-config hindi \
  --split valid
```

This creates:

- `artifacts/indicvoices_hindi_valid_repeated_errors/manifest.jsonl`
- `artifacts/indicvoices_hindi_valid_repeated_errors/audio/*.wav`

Important training caution:

- the repeated-error export comes from the `valid` split, so using it directly for final training will leak evaluation data
- use it for error analysis, phrase mining, and hard-example discovery
- for a clean training loop, map the repeated-error patterns back to similar samples from `train` and keep `valid` untouched for reporting

## 18. Suggested Day-1 KT Walkthrough

For a handoff session, the most useful order is:

1. explain the product flow in the README
2. open `docker-compose.yml` and show service boundaries
3. walk through `gateway/app/main.py` from websocket connect to `finalize_audio`
4. walk through `gateway/app/worker_client.py`
5. walk through `worker/app/main.py`
6. walk through `worker/app/model.py`
7. run the smoke test with `tools/ws_client_send_wav.py`
8. inspect logs in gateway and worker side-by-side
9. open Grafana and Prometheus to show where operational visibility lives

## 19. Suggested Next Documentation Improvements

Good follow-up docs to add later:

- sequence diagram for websocket -> worker flow
- incident runbook for worker degraded state
- language support matrix generated from the loaded model snapshot
- deployment notes for EC2/NVIDIA driver setup
- performance tuning guide for concurrency and latency tradeoffs

## 20. Summary

If you remember only three things:

- gateway is the control plane for sessions, validation, throttling, and segmentation
- worker is the execution plane for language resolution and GPU ASR inference
- most production debugging starts by correlating one `session_id` and `utterance_id` across gateway logs, worker logs, and metrics
