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

- `redis` for connection admission and rate limiting
- `prometheus` + `grafana` + `node-exporter` for monitoring
- `frontend` for browser-based testing and demo usage

Primary deployment wiring lives in `docker-compose.yml`.

## 3. Service Responsibilities

### Gateway

Location: `gateway/app/`

The gateway is the public entrypoint and owns session management.

Responsibilities:

- WebSocket handshake and auth
- query-param based session config validation
- JSON/base64 audio decoding
- audio format normalization
- audio byte-rate limiting
- connection admission limiting
- VAD-based utterance segmentation
- worker dispatch and backpressure
- circuit breaker behavior when downstream is failing
- final transcript emission to clients

Important files:

- `gateway/app/main.py`: websocket API, buffering, VAD flow, worker dispatch
- `gateway/app/worker_client.py`: HTTP client used to call the worker
- `gateway/app/vad.py`: WebRTC-VAD based segmenter
- `gateway/app/config.py`: gateway runtime knobs
- `gateway/app/redis_limiter.py`: Redis-backed session limiting
- `gateway/app/fallback_limiter.py`: in-memory fallback limiter
- `gateway/app/circuit_breaker.py`: downstream protection

### Worker

Location: `worker/app/`

The worker is a private HTTP service focused on model execution.

Responsibilities:

- ASR model download/load on startup
- supported-language resolution
- optional LID when requested language is `auto`
- audio preparation and resampling
- inference timeout handling
- threadpool-based inference execution
- soft fallback responses instead of hard process failures

Important files:

- `worker/app/main.py`: `/v1/transcribe`, worker concurrency, fallback behavior
- `worker/app/model.py`: model load, language resolution, inference execution
- `worker/app/lid.py`: language ID providers and fallback chain
- `worker/app/config.py`: worker runtime knobs

## 4. End-to-End Request Flow

### WebSocket flow

1. Client connects to `/ws/stt`.
2. Gateway authenticates using `Api-Subscription-Key` or `Sec-WebSocket-Protocol`.
3. Gateway validates required query params like `language-code`, `sample_rate`, and `input_audio_codec`.
4. Gateway admits or rejects the session using Redis limiter or in-memory fallback limiter.
5. Client sends JSON `audio` messages containing base64 payloads.
6. Gateway decodes and normalizes the audio into PCM16.
7. Gateway feeds 20 ms frames into `VADSegmenter`.
8. On `speech_end`, `max_utt`, or explicit `{"type":"flush"}`, gateway finalizes the utterance.
9. Gateway sends the utterance bytes to the worker over HTTP.
10. Worker returns `{text, language, language_source}`.
11. Gateway emits a final `type:"data"` message to the client.

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

## 7. Environment and Key Runtime Knobs

Defaults live in `.env.example`.

Most important gateway knobs:

- `REDIS_URL`
- `GATEWAY_DISABLE_RATE_LIMITING`
- `MAX_CONNS_PER_KEY`
- `NEW_CONN_PER_MIN`
- `CONN_BURST`
- `MAX_BYTES_PER_SEC`
- `WS_DISABLE_AUDIO_RATE_LIMIT`
- `GATEWAY_MAX_INFLIGHT_WORKER`
- `WORKER_TIMEOUT_MS`
- `WORKER_URL`
- `VAD_END_SILENCE_MS`
- `VAD_KEEP_SILENCE_MS`
- `VAD_MAX_UTT_MS`
- `CIRCUIT_BREAKER_FAILS`
- `CIRCUIT_BREAKER_RESET_MS`
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
- Redis/session rate limits

### 4. Sessions close with `TOO_MUCH_DATA`

Check:

- incoming audio frame size and send rate
- `MAX_BYTES_PER_SEC`
- whether `WS_DISABLE_AUDIO_RATE_LIMIT` is intentionally disabled in local testing

### 5. Worker returns `worker-fallback`

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

## 17. Suggested Day-1 KT Walkthrough

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

## 18. Suggested Next Documentation Improvements

Good follow-up docs to add later:

- sequence diagram for websocket -> worker flow
- incident runbook for worker degraded state
- language support matrix generated from the loaded model snapshot
- deployment notes for EC2/NVIDIA driver setup
- performance tuning guide for concurrency and latency tradeoffs

## 19. Summary

If you remember only three things:

- gateway is the control plane for sessions, validation, throttling, and segmentation
- worker is the execution plane for language resolution and GPU ASR inference
- most production debugging starts by correlating one `session_id` and `utterance_id` across gateway logs, worker logs, and metrics
