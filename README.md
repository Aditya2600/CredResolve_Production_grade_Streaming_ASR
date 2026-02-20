# CredResolve – Production-grade Streaming ASR (IndicConformer) over WebSocket

This repo is a production-ready reference implementation for a **streaming-ish** STT service:
- WebSocket (WSS) ingest of **audio bytes**
- VAD / endpointing (WebRTC VAD)
- Rate limiting + connection quotas (Redis token bucket + active-conn tracking)
- Traffic management (backpressure, GPU worker concurrency caps, load shedding, timeouts, circuit breaker)
- Split architecture:
  - `gateway` (CPU): WebSocket sessions + auth + VAD + partial/final events
  - `worker` (GPU): IndicConformer ONNX/TorchScript transcription service

---

## Architecture

Client/Telephony → Nginx (TLS + basic limits) → gateway (WS) → worker (HTTP, GPU) → transcripts

---

## Quickstart (Docker Compose on EC2)

```bash
cp .env.example .env
docker system df   # preflight: check free Docker disk space
docker compose up --build -d
docker compose logs -f gateway
```

Test (put a 16kHz mono PCM16 wav at `sample_data/sample_16k_mono.wav`):
```bash
python3 tools/ws_client_send_wav.py --ws ws://localhost/ws/stt --wav sample_data/sample_16k_mono.wav --api-key dev --language hi
```

Supported languages include `hi`, `en`, `bn`, `ta`, etc. (Check model documentation for full list).

`--language auto` enables runtime LID when worker env `ASR_ENABLE_LID=true`.

---

## Language Auto Mode (LID)

- LID runs only when requested language is `auto` (or empty).
- LID is CPU-only and feature-flagged (`ASR_ENABLE_LID=false` by default).
- Optional mid-utterance recheck can be enabled with `ASR_ENABLE_LID_RECHECK=true`.
- Language is resolved once per utterance (`speech_start` -> `speech_end`/`max_utt`) and locked for that utterance.
- LID inspects the first ~1.0s of speech audio on utterance start, then cache is reused for remaining partial/final events of the same utterance.
- Language can switch on the next utterance (for borrower mid-call language switching).
- Very short utterances (`<500ms`) skip LID and fall back to session `last_language` or `ASR_DEFAULT_LANGUAGE`.
- With recheck enabled, worker evaluates the last ~1.0s window every ~1.0s and can switch language mid-utterance using anti-jitter voting.
- Worker `/v1/transcribe` responses include:
  - `text`
  - `language`
  - `language_source` (`client`, `auto_default`, `lid_detected`, `lid_cached`, `lid_fallback_default`, `lid_recheck`)
- Gateway websocket `partial` and `final` events forward `language` and `language_source`.
- **Language Metadata**:
  - `language`: Resolved language code (e.g. `hi`, `en`, `te`).
  - `language_source`: How the language was determined:
    - `client`: Explicitly requested by user.
    - `lid_detected`: Detect by Language ID model.
    - `lid_cached`: Used cached result for current utterance lock.
    - `lid_recheck`: Mid-utterance recheck switched language.
    - `auto_default`: `auto` mode with LID disabled/unavailable; falls back to session last/default.
    - `lid_fallback_default`: LID failed, unmappable label, or short utterance fallback.

Example timeline (`language=auto`):
- `utt-0001` first partial -> `lid_detected` `hi`; final -> `lid_cached` `hi`
- `utt-0002` first partial -> `lid_detected` `en`; final -> `lid_cached` `en`
- `utt-0003` very short speech -> `lid_fallback_default` `en` (uses previous utterance language)

---

## Key knobs (traffic management)

- `MAX_CONNS_PER_KEY`
- `NEW_CONN_PER_MIN` + `CONN_BURST`
- `MAX_BYTES_PER_SEC`
- `GATEWAY_MAX_INFLIGHT_WORKER`
- `WORKER_MAX_JOBS`
- `WORKER_TIMEOUT_MS`
- `GATEWAY_WS_PING_INTERVAL`
- `GATEWAY_WS_PING_TIMEOUT`
- `PARTIAL_DECODE_INTERVAL_MS`
- `VAD_END_SILENCE_MS`
- `VAD_MAX_UTT_MS`
- `CIRCUIT_BREAKER_FAILS` + `CIRCUIT_BREAKER_RESET_MS`
- `ASR_ENABLE_LID`
- `ASR_ENABLE_LID_RECHECK`
- `ASR_LID_MODEL_SOURCE`
- `ASR_LID_MODEL_DIR`
- `ASR_LID_CACHE_TTL_SEC`
- `ASR_LID_CACHE_MAX_ENTRIES`
- `ASR_SUPPORTED_LANGS`

## Dual Engine Routing (Indic + English)

- In-house routing supports two engines per utterance:
  - Indic utterances -> existing AI4Bharat IndicConformer ONNX engine
  - English utterances (`language=en`) -> self-hosted NeMo English engine
- Worker accepts `8kHz` and `16kHz` PCM16 input and normalizes to `16kHz mono` exactly once before LID/decoding.
- Routing is utterance-locked via current LID behavior in `language=auto`.
- Explicit language behavior:
  - `language=en` forces English engine
  - other explicit languages force Indic engine
- If English engine is disabled/unavailable, worker does **not** crash:
  - `auto` requests fall back to last/default language with `language_source=auto_default`.

Worker env knobs:
- `ASR_ENABLE_EN_ENGINE`
- `ASR_EN_MODEL_NAME`
- `ASR_EN_MODEL_DEVICE`
- `ASR_MODEL_CACHE_DIR`
- `ASR_PRELOAD_MODELS`

Engine observability:
- `asr_engine_selected_total{engine=\"indic|en\",language,mode}`
- `asr_engine_latency_seconds{engine}`
- `asr_engine_fallback_total{reason=\"en_engine_unavailable|unsupported_language|error\"}`

Optional English engine dependency:
- install `worker/requirements-en.txt` in worker image/runtime when enabling NeMo engine.
- Docker Compose build toggle: set `WORKER_INSTALL_EN_ENGINE=true` before `docker compose build`.

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
*   **Zero connections?** Connect a client! (e.g. Frontend or `wscat`).
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

```bash
# Gateway
cd gateway
uvicorn app.main:app --port 8000

# Worker
cd worker
python -m app.main
```

#sample
