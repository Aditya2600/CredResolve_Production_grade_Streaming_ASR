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
- If LID is disabled/unavailable/unmappable, worker falls back to `ASR_DEFAULT_LANGUAGE`.
- Worker `/v1/transcribe` responses include:
  - `text`
  - `language`
  - `language_source` (`client`, `auto_default`, `lid_detected`, `lid_cached`, `lid_fallback_default`)
- Gateway websocket `partial` and `final` events forward `language` and `language_source`.
- **Language Metadata**:
  - `language`: Resolved language code (e.g. `hi`, `en`, `te`).
  - `language_source`: How the language was determined:
    - `client`: Explicitly requested by user.
    - `lid_detected`: Detect by Language ID model.
    - `lid_cached`: Used cached result from previous utterance in session.
    - `auto_default` / `lid_fallback_*`: Fallback to default language.

---

## Key knobs (traffic management)

- `MAX_CONNS_PER_KEY`
- `NEW_CONN_PER_MIN` + `CONN_BURST`
- `MAX_BYTES_PER_SEC`
- `GATEWAY_MAX_INFLIGHT_WORKER`
- `WORKER_MAX_JOBS`
- `WORKER_TIMEOUT_MS`
- `CIRCUIT_BREAKER_FAILS` + `CIRCUIT_BREAKER_RESET_MS`
- `ASR_ENABLE_LID`
- `ASR_LID_MODEL_SOURCE`
- `ASR_LID_MODEL_DIR`
- `ASR_LID_CACHE_TTL_SEC`
- `ASR_LID_CACHE_MAX_ENTRIES`
- `ASR_SUPPORTED_LANGS`

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
