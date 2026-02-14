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

---

## Key knobs (traffic management)

- `MAX_CONNS_PER_KEY`
- `NEW_CONN_PER_MIN` + `CONN_BURST`
- `MAX_BYTES_PER_SEC`
- `GATEWAY_MAX_INFLIGHT_WORKER`
- `WORKER_MAX_JOBS`
- `WORKER_TIMEOUT_MS`
- `CIRCUIT_BREAKER_FAILS` + `CIRCUIT_BREAKER_RESET_MS`
