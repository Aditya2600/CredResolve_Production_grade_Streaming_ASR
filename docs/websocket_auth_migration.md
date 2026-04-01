# STT WebSocket Migration

Backward compatibility is intentionally off. `/ws/stt` now follows a Sarvam-like session contract.

## Before

- Client connected to `/ws/stt`
- Auth lived in a JSON `start` frame
- Client sent raw binary PCM frames
- Client ended the session with `{"type":"stop"}`
- Server emitted `ready`, `partial`, `final`, `done`, and `error`

Example:

```json
{"type":"start","call_id":"test123","sample_rate":16000,"encoding":"pcm_s16le","frame_ms":20,"language":"auto","decoder":"rnnt"}
```

## After

- Client still connects to `/ws/stt`
- Auth comes from the handshake:
  - canonical: `Api-Subscription-Key: <token>`
  - browser fallback: `Sec-WebSocket-Protocol: token,<token>`
- Session config comes from query params
- Client sends JSON `audio` messages with base64 payloads
- Client finalizes buffered audio with `{"type":"flush"}`
- Server emits final-only `type:"data"` transcript payloads and optional `type:"vad"` signals
- For local/dev benchmarking, `WS_DISABLE_AUDIO_RATE_LIMIT=true` can bypass the websocket audio-byte limiter without changing the separate connection/session limiter

## Handshake

Required query param:

- `language-code`

Supported query params:

- `model` default `credresolve:v1`
- `mode` default `transcribe`
- `sample_rate` default `16000`, allowed `8000|16000`
- `high_vad_sensitivity` boolean
- `vad_signals` boolean
- `flush_signal` boolean, accepted for compatibility
- `input_audio_codec` default `pcm_s16le`, allowed `wav|pcm_s16le|pcm_l16|pcm_raw`

Example URL:

```text
ws://localhost/ws/stt?language-code=hi&model=credresolve:v1&mode=transcribe&sample_rate=16000&vad_signals=true&flush_signal=true&input_audio_codec=pcm_s16le
```

Example headers:

```http
Api-Subscription-Key: dev
```

Browser fallback:

```http
Sec-WebSocket-Protocol: token,dev
```

The gateway validates tokens against `WS_API_KEYS` in `.env` / `.env.example`:

```env
WS_API_KEYS=dev
```

## Message Shapes

Audio message:

```json
{
  "audio": {
    "data": "<base64>",
    "sample_rate": "16000",
    "encoding": "pcm_s16le"
  }
}
```

Flush message:

```json
{"type":"flush"}
```

Main transcript response:

```json
{
  "type": "data",
  "data": {
    "request_id": "f7f4d6f1-92cf-4a5f-9034-8b6de6f62d8f",
    "transcript": "namaste mera naam aditya hai",
    "language_code": "hi",
    "language_source": "client",
    "metrics": {
      "audio_duration": 1.24,
      "processing_latency": 0.18
    }
  }
}
```

Optional VAD response when `vad_signals=true`:

```json
{
  "type": "vad",
  "data": {
    "request_id": "f7f4d6f1-92cf-4a5f-9034-8b6de6f62d8f",
    "event": "speech_start"
  }
}
```

Error response:

```json
{
  "type": "error",
  "code": "VALIDATION_ERROR",
  "message": "missing required query param: language-code"
}
```

## Example Command

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
