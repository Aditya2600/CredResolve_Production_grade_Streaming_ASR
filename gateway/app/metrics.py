from prometheus_client import Counter, Histogram, Gauge

WS_CONNECTIONS = Gauge("asr_ws_connections", "Active WebSocket connections")
WS_REJECTS = Counter("asr_ws_rejects_total", "Rejected WS connections", ["reason"])

WORKER_CALLS = Counter("asr_worker_calls_total", "Worker decode calls", ["mode", "status"])
WORKER_LATENCY = Histogram("asr_worker_latency_seconds", "Worker latency seconds", buckets=(0.05,0.1,0.2,0.5,1,2,5))

VAD_UTTERANCES = Counter("asr_vad_utterances_total", "Utterances finalized by VAD")
BYTES_IN = Counter("asr_audio_bytes_in_total", "Audio bytes received")
