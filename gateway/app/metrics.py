from prometheus_client import Counter, Histogram, Gauge

WS_CONNECTIONS = Gauge("asr_ws_connections", "Active WebSocket connections")
WS_REJECTS = Counter("asr_ws_rejects_total", "Rejected WS connections", ["reason"])
WS_DISCONNECTS = Counter("asr_ws_disconnects_total", "WebSocket disconnects", ["reason"])

AUDIO_BYTES_RECEIVED = Counter("asr_audio_bytes_received_total", "Total audio bytes received from clients")
AUDIO_FRAMES_RECEIVED = Counter("asr_audio_frames_received_total", "Total audio frames/chunks received")
UTTERANCES = Counter("asr_utterances_total", "Total finalized utterances")

VAD_FRAMES = Counter("asr_vad_frames_total", "VAD frames processed", ["state"])
LONG_SILENCE_PROMPTS = Counter("asr_long_silence_prompts_total", "Long silence prompts triggered", ["threshold"])

# Backward-compatible metrics used by worker_client.py.
WORKER_CALLS = Counter("asr_worker_calls_total", "Gateway->worker HTTP calls", ["mode", "status"])
WORKER_LATENCY = Histogram(
    "asr_worker_latency_seconds",
    "Gateway->worker HTTP latency",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8),
)

GATEWAY_LATENCY = Histogram(
    "asr_gateway_to_worker_seconds",
    "Time in gateway logic per utterance before worker call",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8),
)
E2E_LATENCY = Histogram(
    "asr_e2e_seconds",
    "End-to-end latency from first audio frame to final transcript",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8),
)
