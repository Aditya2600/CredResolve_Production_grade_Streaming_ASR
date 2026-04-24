from prometheus_client import Counter, Histogram, Gauge

WS_CONNECTIONS = Gauge("asr_ws_connections", "Active WebSocket connections")
WS_REJECTS = Counter("asr_ws_rejects_total", "Rejected WS connections", ["reason"])
WS_DISCONNECTS = Counter("asr_ws_disconnects_total", "WebSocket disconnects", ["reason"])

AUDIO_BYTES_RECEIVED = Counter("asr_audio_bytes_received_total", "Total audio bytes received from clients")
AUDIO_FRAMES_RECEIVED = Counter("asr_audio_frames_received_total", "Total audio frames/chunks received")
UTTERANCES = Counter("asr_utterances_total", "Total finalized utterances")

VAD_FRAMES = Counter("asr_vad_frames_total", "VAD frames processed", ["state"])
LONG_SILENCE_PROMPTS = Counter("asr_long_silence_prompts_total", "Long silence prompts triggered", ["threshold"])
GATE_TRANSITIONS = Counter(
    "asr_gate_transitions_total",
    "Gate state transitions",
    ["from_state", "to_state", "reason"],
)
SPEAKER_SCORE_EVENTS = Counter(
    "asr_speaker_score_events_total",
    "Speaker verification score decisions",
    ["mode", "decision"],
)
SPEAKER_SIMILARITY = Histogram(
    "asr_speaker_similarity",
    "Speaker verification cosine similarity",
    buckets=(-1.0, -0.5, 0.0, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 0.95, 1.0),
)
SPEAKER_FALSE_ACCEPTS = Counter(
    "asr_speaker_false_accepts_total",
    "Manually recorded false speaker accepts",
)
SPEAKER_FALSE_REJECTS = Counter(
    "asr_speaker_false_rejects_total",
    "Manually recorded false speaker rejects",
)
TIME_TO_FIRST_GATE_OPEN = Histogram(
    "asr_time_to_first_gate_open_seconds",
    "Time from first speech frame to first gate open",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8),
)
FINAL_TRANSCRIPT_LATENCY = Histogram(
    "asr_final_transcript_latency_seconds",
    "Latency from first speech frame to final transcript",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8),
)

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
