from prometheus_client import Counter, Histogram, Gauge

# Counters
REQS = Counter("asr_worker_requests_total", "Worker requests", ["mode", "status"])
ERRORS = Counter("asr_worker_errors_total", "Worker internal errors", ["type"])

LID_REQS = Counter(
    "asr_worker_lid_requests_total",
    "LID resolution requests",
    ["status"],
)
LID_DETECTED = Counter(
    "asr_worker_lid_detected_total",
    "Detected supported languages from LID",
    ["language"],
)

# Gauges
INFLIGHT_REQUESTS = Gauge("asr_worker_inflight_requests", "Current active worker requests")
CIRCUIT_BREAKER_STATE = Gauge(
    "asr_worker_circuit_breaker_state",
    "Circuit breaker state; one series per state is set to 1 for the active state",
    ["name", "state"],
)

# Histograms
LAT = Histogram(
    "asr_worker_latency_seconds",
    "Worker full request latency",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5),
)
INFERENCE_LATENCY = Histogram(
    "asr_worker_inference_seconds",
    "Pure model inference time",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5),
)
LID_LAT = Histogram(
    "asr_worker_lid_latency_seconds",
    "LID latency seconds",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
)
TRITON_INFER_LATENCY = Histogram(
    "asr_worker_triton_infer_seconds",
    "Triton client.infer() round-trip latency, by protocol and model",
    ["protocol", "model"],
    buckets=(0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5),
)
CIRCUIT_BREAKER_CALLS = Counter(
    "asr_worker_circuit_breaker_calls_total",
    "Circuit breaker protected calls by observed state and result",
    ["name", "state", "result"],
)
CIRCUIT_BREAKER_TRANSITIONS = Counter(
    "asr_worker_circuit_breaker_transitions_total",
    "Circuit breaker state transitions",
    ["name", "from_state", "to_state", "reason"],
)

# Legacy/Unused (kept if needed or remove if safe)
FALLBACKS = Counter("asr_worker_fallback_total", "Worker fallback count", ["reason"])
MODEL_INIT = Counter("asr_worker_model_init_total", "Model initialization count", ["status"])
CONTEXT_BIASING_REQUESTS = Counter(
    "asr_worker_context_biasing_requests_total",
    "Context-biasing requests by mode and status",
    ["mode", "status"],
)
CONTEXT_BIASING_LATENCY = Histogram(
    "asr_worker_context_biasing_latency_seconds",
    "Context-biasing inference latency excluding queue wait",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12),
)
CONTEXT_BIASING_TOTAL_LATENCY = Histogram(
    "asr_worker_context_biasing_total_latency_seconds",
    "Context-biasing caller-visible total latency including queue wait",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12),
)
CONTEXT_BIASING_QUEUE_WAIT_MS = Histogram(
    "asr_worker_context_biasing_queue_wait_ms",
    "Context-biasing queue wait before a model lease is acquired",
    buckets=(1, 2.5, 5, 10, 20, 35, 50, 75, 100, 200, 350, 500, 750, 1000, 2000, 4000, 8000),
)
CONTEXT_BIASING_FALLBACKS = Counter(
    "asr_worker_context_biasing_fallback_total",
    "Context-biasing fallback count",
    ["reason"],
)
CONTEXT_BIASING_INFLIGHT = Gauge(
    "asr_worker_context_biasing_inflight",
    "Current context-biasing model leases running or draining after timeout",
)
CONTEXT_BIASING_POOL_AVAILABLE = Gauge(
    "asr_worker_context_biasing_pool_available",
    "Available context-biasing model instances",
)

# Audio preprocessing (VAD + denoise) instrumentation. Labels are kept low
# cardinality so swapping the denoiser/VAD can be evaluated against measurable
# criteria without dashboard explosions.
AUDIO_STAGE_LATENCY = Histogram(
    "asr_worker_audio_stage_seconds",
    "AudioPreprocessor.process() per-stage latency in seconds",
    ["stage", "vad_select_mode", "denoise_enabled"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0),
)
AUDIO_VAD_SEGMENTS = Histogram(
    "asr_worker_audio_vad_segments",
    "Number of VAD speech segments detected per process() call",
    ["vad_select_mode"],
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34),
)
AUDIO_FRAMES = Counter(
    "asr_worker_audio_frames_total",
    "Audio sample counts entering/leaving each preprocessing stage",
    ["stage", "direction"],
)
AUDIO_SPEECH_RATIO = Histogram(
    "asr_worker_audio_speech_ratio",
    "Per-call ratio of post-VAD speech samples to input samples",
    ["vad_select_mode"],
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0),
)
