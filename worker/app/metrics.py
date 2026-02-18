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

# Legacy/Unused (kept if needed or remove if safe)
FALLBACKS = Counter("asr_worker_fallback_total", "Worker fallback count", ["reason"])
MODEL_INIT = Counter("asr_worker_model_init_total", "Model initialization count", ["status"])
