from prometheus_client import Counter, Histogram

REQS = Counter("asr_worker_requests_total", "Worker requests", ["mode", "status"])
LAT = Histogram("asr_worker_latency_seconds", "Worker latency seconds", buckets=(0.05,0.1,0.2,0.5,1,2,5))
FALLBACKS = Counter("asr_worker_fallback_total", "Worker fallback count", ["reason"])
MODEL_INIT = Counter("asr_worker_model_init_total", "Model initialization count", ["status"])
