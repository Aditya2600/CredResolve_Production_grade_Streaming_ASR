import os


def getenv_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default

def getenv_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def getenv_csv_set(name: str, default: str = "") -> frozenset[str]:
    raw = os.environ.get(name, default)
    return frozenset(part.strip() for part in raw.split(",") if part.strip())

REDIS_URL = getenv_str("REDIS_URL", "redis://localhost:6379/0")
GATEWAY_DISABLE_RATE_LIMITING = getenv_bool("GATEWAY_DISABLE_RATE_LIMITING", False)
MAX_CONNS_PER_KEY = getenv_int("MAX_CONNS_PER_KEY", 5)
NEW_CONN_PER_MIN = getenv_int("NEW_CONN_PER_MIN", 30)
CONN_BURST = getenv_int("CONN_BURST", 10)
MAX_BYTES_PER_SEC = getenv_int("MAX_BYTES_PER_SEC", 128000)
WS_DISABLE_AUDIO_RATE_LIMIT = getenv_bool("WS_DISABLE_AUDIO_RATE_LIMIT", False)

GATEWAY_MAX_INFLIGHT_WORKER = getenv_int("GATEWAY_MAX_INFLIGHT_WORKER", 4)
WORKER_TIMEOUT_MS = getenv_int("WORKER_TIMEOUT_MS", 2000)
WORKER_URL = getenv_str("WORKER_URL", "http://localhost:9000")
PARTIAL_DECODE_INTERVAL_MS = max(100, getenv_int("PARTIAL_DECODE_INTERVAL_MS", 900))
VAD_END_SILENCE_MS = max(100, getenv_int("VAD_END_SILENCE_MS", 320))
VAD_KEEP_SILENCE_MS = max(0, getenv_int("VAD_KEEP_SILENCE_MS", 120))
VAD_MAX_UTT_MS = max(1000, getenv_int("VAD_MAX_UTT_MS", 12000))

CIRCUIT_BREAKER_FAILS = getenv_int("CIRCUIT_BREAKER_FAILS", 5)
CIRCUIT_BREAKER_RESET_MS = getenv_int("CIRCUIT_BREAKER_RESET_MS", 15000)

LOG_LEVEL = getenv_str("LOG_LEVEL", "INFO")
WS_API_KEYS = getenv_csv_set("WS_API_KEYS", "dev")

EVAL_LOGS_ENABLED = getenv_bool("EVAL_LOGS_ENABLED", False)
EVAL_LOG_SAMPLE_RATE = min(1.0, max(0.0, getenv_float("EVAL_LOG_SAMPLE_RATE", 1.0)))
EVAL_LOG_TEXT_PREVIEW_CHARS = max(0, getenv_int("EVAL_LOG_TEXT_PREVIEW_CHARS", 16))
