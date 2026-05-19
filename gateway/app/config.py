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


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return default

def getenv_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def getenv_csv_set(name: str, default: str = "") -> frozenset[str]:
    raw = os.environ.get(name, default)
    return frozenset(part.strip() for part in raw.split(",") if part.strip())

GATEWAY_MAX_INFLIGHT_WORKER = getenv_int("GATEWAY_MAX_INFLIGHT_WORKER", 4)
WORKER_TIMEOUT_MS = getenv_int("WORKER_TIMEOUT_MS", 15000)
WORKER_URL = getenv_str("WORKER_URL", "http://localhost:9000")
ITN_GRPC_TARGET = getenv_str("ITN_GRPC_TARGET", "")
ITN_TIMEOUT_MS = max(1, getenv_int("ITN_TIMEOUT_MS", 250))
ITN_LOCALE_POLICY = getenv_str("ITN_LOCALE_POLICY", "")
PARTIAL_DECODE_INTERVAL_MS = max(100, getenv_int("PARTIAL_DECODE_INTERVAL_MS", 900))
STREAMING_DENOISE_ENABLED = getenv_bool("STREAMING_DENOISE_ENABLED", True)
STREAMING_VAD_ENABLED = getenv_bool("STREAMING_VAD_ENABLED", False)
STREAMING_RING_BUFFER_MS = max(600, getenv_int("STREAMING_RING_BUFFER_MS", 600))
STREAMING_MIN_FINAL_AUDIO_MS = max(0, getenv_int("STREAMING_MIN_FINAL_AUDIO_MS", 0))
STREAMING_VAD_MODE = getenv_int("STREAMING_VAD_MODE", 3)
STREAMING_GATE_OPEN_WINDOW_FRAMES = max(1, getenv_int("STREAMING_GATE_OPEN_WINDOW_FRAMES", 3))
STREAMING_GATE_OPEN_REQUIRED_VOICED_FRAMES = max(
    1,
    getenv_int("STREAMING_GATE_OPEN_REQUIRED_VOICED_FRAMES", 2),
)
STREAMING_GATE_CLOSE_WINDOW_FRAMES = max(
    1,
    getenv_int("STREAMING_GATE_CLOSE_WINDOW_FRAMES", 5),
)
STREAMING_GATE_CLOSE_REQUIRED_UNVOICED_FRAMES = max(
    1,
    getenv_int("STREAMING_GATE_CLOSE_REQUIRED_UNVOICED_FRAMES", 4),
)
STREAMING_HANGOVER_MS = max(20, getenv_int("STREAMING_HANGOVER_MS", 200))
SPEAKER_VERIFICATION_MODE = getenv_str("SPEAKER_VERIFICATION_MODE", "disabled").lower()
SPEAKER_VERIFICATION_BACKEND = getenv_str(
    "SPEAKER_VERIFICATION_BACKEND",
    "external",
).lower()
SPEAKER_VERIFICATION_THRESHOLD = min(
    1.0,
    max(0.0, getenv_float("SPEAKER_VERIFICATION_THRESHOLD", 0.70)),
)
SPEAKER_VERIFICATION_FIRST_DECISION_MS = getenv_int(
    "SPEAKER_VERIFICATION_FIRST_DECISION_MS",
    360,
)
SPEAKER_VERIFICATION_RESCORE_MS = getenv_int(
    "SPEAKER_VERIFICATION_RESCORE_MS",
    200,
)
SPEAKER_VERIFICATION_ENROLLED_EMBEDDING_PATH = getenv_str(
    "SPEAKER_VERIFICATION_ENROLLED_EMBEDDING_PATH",
    "",
)
SPEAKER_VERIFICATION_DEBUG_SIMILARITY = max(
    -1.0,
    min(1.0, getenv_float("SPEAKER_VERIFICATION_DEBUG_SIMILARITY", 0.95)),
)

LOG_LEVEL = getenv_str("LOG_LEVEL", "INFO")
GATEWAY_LOG_FILE = getenv_str("GATEWAY_LOG_FILE", "")
LOG_TRANSCRIPTS = getenv_bool("LOG_TRANSCRIPTS", False)
WS_API_KEYS = getenv_csv_set("WS_API_KEYS", "dev")
