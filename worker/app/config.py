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


def getenv_csv(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    if not raw:
        return tuple()

    items: list[str] = []
    for part in raw.split(","):
        value = part.strip().lower()
        if value and value not in items:
            items.append(value)
    return tuple(items)


ASR_MODEL_NAME = getenv_str("ASR_MODEL_NAME", "")
ASR_MODEL_CACHE_DIR = getenv_str("ASR_MODEL_CACHE_DIR", "models/cache")
ASR_PRELOAD_MODELS = getenv_bool("ASR_PRELOAD_MODELS", True)
ASR_DECODER = getenv_str("ASR_DECODER", "rnnt")
ASR_INFERENCE_TIMEOUT_MS = getenv_int("ASR_INFERENCE_TIMEOUT_MS", 4000)
ASR_DEFAULT_LANGUAGE = getenv_str("ASR_DEFAULT_LANGUAGE", "hi")
ASR_SUPPORTED_LANGS = getenv_csv("ASR_SUPPORTED_LANGS")
ASR_ENABLE_EN_ENGINE = getenv_bool("ASR_ENABLE_EN_ENGINE", True)
ASR_EN_MODEL_NAME = getenv_str(
    "ASR_EN_MODEL_NAME",
    "stt_en_fastconformer_hybrid_large_streaming_80ms",
)
ASR_EN_MODEL_DEVICE = getenv_str(
    "ASR_EN_MODEL_DEVICE",
    "cuda",
)
ASR_ENABLE_LID = getenv_bool("ASR_ENABLE_LID", False)
ASR_ENABLE_LID_RECHECK = getenv_bool("ASR_ENABLE_LID_RECHECK", False)
ASR_LID_MODEL_SOURCE = getenv_str("ASR_LID_MODEL_SOURCE", "speechbrain/lang-id-voxlingua107-ecapa")
ASR_LID_MODEL_DIR = getenv_str("ASR_LID_MODEL_DIR", "models/lid_model")
ASR_LID_CACHE_TTL_SEC = max(1, getenv_int("ASR_LID_CACHE_TTL_SEC", 600))
ASR_LID_CACHE_MAX_ENTRIES = max(1, getenv_int("ASR_LID_CACHE_MAX_ENTRIES", 10000))
LID_MIN_SPEECH_MS = 500
LID_DETECT_WINDOW_MS = 1000
HUGGINGFACE_HUB_TOKEN = getenv_str("HUGGINGFACE_HUB_TOKEN", getenv_str("HF_TOKEN", ""))
WORKER_MAX_JOBS = getenv_int("WORKER_MAX_JOBS", 2)
LOG_LEVEL = getenv_str("LOG_LEVEL", "INFO")
EVAL_LOGS_ENABLED = getenv_bool("EVAL_LOGS_ENABLED", False)
EVAL_LOG_SAMPLE_RATE = min(1.0, max(0.0, getenv_float("EVAL_LOG_SAMPLE_RATE", 1.0)))
EVAL_LOG_TEXT_PREVIEW_CHARS = max(0, getenv_int("EVAL_LOG_TEXT_PREVIEW_CHARS", 16))
