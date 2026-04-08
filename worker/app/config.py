import os

from .context_biasing import normalize_context_biasing_method, normalize_context_biasing_mode


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


def getenv_optional_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


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


def infer_lid_provider(source: str, default: str = "speechbrain") -> str:
    normalized = (source or "").strip().lower()
    if "speechbrain" in normalized or "voxlingua" in normalized:
        return "speechbrain"
    if "vakgyata" in normalized or "onecxi/" in normalized:
        return "vakgyata"
    return default


ASR_MODEL_NAME = getenv_str("ASR_MODEL_NAME", "")
ASR_BACKEND = getenv_str("ASR_BACKEND", "local").lower()
ASR_DECODER = getenv_str("ASR_DECODER", "rnnt")
ASR_INFERENCE_TIMEOUT_MS = getenv_int("ASR_INFERENCE_TIMEOUT_MS", 4000)
ASR_DEFAULT_LANGUAGE = getenv_str("ASR_DEFAULT_LANGUAGE", "hi")
ASR_SUPPORTED_LANGS = getenv_csv("ASR_SUPPORTED_LANGS")
TRITON_URL = getenv_str("TRITON_URL", "triton:8000")
TRITON_MODEL_NAME = getenv_str("TRITON_MODEL_NAME", "indic_asr")
TRITON_MODEL_VERSION = getenv_str("TRITON_MODEL_VERSION", "")
ASR_ENABLE_LID = getenv_bool("ASR_ENABLE_LID", False)
ASR_LID_MODEL_SOURCE = getenv_str("ASR_LID_MODEL_SOURCE", "speechbrain/lang-id-voxlingua107-ecapa")
ASR_LID_MODEL_DIR = getenv_str("ASR_LID_MODEL_DIR", "models/lid_model")
_HAS_NEW_LID_CHAIN_CONFIG = any(
    getenv_optional_str(name) is not None
    for name in (
        "ASR_LID_PRIMARY_PROVIDER",
        "ASR_LID_PRIMARY_SOURCE",
        "ASR_LID_PRIMARY_MODEL_DIR",
        "ASR_LID_FALLBACK_PROVIDER",
        "ASR_LID_FALLBACK_SOURCE",
        "ASR_LID_FALLBACK_MODEL_DIR",
        "ASR_LID_CONFIDENCE_THRESHOLD",
    )
)
ASR_LID_PRIMARY_PROVIDER = (
    getenv_str("ASR_LID_PRIMARY_PROVIDER", "vakgyata")
    if _HAS_NEW_LID_CHAIN_CONFIG
    else infer_lid_provider(ASR_LID_MODEL_SOURCE, default="speechbrain")
)
ASR_LID_PRIMARY_SOURCE = (
    getenv_str("ASR_LID_PRIMARY_SOURCE", "onecxi/vakgyata-small")
    if _HAS_NEW_LID_CHAIN_CONFIG
    else ASR_LID_MODEL_SOURCE
)
ASR_LID_PRIMARY_MODEL_DIR = (
    getenv_str("ASR_LID_PRIMARY_MODEL_DIR", "models/lid_primary")
    if _HAS_NEW_LID_CHAIN_CONFIG
    else ASR_LID_MODEL_DIR
)
ASR_LID_FALLBACK_PROVIDER = (
    getenv_str("ASR_LID_FALLBACK_PROVIDER", "speechbrain")
    if _HAS_NEW_LID_CHAIN_CONFIG
    else ""
)
ASR_LID_FALLBACK_SOURCE = (
    getenv_str("ASR_LID_FALLBACK_SOURCE", "speechbrain/lang-id-voxlingua107-ecapa")
    if _HAS_NEW_LID_CHAIN_CONFIG
    else ""
)
ASR_LID_FALLBACK_MODEL_DIR = (
    getenv_str("ASR_LID_FALLBACK_MODEL_DIR", "models/lid_fallback")
    if _HAS_NEW_LID_CHAIN_CONFIG
    else ""
)
ASR_LID_CONFIDENCE_THRESHOLD = min(1.0, max(0.0, getenv_float("ASR_LID_CONFIDENCE_THRESHOLD", 0.70)))
ASR_LID_CACHE_TTL_SEC = max(1, getenv_int("ASR_LID_CACHE_TTL_SEC", 600))
ASR_LID_CACHE_MAX_ENTRIES = max(1, getenv_int("ASR_LID_CACHE_MAX_ENTRIES", 10000))
ASR_CONTEXT_BIASING_MODE = normalize_context_biasing_mode(getenv_str("ASR_CONTEXT_BIASING_MODE", "disabled"))
ASR_CONTEXT_BIASING_METHOD = normalize_context_biasing_method(getenv_str("ASR_CONTEXT_BIASING_METHOD", "ctc_ws"))
ASR_CONTEXT_BIASING_NEMO_SOURCE = getenv_str("ASR_CONTEXT_BIASING_NEMO_SOURCE", "")
ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS = getenv_str("ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS", "")
ASR_CONTEXT_BIASING_PHRASES_DIR = getenv_str("ASR_CONTEXT_BIASING_PHRASES_DIR", "")
ASR_CONTEXT_BIASING_TIMEOUT_MS = max(
    1,
    getenv_int("ASR_CONTEXT_BIASING_TIMEOUT_MS", ASR_INFERENCE_TIMEOUT_MS),
)
ASR_CONTEXT_BIASING_DEVICE = getenv_str("ASR_CONTEXT_BIASING_DEVICE", "cuda")
ASR_CONTEXT_BIASING_SHADOW_SAMPLE_RATE = min(
    1.0,
    max(0.0, getenv_float("ASR_CONTEXT_BIASING_SHADOW_SAMPLE_RATE", 1.0)),
)
ASR_CONTEXT_BIASING_BEAM_THRESHOLD = max(0.0, getenv_float("ASR_CONTEXT_BIASING_BEAM_THRESHOLD", 8.0))
ASR_CONTEXT_BIASING_CONTEXT_SCORE = max(0.0, getenv_float("ASR_CONTEXT_BIASING_CONTEXT_SCORE", 3.0))
ASR_CONTEXT_BIASING_CTC_ALI_TOKEN_WEIGHT = max(
    0.0,
    getenv_float("ASR_CONTEXT_BIASING_CTC_ALI_TOKEN_WEIGHT", 0.6),
)
HUGGINGFACE_HUB_TOKEN = getenv_str("HUGGINGFACE_HUB_TOKEN", getenv_str("HF_TOKEN", ""))
WORKER_MAX_JOBS = getenv_int("WORKER_MAX_JOBS", 2)
LOG_LEVEL = getenv_str("LOG_LEVEL", "INFO")
EVAL_LOGS_ENABLED = getenv_bool("EVAL_LOGS_ENABLED", False)
EVAL_LOG_SAMPLE_RATE = min(1.0, max(0.0, getenv_float("EVAL_LOG_SAMPLE_RATE", 1.0)))
EVAL_LOG_TEXT_PREVIEW_CHARS = max(0, getenv_int("EVAL_LOG_TEXT_PREVIEW_CHARS", 16))
