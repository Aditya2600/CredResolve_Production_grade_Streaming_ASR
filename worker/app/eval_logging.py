import hashlib
import json
import time
from typing import Any, Optional

from .config import EVAL_LOG_SAMPLE_RATE, EVAL_LOG_TEXT_PREVIEW_CHARS, EVAL_LOGS_ENABLED

SERVICE_NAME = "worker"


def _sample_key(session_id: Optional[str], utterance_id: Optional[str]) -> str:
    if session_id:
        return session_id
    if utterance_id:
        return utterance_id
    return "worker-default"


def _sample_ratio(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float((1 << 64) - 1)


def should_sample(session_id: Optional[str] = None, utterance_id: Optional[str] = None) -> bool:
    if not EVAL_LOGS_ENABLED:
        return False
    if EVAL_LOG_SAMPLE_RATE <= 0.0:
        return False
    if EVAL_LOG_SAMPLE_RATE >= 1.0:
        return True
    key = _sample_key(session_id=session_id, utterance_id=utterance_id)
    return _sample_ratio(key) <= EVAL_LOG_SAMPLE_RATE


def hash_value(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text_metadata(text: str) -> dict[str, Any]:
    safe_text = (text or "").strip()
    preview = safe_text[:EVAL_LOG_TEXT_PREVIEW_CHARS] if EVAL_LOG_TEXT_PREVIEW_CHARS > 0 else ""
    return {
        "text_chars": len(safe_text),
        "text_preview": preview,
        "text_sha256": hash_value(safe_text),
    }


def emit_eval_event(
    logger: Any,
    event: str,
    *,
    session_id: Optional[str] = None,
    utterance_id: Optional[str] = None,
    sampled: Optional[bool] = None,
    **fields: Any,
) -> None:
    if not EVAL_LOGS_ENABLED:
        return
    sampled_value = should_sample(session_id=session_id, utterance_id=utterance_id) if sampled is None else sampled
    if not sampled_value:
        return

    payload: dict[str, Any] = {
        "event": event,
        "ts_ms": int(time.time() * 1000),
        "service": SERVICE_NAME,
        "session_id": session_id,
        "utterance_id": utterance_id,
        "sampled": sampled_value,
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    logger.info(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
