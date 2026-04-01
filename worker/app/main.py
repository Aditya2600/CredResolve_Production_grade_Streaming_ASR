import asyncio
import logging
import time

from fastapi import FastAPI, Header, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import (
    ASR_DECODER,
    ASR_DEFAULT_LANGUAGE,
    ASR_ENABLE_LID,
    ASR_LID_CONFIDENCE_THRESHOLD,
    ASR_INFERENCE_TIMEOUT_MS,
    ASR_LID_CACHE_MAX_ENTRIES,
    ASR_LID_CACHE_TTL_SEC,
    ASR_LID_FALLBACK_MODEL_DIR,
    ASR_LID_FALLBACK_PROVIDER,
    ASR_LID_FALLBACK_SOURCE,
    ASR_LID_PRIMARY_MODEL_DIR,
    ASR_LID_PRIMARY_PROVIDER,
    ASR_LID_PRIMARY_SOURCE,
    ASR_MODEL_NAME,
    ASR_SUPPORTED_LANGS,
    HUGGINGFACE_HUB_TOKEN,
    WORKER_MAX_JOBS,
)
from .logging_setup import setup_logging
from .metrics import FALLBACKS, LAT, MODEL_INIT, REQS, ERRORS, INFLIGHT_REQUESTS, INFERENCE_LATENCY
from .eval_logging import emit_eval_event, should_sample, text_metadata
from .model import (
    InferenceError,
    InferenceTimeoutError,
    ModelNotReadyError,
    ONNXIndicASRWorker,
    UnsupportedLanguageError,
)

setup_logging()
log = logging.getLogger("worker")

app = FastAPI()
model = ONNXIndicASRWorker(
    model_name=ASR_MODEL_NAME,
    default_decoder=ASR_DECODER,
    hf_token=HUGGINGFACE_HUB_TOKEN,
    inference_timeout_ms=ASR_INFERENCE_TIMEOUT_MS,
    default_language=ASR_DEFAULT_LANGUAGE,
    supported_language_allowlist=ASR_SUPPORTED_LANGS,
    enable_lid=ASR_ENABLE_LID,
    lid_primary_provider=ASR_LID_PRIMARY_PROVIDER,
    lid_primary_source=ASR_LID_PRIMARY_SOURCE,
    lid_primary_model_dir=ASR_LID_PRIMARY_MODEL_DIR,
    lid_fallback_provider=ASR_LID_FALLBACK_PROVIDER,
    lid_fallback_source=ASR_LID_FALLBACK_SOURCE,
    lid_fallback_model_dir=ASR_LID_FALLBACK_MODEL_DIR,
    lid_confidence_threshold=ASR_LID_CONFIDENCE_THRESHOLD,
    lid_cache_ttl_sec=ASR_LID_CACHE_TTL_SEC,
    lid_cache_max_entries=ASR_LID_CACHE_MAX_ENTRIES,
    max_jobs=WORKER_MAX_JOBS,
)
sem = asyncio.Semaphore(WORKER_MAX_JOBS)


def lid_eval_fields(*, include_resolution: bool) -> dict[str, object]:
    fields: dict[str, object] = {
        "lid_enabled": model.enable_lid,
        "lid_available": model.lid_available,
        "lid_error": model.lid_last_error or None,
    }
    if include_resolution:
        fields.update(
            {
                "lid_provider": model.lid_last_provider or None,
                "lid_confidence": model.lid_last_confidence,
                "lid_fallback_from": model.lid_last_fallback_from or None,
                "lid_fallback_reason": model.lid_last_fallback_reason or None,
            }
        )
    return fields


@app.on_event("startup")
async def startup_event():
    log.info(
        "Worker startup model=%s decoder=%s default_language=%s lid_enabled=%s lid_primary_provider=%s lid_primary_source=%s lid_fallback_provider=%s lid_fallback_source=%s lid_confidence_threshold=%.2f timeout_ms=%s max_jobs=%s",
        ASR_MODEL_NAME,
        ASR_DECODER,
        ASR_DEFAULT_LANGUAGE,
        ASR_ENABLE_LID,
        ASR_LID_PRIMARY_PROVIDER,
        ASR_LID_PRIMARY_SOURCE,
        ASR_LID_FALLBACK_PROVIDER or "-",
        ASR_LID_FALLBACK_SOURCE or "-",
        ASR_LID_CONFIDENCE_THRESHOLD,
        ASR_INFERENCE_TIMEOUT_MS,
        WORKER_MAX_JOBS,
    )
    try:
        await asyncio.to_thread(model.load)
        MODEL_INIT.labels(status="ok").inc()
        log.info("Worker model ready")
    except Exception as exc:
        MODEL_INIT.labels(status="err").inc()
        log.exception("Worker model failed to initialize: %s", exc)


@app.get("/healthz")
async def healthz():
    if model.ready:
        return PlainTextResponse("ok")
    detail = model.init_error or "model-not-ready"
    return PlainTextResponse(f"degraded:{detail}")


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/transcribe")
async def transcribe(
    request: Request,
    x_sample_rate: str = Header(default="16000"),
    x_decoder: str = Header(default=ASR_DECODER),
    x_language: str = Header(default=ASR_DEFAULT_LANGUAGE),
    x_mode: str = Header(default="final"),
    x_session_id: str = Header(default=""),
    x_utterance_id: str = Header(default=""),
):
    pcm = await request.body()
    mode = (x_mode or "final").lower()
    session_id = (x_session_id or "").strip() or None
    utterance_id = (x_utterance_id or "").strip() or None
    sampled = should_sample(session_id=session_id, utterance_id=utterance_id)
    t0 = time.time()
    log.info(
        "Transcribe request received session_id=%s utterance_id=%s mode=%s bytes=%s sample_rate=%s decoder=%s language=%s",
        session_id or "-",
        utterance_id or "-",
        mode,
        len(pcm),
        x_sample_rate,
        x_decoder,
        x_language,
    )
    emit_eval_event(
        log,
        "transcribe_request_received",
        session_id=session_id,
        utterance_id=utterance_id,
        sampled=sampled,
        mode=mode,
        request_bytes=len(pcm),
        x_sample_rate=x_sample_rate,
        **lid_eval_fields(include_resolution=False),
    )
    
    INFLIGHT_REQUESTS.inc()
    try:
        sem_wait_t0 = time.time()
        async with sem:
            queue_wait_ms = int((time.time() - sem_wait_t0) * 1000)
            log.info(
                "Transcribe worker slot acquired session_id=%s utterance_id=%s mode=%s queue_wait_ms=%s",
                session_id or "-",
                utterance_id or "-",
                mode,
                queue_wait_ms,
            )
            try:
                sample_rate = int(x_sample_rate)
            except Exception:
                REQS.labels(mode=mode, status="err").inc()
                LAT.observe(time.time() - t0)
                log.warning(
                    "Invalid sample rate session_id=%s utterance_id=%s value=%s",
                    session_id or "-",
                    utterance_id or "-",
                    x_sample_rate,
                )
                emit_eval_event(
                    log,
                    "transcribe_result_error",
                    session_id=session_id,
                    utterance_id=utterance_id,
                    sampled=sampled,
                    mode=mode,
                    reason="invalid_sample_rate_header",
                    latency_ms=int((time.time() - t0) * 1000),
                    **lid_eval_fields(include_resolution=False),
                )
                return PlainTextResponse("error: invalid X-Sample-Rate header", status_code=400)

            try:
                inference_t0 = time.time()
                result = await model.transcribe_with_timeout(
                    pcm16le=pcm,
                    sample_rate=sample_rate,
                    decoder=x_decoder,
                    language=x_language,
                    session_id=session_id,
                    utterance_id=utterance_id,
                    mode=mode,
                )
                INFERENCE_LATENCY.observe(time.time() - inference_t0)

                REQS.labels(mode=mode, status="ok").inc()
                log.info(
                    "Transcribe request completed session_id=%s utterance_id=%s mode=%s latency_ms=%s text_chars=%s language=%s language_source=%s",
                    session_id or "-",
                    utterance_id or "-",
                    mode,
                    int((time.time() - t0) * 1000),
                    len(result.text),
                    result.language,
                    result.language_source,
                )
                emit_eval_event(
                    log,
                    "transcribe_result_ok",
                    session_id=session_id,
                    utterance_id=utterance_id,
                    sampled=sampled,
                    mode=mode,
                    sample_rate=sample_rate,
                    latency_ms=int((time.time() - t0) * 1000),
                    resolved_language=result.language,
                    language_source=result.language_source,
                    **lid_eval_fields(include_resolution=True),
                    **text_metadata(result.text),
                )
                return {
                    "text": result.text,
                    "language": result.language,
                    "language_source": result.language_source,
                }
            except (UnsupportedLanguageError, ValueError) as exc:
                REQS.labels(mode=mode, status="err").inc()
                log.warning(
                    "Transcribe validation error session_id=%s utterance_id=%s mode=%s error=%s",
                    session_id or "-",
                    utterance_id or "-",
                    mode,
                    exc,
                )
                emit_eval_event(
                    log,
                    "transcribe_result_error",
                    session_id=session_id,
                    utterance_id=utterance_id,
                    sampled=sampled,
                    mode=mode,
                    reason="validation_error",
                    error=str(exc),
                    latency_ms=int((time.time() - t0) * 1000),
                    **lid_eval_fields(include_resolution=True),
                )
                return PlainTextResponse(f"error: {exc}", status_code=400)
            except InferenceTimeoutError as exc:
                ERRORS.labels(type="Timeout").inc()
                log.warning("transcribe timeout: %s", exc)
                REQS.labels(mode=mode, status="timeout").inc()
                FALLBACKS.labels(reason="timeout").inc()
                log.warning(
                    "Returning timeout fallback session_id=%s utterance_id=%s mode=%s",
                    session_id or "-",
                    utterance_id or "-",
                    mode,
                )
                emit_eval_event(
                    log,
                    "transcribe_result_timeout",
                    session_id=session_id,
                    utterance_id=utterance_id,
                    sampled=sampled,
                    mode=mode,
                    reason="timeout",
                    latency_ms=int((time.time() - t0) * 1000),
                    **lid_eval_fields(include_resolution=True),
                )
                return {
                    "text": "worker-fallback",
                    "language": x_language if x_language != "auto" else ASR_DEFAULT_LANGUAGE,
                    "language_source": "fallback_timeout",
                }
            except ModelNotReadyError as exc:
                ERRORS.labels(type="ModelNotReady").inc()
                log.warning("model not ready: %s", exc)
                REQS.labels(mode=mode, status="not_ready").inc()
                FALLBACKS.labels(reason="not_ready").inc()
                log.warning(
                    "Returning not-ready fallback session_id=%s utterance_id=%s mode=%s",
                    session_id or "-",
                    utterance_id or "-",
                    mode,
                )
                emit_eval_event(
                    log,
                    "transcribe_result_not_ready",
                    session_id=session_id,
                    utterance_id=utterance_id,
                    sampled=sampled,
                    mode=mode,
                    reason="not_ready",
                    latency_ms=int((time.time() - t0) * 1000),
                    **lid_eval_fields(include_resolution=True),
                )
                return {
                    "text": "worker-fallback",
                    "language": x_language if x_language != "auto" else ASR_DEFAULT_LANGUAGE,
                    "language_source": "fallback_not_ready",
                }
            except InferenceError as exc:
                ERRORS.labels(type="InferenceError").inc()
                log.exception("transcribe inference error: %s", exc)
                REQS.labels(mode=mode, status="fallback").inc()
                FALLBACKS.labels(reason="inference_error").inc()
                log.warning(
                    "Returning inference fallback session_id=%s utterance_id=%s mode=%s",
                    session_id or "-",
                    utterance_id or "-",
                    mode,
                )
                emit_eval_event(
                    log,
                    "transcribe_result_fallback",
                    session_id=session_id,
                    utterance_id=utterance_id,
                    sampled=sampled,
                    mode=mode,
                    reason="inference_error",
                    error=str(exc),
                    latency_ms=int((time.time() - t0) * 1000),
                    **lid_eval_fields(include_resolution=True),
                )
                return {
                    "text": "worker-fallback",
                    "language": x_language if x_language != "auto" else ASR_DEFAULT_LANGUAGE,
                    "language_source": "fallback_inference_error",
                }
            except Exception as exc:
                ERRORS.labels(type="Unknown").inc()
                log.exception("transcribe unexpected error: %s", exc)
                REQS.labels(mode=mode, status="fallback").inc()
                FALLBACKS.labels(reason="unexpected").inc()
                log.warning(
                    "Returning unexpected fallback session_id=%s utterance_id=%s mode=%s",
                    session_id or "-",
                    utterance_id or "-",
                    mode,
                )
                emit_eval_event(
                    log,
                    "transcribe_result_fallback",
                    session_id=session_id,
                    utterance_id=utterance_id,
                    sampled=sampled,
                    mode=mode,
                    reason="unexpected",
                    error=str(exc),
                    latency_ms=int((time.time() - t0) * 1000),
                    **lid_eval_fields(include_resolution=True),
                )
                return {
                    "text": "worker-fallback",
                    "language": x_language if x_language != "auto" else ASR_DEFAULT_LANGUAGE,
                    "language_source": "fallback_unexpected",
                }
            finally:
                LAT.observe(time.time() - t0)
    finally:
        INFLIGHT_REQUESTS.dec()
