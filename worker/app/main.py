import asyncio
from dataclasses import replace
from functools import lru_cache
import logging
import time

from fastapi import FastAPI, Header, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import (
    ASR_BACKEND,
    ASR_CONTEXT_BIASING_BEAM_THRESHOLD,
    ASR_CONTEXT_BIASING_CONTEXT_SCORE,
    ASR_CONTEXT_BIASING_CTC_ALI_TOKEN_WEIGHT,
    ASR_CONTEXT_BIASING_DEVICE,
    ASR_CONTEXT_BIASING_METHOD,
    ASR_CONTEXT_BIASING_MODE,
    ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS,
    ASR_CONTEXT_BIASING_NEMO_SOURCE,
    ASR_CONTEXT_BIASING_PHRASES_DIR,
    ASR_CONTEXT_BIASING_SHADOW_SAMPLE_RATE,
    ASR_CONTEXT_BIASING_TIMEOUT_MS,
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
    TRITON_MODEL_NAME,
    TRITON_MODEL_VERSION,
    TRITON_URL,
    WORKER_MAX_JOBS,
)
from .context_biasing import (
    ContextBiasingConfig,
    ContextBiasingDecision,
    ContextBiasingError,
    ContextBiasingNotReadyError,
    ContextBiasingTimeoutError,
    NeMoContextBiasingRuntime,
    PhraseLexicon,
    should_return_active_biasing_transcript,
)
from .logging_setup import setup_logging
from .metrics import (
    CONTEXT_BIASING_FALLBACKS,
    CONTEXT_BIASING_LATENCY,
    CONTEXT_BIASING_REQUESTS,
    FALLBACKS,
    LAT,
    MODEL_INIT,
    REQS,
    ERRORS,
    INFLIGHT_REQUESTS,
    INFERENCE_LATENCY,
)
from .eval_logging import emit_eval_event, should_sample, text_metadata
from .model import (
    InferenceError,
    InferenceTimeoutError,
    ModelNotReadyError,
    ONNXIndicASRWorker,
    UnsupportedLanguageError,
)
from .triton import TritonIndicASRWorker

setup_logging()
log = logging.getLogger("worker")

app = FastAPI()


def build_worker_model():
    common_kwargs = dict(
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

    if ASR_BACKEND == "triton":
        return TritonIndicASRWorker(
            triton_url=TRITON_URL,
            triton_model_name=TRITON_MODEL_NAME,
            triton_model_version=TRITON_MODEL_VERSION,
            **common_kwargs,
        )

    return ONNXIndicASRWorker(
        model_name=ASR_MODEL_NAME,
        **common_kwargs,
    )


def build_context_biasing_runtime() -> NeMoContextBiasingRuntime:
    return NeMoContextBiasingRuntime(
        ContextBiasingConfig(
            mode=ASR_CONTEXT_BIASING_MODE,
            method=ASR_CONTEXT_BIASING_METHOD,
            nemo_source=ASR_CONTEXT_BIASING_NEMO_SOURCE,
            nemo_model_class=ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS,
            phrases_dir=ASR_CONTEXT_BIASING_PHRASES_DIR,
            timeout_ms=ASR_CONTEXT_BIASING_TIMEOUT_MS,
            device=ASR_CONTEXT_BIASING_DEVICE,
            shadow_sample_rate=ASR_CONTEXT_BIASING_SHADOW_SAMPLE_RATE,
            beam_threshold=ASR_CONTEXT_BIASING_BEAM_THRESHOLD,
            context_score=ASR_CONTEXT_BIASING_CONTEXT_SCORE,
            ctc_ali_token_weight=ASR_CONTEXT_BIASING_CTC_ALI_TOKEN_WEIGHT,
        )
    )


model = build_worker_model()
context_biasing = build_context_biasing_runtime()
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


def prefixed_text_metadata(prefix: str, text: str) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in text_metadata(text).items()}


@lru_cache(maxsize=16)
def load_phrase_lexicon(phrase_file: str, language: str) -> PhraseLexicon:
    return PhraseLexicon.from_file(phrase_file, language=language)


async def maybe_apply_context_biasing(
    *,
    baseline_result,
    pcm: bytes,
    sample_rate: int,
    requested_language: str,
    session_id: str | None,
    utterance_id: str | None,
    mode: str,
    sampled: bool,
) -> tuple:
    decision: ContextBiasingDecision = context_biasing.decide(
        requested_language=requested_language,
        session_id=session_id,
        utterance_id=utterance_id,
    )
    if decision.mode == "disabled":
        return baseline_result, decision, None, None

    if not decision.eligible:
        status = "fallback" if decision.reason == "not_ready" and decision.mode == "active" else "skipped"
        CONTEXT_BIASING_REQUESTS.labels(mode=decision.mode, status=status).inc()
        if status == "fallback":
            CONTEXT_BIASING_FALLBACKS.labels(reason="not_ready").inc()
        emit_eval_event(
            log,
            "transcribe_context_biasing_skipped",
            session_id=session_id,
            utterance_id=utterance_id,
            sampled=sampled,
            mode=mode,
            bias_mode=decision.mode,
            bias_method=context_biasing.method,
            reason=decision.reason,
            requested_language=requested_language,
            resolved_language=baseline_result.language,
            phrase_file=decision.phrase_file,
        )
        return baseline_result, decision, None, decision.reason if status == "fallback" else None

    if baseline_result.language != decision.language:
        CONTEXT_BIASING_REQUESTS.labels(mode=decision.mode, status="skipped").inc()
        emit_eval_event(
            log,
            "transcribe_context_biasing_skipped",
            session_id=session_id,
            utterance_id=utterance_id,
            sampled=sampled,
            mode=mode,
            bias_mode=decision.mode,
            bias_method=context_biasing.method,
            reason="resolved_language_mismatch",
            requested_language=requested_language,
            resolved_language=baseline_result.language,
            phrase_file=decision.phrase_file,
        )
        return baseline_result, decision, None, None

    attempt_t0 = time.time()
    try:
        biased = await context_biasing.transcribe_with_timeout(
            pcm16le=pcm,
            sample_rate=sample_rate,
            language=decision.language,
            phrase_file=decision.phrase_file or "",
            session_id=session_id,
            utterance_id=utterance_id,
            mode=mode,
        )
        elapsed = max(time.time() - attempt_t0, 0.0)
        CONTEXT_BIASING_LATENCY.observe(elapsed)
        status = "ok" if biased.text.strip() else "empty_candidate"
        CONTEXT_BIASING_REQUESTS.labels(mode=decision.mode, status=status).inc()
        returned = baseline_result
        returned_source = "baseline"
        selection_reason = "shadow_mode" if decision.mode == "shadow" else "baseline_only"
        baseline_phrase_hits = 0
        biased_phrase_hits = 0
        returned_phrase_hits = 0
        if decision.mode == "active":
            lexicon = None
            if decision.phrase_file:
                try:
                    lexicon = load_phrase_lexicon(decision.phrase_file, decision.language)
                except Exception as exc:
                    log.warning(
                        "Failed to load context-biasing phrase lexicon phrase_file=%s language=%s error=%s",
                        decision.phrase_file,
                        decision.language or "-",
                        exc,
                    )
            should_return_biased, selection_reason, baseline_phrase_hits, biased_phrase_hits = (
                should_return_active_biasing_transcript(
                    baseline_text=baseline_result.text,
                    biased_text=biased.text,
                    lexicon=lexicon,
                )
            )
            if should_return_biased:
                returned = replace(baseline_result, text=biased.text)
                returned_source = "biased"
                returned_phrase_hits = biased_phrase_hits
            else:
                returned_phrase_hits = baseline_phrase_hits
                if biased.text.strip():
                    CONTEXT_BIASING_FALLBACKS.labels(reason=selection_reason).inc()
        emit_eval_event(
            log,
            "transcribe_context_biasing_result",
            session_id=session_id,
            utterance_id=utterance_id,
            sampled=sampled,
            mode=mode,
            bias_mode=decision.mode,
            bias_method=context_biasing.method,
            status=status,
            requested_language=requested_language,
            resolved_language=baseline_result.language,
            phrase_file=biased.phrase_file,
            bias_latency_ms=biased.latency_ms,
            returned_source=returned_source,
            selection_reason=selection_reason,
            baseline_phrase_hits=baseline_phrase_hits,
            biased_phrase_hits=biased_phrase_hits,
            returned_phrase_hits=returned_phrase_hits,
            **prefixed_text_metadata("baseline", baseline_result.text),
            **prefixed_text_metadata("biased", biased.text),
            **prefixed_text_metadata("returned", returned.text),
        )
        if decision.mode == "active" and not biased.text.strip():
            CONTEXT_BIASING_FALLBACKS.labels(reason="empty_candidate").inc()
        return returned, decision, biased.text, None
    except ContextBiasingTimeoutError as exc:
        CONTEXT_BIASING_REQUESTS.labels(mode=decision.mode, status="fallback").inc()
        CONTEXT_BIASING_FALLBACKS.labels(reason="timeout").inc()
        emit_eval_event(
            log,
            "transcribe_context_biasing_result",
            session_id=session_id,
            utterance_id=utterance_id,
            sampled=sampled,
            mode=mode,
            bias_mode=decision.mode,
            bias_method=context_biasing.method,
            status="fallback",
            reason="timeout",
            error=str(exc),
            requested_language=requested_language,
            resolved_language=baseline_result.language,
            phrase_file=decision.phrase_file,
            **prefixed_text_metadata("baseline", baseline_result.text),
        )
        return baseline_result, decision, None, "timeout"
    except (ContextBiasingNotReadyError, ContextBiasingError) as exc:
        CONTEXT_BIASING_REQUESTS.labels(mode=decision.mode, status="fallback").inc()
        CONTEXT_BIASING_FALLBACKS.labels(reason="error").inc()
        emit_eval_event(
            log,
            "transcribe_context_biasing_result",
            session_id=session_id,
            utterance_id=utterance_id,
            sampled=sampled,
            mode=mode,
            bias_mode=decision.mode,
            bias_method=context_biasing.method,
            status="fallback",
            reason="error",
            error=str(exc),
            requested_language=requested_language,
            resolved_language=baseline_result.language,
            phrase_file=decision.phrase_file,
            **prefixed_text_metadata("baseline", baseline_result.text),
        )
        return baseline_result, decision, None, "error"


@app.on_event("startup")
async def startup_event():
    log.info(
        "Worker startup backend=%s model=%s triton_model=%s triton_url=%s decoder=%s default_language=%s lid_enabled=%s lid_primary_provider=%s lid_primary_source=%s lid_fallback_provider=%s lid_fallback_source=%s lid_confidence_threshold=%.2f timeout_ms=%s max_jobs=%s context_biasing_mode=%s context_biasing_method=%s context_biasing_phrases_dir=%s",
        ASR_BACKEND,
        ASR_MODEL_NAME or "-",
        TRITON_MODEL_NAME,
        TRITON_URL,
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
        ASR_CONTEXT_BIASING_MODE,
        ASR_CONTEXT_BIASING_METHOD,
        ASR_CONTEXT_BIASING_PHRASES_DIR or "-",
    )
    try:
        await asyncio.to_thread(model.load)
        MODEL_INIT.labels(status="ok").inc()
        log.info("Worker model ready")
    except Exception as exc:
        MODEL_INIT.labels(status="err").inc()
        log.exception("Worker model failed to initialize: %s", exc)

    if context_biasing.mode != "disabled":
        await asyncio.to_thread(context_biasing.load)
        if context_biasing.ready:
            log.info(
                "Context biasing ready mode=%s method=%s source=%s model_class=%s phrases_dir=%s",
                context_biasing.mode,
                context_biasing.method,
                ASR_CONTEXT_BIASING_NEMO_SOURCE or "-",
                ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS or "-",
                ASR_CONTEXT_BIASING_PHRASES_DIR or "-",
            )
        else:
            log.warning(
                "Context biasing unavailable mode=%s reason=%s",
                context_biasing.mode,
                context_biasing.init_error or "not-ready",
            )


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
                result, bias_decision, _biased_text, bias_fallback_reason = await maybe_apply_context_biasing(
                    baseline_result=result,
                    pcm=pcm,
                    sample_rate=sample_rate,
                    requested_language=x_language,
                    session_id=session_id,
                    utterance_id=utterance_id,
                    mode=mode,
                    sampled=sampled,
                )

                REQS.labels(mode=mode, status="ok").inc()
                log.info(
                    "Transcribe request completed session_id=%s utterance_id=%s mode=%s latency_ms=%s text_chars=%s language=%s language_source=%s context_biasing_mode=%s context_biasing_reason=%s context_biasing_fallback_reason=%s",
                    session_id or "-",
                    utterance_id or "-",
                    mode,
                    int((time.time() - t0) * 1000),
                    len(result.text),
                    result.language,
                    result.language_source,
                    bias_decision.mode,
                    bias_decision.reason,
                    bias_fallback_reason or "-",
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
                    context_biasing_mode=bias_decision.mode,
                    context_biasing_reason=bias_decision.reason,
                    context_biasing_phrase_file=bias_decision.phrase_file,
                    context_biasing_fallback_reason=bias_fallback_reason,
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
