import asyncio
import base64
import binascii
import io
import json
import logging
import time
import uuid
import wave
from dataclasses import dataclass
from typing import Any, Optional

import orjson
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from .circuit_breaker import CircuitBreaker
from .config import (
    CIRCUIT_BREAKER_FAILS,
    CIRCUIT_BREAKER_RESET_MS,
    CONN_BURST,
    GATEWAY_DISABLE_RATE_LIMITING,
    GATEWAY_MAX_INFLIGHT_WORKER,
    MAX_BYTES_PER_SEC,
    MAX_CONNS_PER_KEY,
    NEW_CONN_PER_MIN,
    REDIS_URL,
    WORKER_TIMEOUT_MS,
    WORKER_URL,
    VAD_END_SILENCE_MS,
    VAD_KEEP_SILENCE_MS,
    VAD_MAX_UTT_MS,
    WS_DISABLE_AUDIO_RATE_LIMIT,
    WS_API_KEYS,
)
from .eval_logging import emit_eval_event, hash_value, should_sample, text_metadata
from .fallback_limiter import FallbackLimiter
from .logging_setup import setup_logging
from .metrics import (
    AUDIO_BYTES_RECEIVED,
    AUDIO_FRAMES_RECEIVED,
    E2E_LATENCY,
    GATEWAY_LATENCY,
    UTTERANCES,
    VAD_FRAMES,
    WS_CONNECTIONS,
    WS_DISCONNECTS,
    WS_REJECTS,
)
from .redis_limiter import RedisLimiter
from .vad import VADSegmenter
from .worker_client import WorkerClient

DEFAULT_MODEL = "credresolve:v1"
DEFAULT_MODE = "transcribe"
DEFAULT_INPUT_AUDIO_CODEC = "pcm_s16le"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_VAD_MODE = 2
HIGH_SENSITIVITY_VAD_MODE = 3
FRAME_MS = 20
INTERNAL_DECODER = "rnnt"
ALLOWED_SAMPLE_RATES = {8000, 16000}
ALLOWED_AUDIO_CODECS = {"wav", "pcm_s16le", "pcm_l16", "pcm_raw"}

setup_logging()
log = logging.getLogger("gateway")

app = FastAPI()
worker = WorkerClient(WORKER_URL, WORKER_TIMEOUT_MS)
worker_sem = asyncio.Semaphore(GATEWAY_MAX_INFLIGHT_WORKER)
breaker = CircuitBreaker(CIRCUIT_BREAKER_FAILS, CIRCUIT_BREAKER_RESET_MS)

redis: Optional[Redis] = None
redis_limiter: Optional[RedisLimiter] = None
fallback_limiter = FallbackLimiter(MAX_CONNS_PER_KEY, NEW_CONN_PER_MIN, CONN_BURST)


def connection_rate_limit_enabled() -> bool:
    return not GATEWAY_DISABLE_RATE_LIMITING


def audio_rate_limit_enabled() -> bool:
    return not GATEWAY_DISABLE_RATE_LIMITING and not WS_DISABLE_AUDIO_RATE_LIMIT


@dataclass(frozen=True)
class SessionConfig:
    request_id: str
    language_code: str
    model: str
    mode: str
    sample_rate: int
    high_vad_sensitivity: bool
    vad_signals: bool
    flush_signal: bool
    input_audio_codec: str


class HandshakeValidationError(ValueError):
    pass


class BadMessageError(ValueError):
    pass


@app.on_event("startup")
async def startup():
    global redis, redis_limiter
    log.info(
        "Gateway startup redis_url=%s worker_url=%s worker_timeout_ms=%s max_inflight_worker=%s connection_rate_limit_enabled=%s audio_rate_limit_enabled=%s",
        REDIS_URL,
        WORKER_URL,
        WORKER_TIMEOUT_MS,
        GATEWAY_MAX_INFLIGHT_WORKER,
        connection_rate_limit_enabled(),
        audio_rate_limit_enabled(),
    )
    if not connection_rate_limit_enabled():
        redis = None
        redis_limiter = None
        log.info("All gateway rate limiting disabled")
        return
    try:
        redis = Redis.from_url(REDIS_URL, decode_responses=False)
        await redis.ping()
        redis_limiter = RedisLimiter(redis, MAX_CONNS_PER_KEY, NEW_CONN_PER_MIN, CONN_BURST)
        log.info("Redis limiter enabled")
    except Exception as exc:
        redis = None
        redis_limiter = None
        log.warning("Redis unavailable; using in-memory limiter: %s", exc)


@app.on_event("shutdown")
async def shutdown():
    log.info("Gateway shutdown initiated")
    if redis:
        await redis.close()
    await worker.close()
    log.info("Gateway shutdown complete")


@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")


@app.get("/", include_in_schema=False)
async def root():
    return PlainTextResponse("ok")


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


def jdump(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")


def parse_sec_websocket_protocol_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    if len(items) >= 2 and items[0].lower() == "token" and items[1]:
        return items[1]
    return None


def extract_ws_auth(ws: WebSocket) -> tuple[str, Optional[str]]:
    api_key = (ws.headers.get("api-subscription-key") or "").strip()
    if api_key:
        return api_key, None

    protocol_token = parse_sec_websocket_protocol_token(ws.headers.get("sec-websocket-protocol"))
    if protocol_token:
        return protocol_token, "token"
    return "", None


def is_valid_ws_api_key(token: Optional[str]) -> bool:
    return bool(token) and token in WS_API_KEYS


def parse_bool_query(name: str, value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise HandshakeValidationError(f"invalid boolean value for {name}: {value}")


def parse_sample_rate(value: Optional[str]) -> int:
    raw = (value or str(DEFAULT_SAMPLE_RATE)).strip()
    try:
        sample_rate = int(raw)
    except ValueError as exc:
        raise HandshakeValidationError(f"invalid sample_rate: {raw}") from exc
    if sample_rate not in ALLOWED_SAMPLE_RATES:
        raise HandshakeValidationError("sample_rate must be 8000 or 16000")
    return sample_rate


def parse_input_audio_codec(value: Optional[str]) -> str:
    codec = (value or DEFAULT_INPUT_AUDIO_CODEC).strip().lower()
    if codec not in ALLOWED_AUDIO_CODECS:
        raise HandshakeValidationError(
            "input_audio_codec must be one of wav, pcm_s16le, pcm_l16, pcm_raw"
        )
    return codec


def parse_session_config(ws: WebSocket, request_id: str) -> SessionConfig:
    params = ws.query_params
    language_code = (params.get("language-code") or "").strip()
    if not language_code:
        raise HandshakeValidationError("missing required query param: language-code")

    return SessionConfig(
        request_id=request_id,
        language_code=language_code,
        model=(params.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        mode=(params.get("mode") or DEFAULT_MODE).strip() or DEFAULT_MODE,
        sample_rate=parse_sample_rate(params.get("sample_rate")),
        high_vad_sensitivity=parse_bool_query(
            "high_vad_sensitivity", params.get("high_vad_sensitivity"), False
        ),
        vad_signals=parse_bool_query("vad_signals", params.get("vad_signals"), False),
        flush_signal=parse_bool_query("flush_signal", params.get("flush_signal"), False),
        input_audio_codec=parse_input_audio_codec(params.get("input_audio_codec")),
    )


def swap_pcm_byte_order(raw_audio: bytes) -> bytes:
    if len(raw_audio) % 2 != 0:
        raise BadMessageError("pcm payload must contain an even number of bytes")
    swapped = bytearray(len(raw_audio))
    for index in range(0, len(raw_audio), 2):
        swapped[index] = raw_audio[index + 1]
        swapped[index + 1] = raw_audio[index]
    return bytes(swapped)


def decode_wav_payload(raw_audio: bytes, expected_sample_rate: int) -> bytes:
    try:
        with wave.open(io.BytesIO(raw_audio), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise BadMessageError("wav audio must be mono")
            if wav_file.getsampwidth() != 2:
                raise BadMessageError("wav audio must be 16-bit PCM")
            actual_rate = wav_file.getframerate()
            if actual_rate != expected_sample_rate:
                raise BadMessageError(
                    f"wav sample rate {actual_rate} does not match negotiated sample_rate {expected_sample_rate}"
                )
            return wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise BadMessageError(f"invalid wav audio payload: {exc}") from exc


def normalize_audio_payload(raw_audio: bytes, codec: str, sample_rate: int) -> bytes:
    if not raw_audio:
        raise BadMessageError("audio.data decoded to an empty payload")
    if codec == "wav":
        return decode_wav_payload(raw_audio, sample_rate)
    if codec in {"pcm_s16le", "pcm_raw"}:
        if len(raw_audio) % 2 != 0:
            raise BadMessageError("pcm payload must contain an even number of bytes")
        return raw_audio
    if codec == "pcm_l16":
        return swap_pcm_byte_order(raw_audio)
    raise BadMessageError(f"unsupported input_audio_codec: {codec}")


def decode_audio_message(payload: dict[str, Any], session: SessionConfig) -> bytes:
    audio = payload.get("audio")
    if not isinstance(audio, dict):
        raise BadMessageError("missing audio object")

    data = audio.get("data")
    if not isinstance(data, str) or not data.strip():
        raise BadMessageError("audio.data must be a non-empty base64 string")

    message_sample_rate = audio.get("sample_rate")
    if message_sample_rate is not None:
        try:
            if int(str(message_sample_rate).strip()) != session.sample_rate:
                raise BadMessageError("audio.sample_rate does not match negotiated sample_rate")
        except ValueError as exc:
            raise BadMessageError("audio.sample_rate must be an integer") from exc

    message_encoding = audio.get("encoding")
    if message_encoding is not None:
        if str(message_encoding).strip().lower() != session.input_audio_codec:
            raise BadMessageError("audio.encoding does not match negotiated input_audio_codec")

    try:
        raw_audio = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadMessageError("audio.data must be valid base64") from exc

    return normalize_audio_payload(raw_audio, session.input_audio_codec, session.sample_rate)


async def send_ws_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_text(jdump({"type": "error", "code": code, "message": message}))


async def send_ws_error_and_close(ws: WebSocket, code: str, message: str, close_code: int) -> None:
    await send_ws_error(ws, code, message)
    await ws.close(code=close_code)


class ByteRateGuard:
    def __init__(self, max_bps: int):
        self.rate = float(max_bps)
        self.capacity = float(max_bps) * 2.0
        self.tokens = self.capacity
        self.last_check = time.monotonic()

    def add(self, n: int) -> bool:
        now = time.monotonic()
        elapsed = max(0.0, now - self.last_check)
        self.last_check = now

        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if n > self.tokens:
            return False
        self.tokens -= n
        return True


@app.websocket("/ws/stt")
async def ws_stt(ws: WebSocket):
    api_key, accepted_subprotocol = extract_ws_auth(ws)
    api_key_hash = hash_value(api_key) if api_key else ""

    await ws.accept(subprotocol=accepted_subprotocol)

    session_id = str(uuid.uuid4())
    sampled = should_sample(session_id=session_id)
    session_started_ms = int(time.time() * 1000)
    close_reason = "unknown"
    total_audio_bytes = 0
    utterance_count = 0
    admitted = False
    session: Optional[SessionConfig] = None
    request_id = session_id
    utt_start_time = 0.0

    WS_CONNECTIONS.inc()
    guard = ByteRateGuard(MAX_BYTES_PER_SEC)
    log.info("WS accepted session_id=%s client=%s", session_id, ws.client)

    try:
        if not api_key or not is_valid_ws_api_key(api_key):
            WS_REJECTS.labels(reason="AUTH_FAILED").inc()
            close_reason = "auth_failed"
            log.warning("WS rejected session_id=%s reason=AUTH_FAILED", session_id)
            emit_eval_event(
                log,
                "ws_session_rejected",
                session_id=session_id,
                sampled=sampled,
                reason="AUTH_FAILED",
                code="AUTH_FAILED",
                api_key_hash=api_key_hash,
            )
            await send_ws_error_and_close(
                ws,
                "AUTH_FAILED",
                "missing or invalid Api-Subscription-Key",
                1008,
            )
            return

        try:
            session = parse_session_config(ws, request_id=session_id)
        except HandshakeValidationError as exc:
            WS_REJECTS.labels(reason="VALIDATION_ERROR").inc()
            close_reason = "validation_error"
            log.warning("WS rejected session_id=%s reason=VALIDATION_ERROR error=%s", session_id, exc)
            emit_eval_event(
                log,
                "ws_session_rejected",
                session_id=session_id,
                sampled=sampled,
                reason="VALIDATION_ERROR",
                code="VALIDATION_ERROR",
                api_key_hash=api_key_hash,
                error=str(exc),
            )
            await send_ws_error_and_close(ws, "VALIDATION_ERROR", str(exc), 1008)
            return

        request_id = session.request_id

        if connection_rate_limit_enabled():
            if redis_limiter:
                res = await redis_limiter.admit(api_key)
                if not res.ok:
                    WS_REJECTS.labels(reason=res.reason).inc()
                    close_reason = "admission_rejected"
                    emit_eval_event(
                        log,
                        "ws_session_rejected",
                        session_id=session_id,
                        sampled=sampled,
                        reason=res.reason,
                        code=res.reason,
                        api_key_hash=api_key_hash,
                    )
                    await send_ws_error_and_close(ws, res.reason, res.reason, 1008)
                    return
                admitted = True
            else:
                ok, reason = fallback_limiter.admit(api_key)
                if not ok:
                    WS_REJECTS.labels(reason=reason).inc()
                    close_reason = "admission_rejected"
                    emit_eval_event(
                        log,
                        "ws_session_rejected",
                        session_id=session_id,
                        sampled=sampled,
                        reason=reason,
                        code=reason,
                        api_key_hash=api_key_hash,
                    )
                    await send_ws_error_and_close(ws, reason, reason, 1008)
                    return
                admitted = True

        vad = VADSegmenter(
            sample_rate=session.sample_rate,
            frame_ms=FRAME_MS,
            mode=HIGH_SENSITIVITY_VAD_MODE if session.high_vad_sensitivity else DEFAULT_VAD_MODE,
            end_silence_ms=VAD_END_SILENCE_MS,
            keep_silence_ms=VAD_KEEP_SILENCE_MS,
            max_utt_ms=VAD_MAX_UTT_MS,
        )
        frame_buffer = bytearray()
        segment_audio_buffer = bytearray()

        log.info(
            "WS session started session_id=%s request_id=%s api_key_hash=%s language=%s model=%s mode=%s sample_rate=%s codec=%s vad_signals=%s audio_rate_limit_enabled=%s",
            session_id,
            session.request_id,
            api_key_hash or "-",
            session.language_code,
            session.model,
            session.mode,
            session.sample_rate,
            session.input_audio_codec,
            session.vad_signals,
            audio_rate_limit_enabled(),
        )
        emit_eval_event(
            log,
            "ws_session_started",
            session_id=session_id,
            sampled=sampled,
            api_key_hash=api_key_hash,
            request_id=session.request_id,
            language=session.language_code,
            model=session.model,
            public_mode=session.mode,
            sample_rate=session.sample_rate,
            input_audio_codec=session.input_audio_codec,
            high_vad_sensitivity=session.high_vad_sensitivity,
            vad_signals=session.vad_signals,
            flush_signal=session.flush_signal,
        )

        async def finalize_audio(audio_bytes: bytes, trigger: str) -> bool:
            nonlocal utterance_count, utt_start_time, close_reason

            if not audio_bytes:
                segment_audio_buffer.clear()
                return True

            UTTERANCES.inc()
            utterance_count += 1
            utterance_id = f"utt-{utterance_count:04d}"
            emit_eval_event(
                log,
                "vad_segment_finalized",
                session_id=session_id,
                utterance_id=utterance_id,
                sampled=sampled,
                audio_bytes=len(audio_bytes),
                trigger=trigger,
            )

            if not breaker.allow():
                close_reason = "overloaded"
                log.warning(
                    "WS rejected session_id=%s utterance_id=%s reason=OVERLOADED",
                    session_id,
                    utterance_id,
                )
                emit_eval_event(
                    log,
                    "ws_session_rejected",
                    session_id=session_id,
                    utterance_id=utterance_id,
                    sampled=sampled,
                    reason="OVERLOADED",
                    code="OVERLOADED",
                    api_key_hash=api_key_hash,
                )
                await send_ws_error_and_close(ws, "OVERLOADED", "gateway overloaded", 1013)
                return False

            out = None
            async with worker_sem:
                try:
                    final_t0 = time.time()
                    log.info(
                        "Dispatching final transcription session_id=%s utterance_id=%s bytes=%s trigger=%s language=%s",
                        session_id,
                        utterance_id,
                        len(audio_bytes),
                        trigger,
                        session.language_code,
                    )
                    out = await worker.transcribe(
                        audio_bytes,
                        session.sample_rate,
                        INTERNAL_DECODER,
                        session.language_code,
                        mode="final",
                        session_id=session_id,
                        utterance_id=utterance_id,
                        sampled=sampled,
                    )
                    GATEWAY_LATENCY.observe(max(0, time.time() - final_t0))
                    processing_latency = max(0.0, time.time() - final_t0)
                    processing_latency_ms = int(processing_latency * 1000)
                    breaker.on_success()
                except Exception:
                    breaker.on_failure()
                    log.exception(
                        "Final transcription failed session_id=%s utterance_id=%s",
                        session_id,
                        utterance_id,
                    )
                    await send_ws_error(ws, "WORKER_ERROR", "worker transcription failed")
                    segment_audio_buffer.clear()
                    return True

            if out is None:
                segment_audio_buffer.clear()
                return True

            audio_duration = len(audio_bytes) / float(session.sample_rate * 2)
            resolved_language = out.language or (
                session.language_code if session.language_code != "auto" else None
            )
            resolved_language_source = out.language_source or (
                "client" if session.language_code != "auto" else None
            )
            await ws.send_text(
                jdump(
                    {
                        "type": "data",
                        "data": {
                            "request_id": session.request_id,
                            "transcript": out.text,
                            "language_code": resolved_language,
                            "language_source": resolved_language_source,
                            "metrics": {
                                "audio_duration": round(audio_duration, 4),
                                "processing_latency": round(processing_latency, 4),
                            },
                        },
                    }
                )
            )
            log.info(
                "Data sent session_id=%s utterance_id=%s latency_ms=%s text_chars=%s language=%s language_source=%s trigger=%s",
                session_id,
                utterance_id,
                processing_latency_ms,
                len(out.text),
                out.language or "-",
                out.language_source or "-",
                trigger,
            )
            emit_eval_event(
                log,
                "final_sent",
                session_id=session_id,
                utterance_id=utterance_id,
                sampled=sampled,
                worker_latency_ms=processing_latency_ms,
                trigger=trigger,
                resolved_language=out.language or None,
                language_source=out.language_source or None,
                **text_metadata(out.text),
            )
            if utt_start_time > 0:
                E2E_LATENCY.observe(time.time() - utt_start_time)
                utt_start_time = 0.0
            segment_audio_buffer.clear()
            return True

        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                close_reason = "client_disconnect"
                break

            if msg.get("bytes"):
                close_reason = "bad_message"
                WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                await send_ws_error_and_close(
                    ws,
                    "BAD_MESSAGE",
                    "binary websocket frames are not supported; send JSON audio messages",
                    1003,
                )
                return

            text = msg.get("text")
            if not text:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                close_reason = "bad_message"
                WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                await send_ws_error_and_close(ws, "BAD_MESSAGE", "message must be valid JSON", 1003)
                return

            if not isinstance(payload, dict):
                close_reason = "bad_message"
                WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                await send_ws_error_and_close(ws, "BAD_MESSAGE", "message must be a JSON object", 1003)
                return

            if payload.get("type") == "flush":
                log.info("WS flush received session_id=%s request_id=%s", session_id, session.request_id)
                if frame_buffer:
                    segment_audio_buffer.extend(frame_buffer)
                    frame_buffer.clear()
                flush_audio = bytes(segment_audio_buffer)
                vad.flush()
                if flush_audio:
                    ok = await finalize_audio(flush_audio, "flush")
                    if not ok:
                        return
                else:
                    segment_audio_buffer.clear()
                continue

            if "audio" not in payload:
                close_reason = "bad_message"
                WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                await send_ws_error_and_close(
                    ws,
                    "BAD_MESSAGE",
                    "unsupported websocket message; expected audio payload or flush",
                    1003,
                )
                return

            try:
                pcm_bytes = decode_audio_message(payload, session)
            except BadMessageError as exc:
                close_reason = "bad_message"
                WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                await send_ws_error_and_close(ws, "BAD_MESSAGE", str(exc), 1003)
                return

            AUDIO_BYTES_RECEIVED.inc(len(pcm_bytes))
            AUDIO_FRAMES_RECEIVED.inc()
            total_audio_bytes += len(pcm_bytes)

            if audio_rate_limit_enabled() and not guard.add(len(pcm_bytes)):
                WS_REJECTS.labels(reason="TOO_MUCH_DATA").inc()
                close_reason = "too_much_data"
                emit_eval_event(
                    log,
                    "ws_session_rejected",
                    session_id=session_id,
                    sampled=sampled,
                    reason="TOO_MUCH_DATA",
                    code="TOO_MUCH_DATA",
                    api_key_hash=api_key_hash,
                )
                await send_ws_error_and_close(ws, "TOO_MUCH_DATA", "audio rate limit exceeded", 1008)
                return

            frame_buffer.extend(pcm_bytes)
            while len(frame_buffer) >= vad.frame_bytes:
                frame = bytes(frame_buffer[: vad.frame_bytes])
                del frame_buffer[: vad.frame_bytes]
                segment_audio_buffer.extend(frame)
                events, audio_ready = vad.push(frame)
                for event in events:
                    if event == "speech_start":
                        utt_start_time = time.time()
                    log.info("VAD event session_id=%s request_id=%s state=%s", session_id, session.request_id, event)
                    VAD_FRAMES.labels(state=event).inc()
                    if session.vad_signals:
                        await ws.send_text(
                            jdump(
                                {
                                    "type": "vad",
                                    "data": {
                                        "request_id": session.request_id,
                                        "event": event,
                                    },
                                }
                            )
                        )

                if audio_ready:
                    trigger = "max_utt" if "max_utt" in events else "speech_end"
                    ok = await finalize_audio(audio_ready, trigger)
                    if not ok:
                        return

    except WebSocketDisconnect:
        close_reason = "client_disconnect"
        log.info("WS client disconnected session_id=%s request_id=%s", session_id, request_id)
    except Exception as exc:
        close_reason = "server_error"
        log.exception("WS error: %s", exc)
        try:
            await send_ws_error(ws, "SERVER_ERROR", "internal websocket server error")
        except Exception:
            pass
    finally:
        WS_CONNECTIONS.dec()
        if close_reason != "unknown":
            WS_DISCONNECTS.labels(reason=close_reason).inc()
        if api_key and admitted:
            try:
                if redis_limiter:
                    await redis_limiter.release(api_key)
                else:
                    fallback_limiter.release(api_key)
            except Exception:
                pass
        emit_eval_event(
            log,
            "ws_session_closed",
            session_id=session_id,
            sampled=sampled,
            api_key_hash=api_key_hash,
            request_id=request_id,
            close_reason=close_reason,
            duration_ms=max(0, int(time.time() * 1000) - session_started_ms),
            total_audio_bytes=total_audio_bytes,
            utterance_count=utterance_count,
        )
        log.info(
            "WS session closed session_id=%s request_id=%s close_reason=%s duration_ms=%s total_audio_bytes=%s utterance_count=%s",
            session_id,
            request_id,
            close_reason,
            max(0, int(time.time() * 1000) - session_started_ms),
            total_audio_bytes,
            utterance_count,
        )
        try:
            await ws.close()
        except Exception:
            pass
