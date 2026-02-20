import asyncio
import json
import time
import uuid
import logging
from typing import Optional

import orjson
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from redis.asyncio import Redis

from .config import (
    REDIS_URL, MAX_CONNS_PER_KEY, NEW_CONN_PER_MIN, CONN_BURST,
    MAX_BYTES_PER_SEC, GATEWAY_MAX_INFLIGHT_WORKER, WORKER_TIMEOUT_MS, WORKER_URL,
    CIRCUIT_BREAKER_FAILS, CIRCUIT_BREAKER_RESET_MS, PARTIAL_DECODE_INTERVAL_MS,
    VAD_END_SILENCE_MS, VAD_MAX_UTT_MS
)
from .logging_setup import setup_logging
from .metrics import (
    WS_CONNECTIONS, WS_REJECTS, WS_DISCONNECTS,
    AUDIO_BYTES_RECEIVED, AUDIO_FRAMES_RECEIVED,
    UTTERANCES, VAD_FRAMES,
    GATEWAY_LATENCY, E2E_LATENCY,
    VAD_FRAMES
)
from .vad import VADSegmenter
from .worker_client import WorkerClient
from .circuit_breaker import CircuitBreaker
from .fallback_limiter import FallbackLimiter
from .redis_limiter import RedisLimiter
from .eval_logging import emit_eval_event, hash_value, should_sample, text_metadata

setup_logging()
log = logging.getLogger("gateway")

app = FastAPI()
worker = WorkerClient(WORKER_URL, WORKER_TIMEOUT_MS)
worker_sem = asyncio.Semaphore(GATEWAY_MAX_INFLIGHT_WORKER)
breaker = CircuitBreaker(CIRCUIT_BREAKER_FAILS, CIRCUIT_BREAKER_RESET_MS)

redis: Optional[Redis] = None
redis_limiter: Optional[RedisLimiter] = None
fallback_limiter = FallbackLimiter(MAX_CONNS_PER_KEY, NEW_CONN_PER_MIN, CONN_BURST)

@app.on_event("startup")
async def startup():
    global redis, redis_limiter
    try:
        redis = Redis.from_url(REDIS_URL, decode_responses=False)
        await redis.ping()
        redis_limiter = RedisLimiter(redis, MAX_CONNS_PER_KEY, NEW_CONN_PER_MIN, CONN_BURST)
        log.info("Redis limiter enabled")
    except Exception as e:
        redis = None
        redis_limiter = None
        log.warning("Redis unavailable; using in-memory limiter: %s", e)

@app.on_event("shutdown")
async def shutdown():
    if redis:
        await redis.close()
    await worker.close()

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

def jdump(obj) -> str:
    return orjson.dumps(obj).decode("utf-8")

class ByteRateGuard:
    def __init__(self, max_bps: int):
        self.rate = float(max_bps)
        # Allow short burst buffering without dropping otherwise valid real-time streams.
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
    await ws.accept()

    session_id = str(uuid.uuid4())
    sampled = should_sample(session_id=session_id)
    session_started_ms = int(time.time() * 1000)
    close_reason = "unknown"
    total_audio_bytes = 0
    utterance_count = 0

    api_key = ""
    api_key_hash = ""
    call_id = ""
    WS_CONNECTIONS.inc()
    guard = ByteRateGuard(MAX_BYTES_PER_SEC)

    try:
        start_msg = await ws.receive_text()
        cfg = json.loads(start_msg)

        if cfg.get("type") != "start":
            WS_REJECTS.labels(reason="BAD_PROTOCOL").inc()
            close_reason = "bad_protocol"
            emit_eval_event(
                log,
                "ws_session_rejected",
                session_id=session_id,
                sampled=sampled,
                reason="BAD_PROTOCOL",
                code="BAD_PROTOCOL",
            )
            await ws.send_text(jdump({"type":"error","code":"BAD_PROTOCOL","detail":"start required"}))
            await ws.close()
            return

        api_key = (cfg.get("api_key") or "").strip()
        api_key_hash = hash_value(api_key)
        if not api_key:
            WS_REJECTS.labels(reason="UNAUTHORIZED").inc()
            close_reason = "unauthorized"
            emit_eval_event(
                log,
                "ws_session_rejected",
                session_id=session_id,
                sampled=sampled,
                reason="UNAUTHORIZED",
                code="UNAUTHORIZED",
                api_key_hash=api_key_hash,
            )
            await ws.send_text(jdump({"type":"error","code":"UNAUTHORIZED"}))
            await ws.close()
            return

        # Admission control (per API key)
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
                await ws.send_text(jdump({"type":"error","code":res.reason}))
                await ws.close()
                return
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
                await ws.send_text(jdump({"type":"error","code":reason}))
                await ws.close()
                return

        sample_rate = int(cfg.get("sample_rate", 16000))
        frame_ms = int(cfg.get("frame_ms", 20))
        decoder = (cfg.get("decoder") or "rnnt").lower()
        language = (cfg.get("language") or "auto").strip()
        call_id = (cfg.get("call_id") or "").strip()
        worker_session_key = call_id or session_id

        vad = VADSegmenter(
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            mode=2,
            end_silence_ms=VAD_END_SILENCE_MS,
            max_utt_ms=VAD_MAX_UTT_MS,
        )
        await ws.send_text(jdump({"type":"ready","call_id":call_id or None}))
        emit_eval_event(
            log,
            "ws_session_started",
            call_id=call_id or None,
            session_id=session_id,
            sampled=sampled,
            api_key_hash=api_key_hash,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            decoder=decoder,
            language=language,
        )

        last_partial = ""
        last_partial_ts_ms = 0
        utt_start_time = 0.0
        last_audio_frame_ts_ms = 0

        while True:
            msg = await ws.receive()
            out = None
            if msg.get("type") == "websocket.disconnect":
                close_reason = "client_disconnect"
                break

            if "text" in msg and msg["text"]:
                t = json.loads(msg["text"])
                if t.get("type") == "stop":
                    close_reason = "client_stop"
                    await ws.send_text(jdump({"type":"done"}))
                    break
                continue

            if "bytes" in msg and msg["bytes"]:
                frame = msg["bytes"]
                frame_received_ts_ms = int(time.time() * 1000)
                AUDIO_BYTES_RECEIVED.inc(len(frame))
                AUDIO_FRAMES_RECEIVED.inc()
                total_audio_bytes += len(frame)
                last_audio_frame_ts_ms = frame_received_ts_ms

                # Throughput abuse guard
                if not guard.add(len(frame)):
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
                    await ws.send_text(jdump({"type":"error","code":"TOO_MUCH_DATA"}))
                    await ws.close()
                    return

                events, audio_ready = vad.push(frame)
                for e in events:
                    if e == "speech_start":
                        utt_start_time = time.time()
                    VAD_FRAMES.labels(state=e).inc()
                    await ws.send_text(jdump({"type":"vad","state":e}))

                now_ms = int(time.time() * 1000)
                partial_utterance_id = f"utt-{utterance_count + 1:04d}"

                # Partial decode cadence is configurable to avoid overloading worker under high concurrency.
                if (
                    vad.in_speech
                    and (now_ms - last_partial_ts_ms) > PARTIAL_DECODE_INTERVAL_MS
                    and len(vad.buffer) > sample_rate * 2 * 1
                ):
                    out = None
                    if breaker.allow() and not worker_sem.locked():
                        async with worker_sem:
                            partial_t0 = time.time()
                            try:
                                out = await worker.transcribe(
                                    bytes(vad.buffer),
                                    sample_rate,
                                    decoder,
                                    language,
                                    mode="partial",
                                    call_id=call_id or None,
                                    session_id=worker_session_key,
                                    utterance_id=partial_utterance_id,
                                    sampled=sampled,
                                )
                                # Approximate Gateway latency for partials (not perfect but valid)
                                GATEWAY_LATENCY.observe(max(0, time.time() - partial_t0))

                                partial_latency_ms = int((time.time() - partial_t0) * 1000)
                                breaker.on_success()
                            except Exception:
                                breaker.on_failure()
                                continue

                    if out and out.text and out.text != last_partial:
                        last_partial = out.text
                        last_partial_ts_ms = now_ms
                        partial_send_ts_ms = int(time.time() * 1000)
                        await ws.send_text(
                            jdump(
                                {
                                    "type": "partial",
                                    "text": out.text,
                                    "ts_ms": now_ms,
                                    "language": out.language,
                                    "language_source": out.language_source,
                                }
                            )
                        )
                        emit_eval_event(
                            log,
                            "partial_sent",
                            call_id=call_id or None,
                            session_id=session_id,
                            utterance_id=partial_utterance_id,
                            sampled=sampled,
                            worker_latency_ms=partial_latency_ms,
                            frame_received_ts_ms=frame_received_ts_ms,
                            ws_send_ts_ms=partial_send_ts_ms,
                            from_last_audio_frame_ms=max(0, partial_send_ts_ms - last_audio_frame_ts_ms),
                            resolved_language=out.language or None,
                            language_source=out.language_source or None,
                            **text_metadata(out.text),
                        )

                # Finalize on VAD end / max_utt
                if audio_ready:
                    UTTERANCES.inc()
                    utterance_count += 1
                    utterance_id = f"utt-{utterance_count:04d}"
                    finalize_trigger = "max_utt" if "max_utt" in events else "speech_end"
                    emit_eval_event(
                        log,
                        "vad_segment_finalized",
                        call_id=call_id or None,
                        session_id=session_id,
                        utterance_id=utterance_id,
                        sampled=sampled,
                        audio_bytes=len(audio_ready),
                        trigger=finalize_trigger,
                    )

                    if not breaker.allow():
                        close_reason = "overloaded"
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
                        await ws.send_text(jdump({"type":"error","code":"OVERLOADED"}))
                        await ws.close()
                        return

                    out = None
                    async with worker_sem:
                        try:
                            final_t0 = time.time()
                            out = await worker.transcribe(
                                audio_ready,
                                sample_rate,
                                decoder,
                                language,
                                mode="final",
                                call_id=call_id or None,
                                session_id=worker_session_key,
                                utterance_id=utterance_id,
                                sampled=sampled,
                            )
                            GATEWAY_LATENCY.observe(max(0, time.time() - final_t0))

                            final_latency_ms = int((time.time() - final_t0) * 1000)
                            breaker.on_success()
                        except Exception:
                            breaker.on_failure()
                            await ws.send_text(jdump({"type":"error","code":"WORKER_ERROR"}))
                            continue

                    if out is None:
                        continue

                    final_send_ts_ms = int(time.time() * 1000)
                    await ws.send_text(
                        jdump(
                            {
                                "type": "final",
                                "text": out.text,
                                "ts_ms": final_send_ts_ms,
                                "language": out.language,
                                "language_source": out.language_source,
                            }
                        )
                    )
                    emit_eval_event(
                        log,
                        "final_sent",
                        call_id=call_id or None,
                        session_id=session_id,
                        utterance_id=utterance_id,
                        sampled=sampled,
                        worker_latency_ms=final_latency_ms,
                        frame_received_ts_ms=frame_received_ts_ms,
                        ws_send_ts_ms=final_send_ts_ms,
                        from_last_audio_frame_ms=max(0, final_send_ts_ms - last_audio_frame_ts_ms),
                        trigger=finalize_trigger,
                        resolved_language=out.language or None,
                        language_source=out.language_source or None,
                        **text_metadata(out.text),
                    )
                    if utt_start_time > 0:
                        E2E_LATENCY.observe(time.time() - utt_start_time)
                        utt_start_time = 0.0
                    last_partial = ""
                    last_partial_ts_ms = 0

    except WebSocketDisconnect:
        close_reason = "client_disconnect"
    except Exception as e:
        close_reason = "server_error"
        log.exception("WS error: %s", e)
        import traceback, sys
        trace = traceback.format_exc()
        print(f"CRITICAL WS ERROR: {trace}", file=sys.stderr)
        try:
            await ws.send_text(jdump({"type":"error","code":"SERVER_ERROR","detail":str(e), "trace": trace}))
        except Exception:
            pass
    finally:
        WS_CONNECTIONS.dec()
        if close_reason != "unknown":
             WS_DISCONNECTS.labels(reason=close_reason).inc()
        if api_key:
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
            call_id=call_id or None,
            session_id=session_id,
            sampled=sampled,
            api_key_hash=api_key_hash,
            close_reason=close_reason,
            duration_ms=max(0, int(time.time() * 1000) - session_started_ms),
            total_audio_bytes=total_audio_bytes,
            utterance_count=utterance_count,
        )
        try:
            await ws.close()
        except Exception:
            pass
