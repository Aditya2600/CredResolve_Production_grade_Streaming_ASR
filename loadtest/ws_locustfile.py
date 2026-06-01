"""Locust load test for the gateway WebSocket STT endpoint.

Each virtual user opens a WebSocket session against `/ws/stt`, streams 20 ms
PCM frames in real time for `STREAM_SECONDS`, reads gateway events while
streaming, sends a `flush`, then closes and starts a new session. Locust events
separate connect cost, paced audio send time, server processing time reported
by the gateway, finals produced before flush, and finals produced after flush.

Env knobs:
  WS_API_KEY              api-subscription-key header value (default: dev)
  WS_LANGUAGE             language-code query param (default: auto)
  WS_MODEL                model query param (default: credresolve:v1)
  WS_MODE                 mode query param (default: transcribe)
  WS_PATH                 ws path (default: /ws/stt)
  WS_BINARY_AUDIO         send raw binary audio frames (default: true)
  WS_VAD_SIGNALS          vad_signals query param (default: false)
  WS_FLUSH_SIGNAL         flush_signal query param (default: true)
  STREAM_SECONDS          seconds of audio to stream per session (default: 10)
  FINAL_TIMEOUT_SECONDS   seconds to wait for a final after flush (default: 15)
  POST_FLUSH_GRACE_SECONDS
                          seconds to wait after flush if a final arrived early (default: 1)
  AUDIO_FILE              optional 16kHz mono wav; if unset, synthesizes silence
"""
import asyncio
import base64
import contextlib
import json
import os
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed
from locust import User, between, events, task

SAMPLE_RATE = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 320
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # int16

API_KEY = os.environ.get("WS_API_KEY", "dev")
LANGUAGE = os.environ.get("WS_LANGUAGE", "auto")
MODEL = os.environ.get("WS_MODEL", "credresolve:v1")
MODE = os.environ.get("WS_MODE", "transcribe")
WS_PATH = os.environ.get("WS_PATH", "/ws/stt")
STREAM_SECONDS = float(os.environ.get("STREAM_SECONDS", "10"))
FINAL_TIMEOUT_SECONDS = float(os.environ.get("FINAL_TIMEOUT_SECONDS", "15"))
POST_FLUSH_GRACE_SECONDS = float(os.environ.get("POST_FLUSH_GRACE_SECONDS", "1"))
AUDIO_FILE = os.environ.get("AUDIO_FILE", "")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "f", "no", "n", "off"}


BINARY_AUDIO = _env_bool("WS_BINARY_AUDIO", True)
VAD_SIGNALS = _env_bool("WS_VAD_SIGNALS", False)
FLUSH_SIGNAL = _env_bool("WS_FLUSH_SIGNAL", True)


@dataclass(frozen=True)
class IncomingMessage:
    received_at: float
    raw: str | bytes
    payload: dict | None
    exc: Exception | None = None


def _load_audio_bytes() -> bytes:
    """Return raw 16-bit PCM bytes at 16 kHz mono."""
    if AUDIO_FILE and Path(AUDIO_FILE).exists():
        with wave.open(AUDIO_FILE, "rb") as wf:
            assert wf.getnchannels() == 1, "AUDIO_FILE must be mono"
            assert wf.getsampwidth() == 2, "AUDIO_FILE must be 16-bit"
            assert wf.getframerate() == SAMPLE_RATE, "AUDIO_FILE must be 16 kHz"
            return wf.readframes(wf.getnframes())
    # Low-amplitude white noise so the worker has signal but nothing meaningful.
    rng = np.random.default_rng(seed=42)
    n_samples = int(SAMPLE_RATE * STREAM_SECONDS)
    pcm = (rng.standard_normal(n_samples) * 800).astype(np.int16)
    return pcm.tobytes()


_AUDIO_CACHE = _load_audio_bytes()


def _build_ws_url(host: str) -> str:
    parsed = urlparse(host)
    scheme = "wss" if parsed.scheme in ("wss", "https") else "ws"
    netloc = parsed.netloc or parsed.path
    query = urlencode(
        {
            "language-code": LANGUAGE,
            "model": MODEL,
            "mode": MODE,
            "sample_rate": SAMPLE_RATE,
            "vad_signals": str(VAD_SIGNALS).lower(),
            "flush_signal": str(FLUSH_SIGNAL).lower(),
            "input_audio_codec": "pcm_s16le",
            **({"binary_audio": "1"} if BINARY_AUDIO else {}),
        }
    )
    return urlunparse((scheme, netloc, WS_PATH, "", query, ""))


def _fire(name: str, start: float, *, response_length: int = 0, exc: Exception | None = None) -> None:
    _fire_elapsed(
        name,
        (time.perf_counter() - start) * 1000,
        response_length=response_length,
        exc=exc,
    )


def _fire_elapsed(
    name: str,
    response_time_ms: float,
    *,
    response_length: int = 0,
    exc: Exception | None = None,
) -> None:
    events.request.fire(
        request_type="WS",
        name=name,
        response_time=max(0.0, response_time_ms),
        response_length=response_length,
        exception=exc,
    )


def _closed_exc(phase: str, exc: ConnectionClosed) -> RuntimeError:
    return RuntimeError(f"websocket closed during {phase}: {exc}")


def _response_length(raw: str | bytes) -> int:
    if isinstance(raw, bytes):
        return len(raw)
    return len(raw.encode("utf-8"))


def _fire_server_processing(msg: dict, response_length: int) -> None:
    data = msg.get("data")
    metrics = data.get("metrics") if isinstance(data, dict) else None
    processing_latency = metrics.get("processing_latency") if isinstance(metrics, dict) else None
    if isinstance(processing_latency, (int, float)):
        _fire_elapsed(
            "server_processing",
            float(processing_latency) * 1000,
            response_length=response_length,
        )


async def _receive_messages(
    ws: websockets.ClientConnection,
    incoming: asyncio.Queue[IncomingMessage],
) -> None:
    while True:
        received_at = time.perf_counter()
        try:
            raw = await ws.recv()
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as exc:
            await incoming.put(IncomingMessage(received_at, b"", None, _closed_exc("receive", exc)))
            return
        except Exception as exc:
            await incoming.put(IncomingMessage(received_at, b"", None, exc))
            return

        received_at = time.perf_counter()
        if isinstance(raw, bytes):
            await incoming.put(IncomingMessage(received_at, raw, None))
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            await incoming.put(IncomingMessage(received_at, raw, None, exc))
            continue
        if not isinstance(payload, dict):
            await incoming.put(
                IncomingMessage(received_at, raw, None, RuntimeError("server message was not a JSON object"))
            )
            continue
        await incoming.put(IncomingMessage(received_at, raw, payload))


def _handle_incoming_message(
    item: IncomingMessage,
    *,
    stream_start: float,
    flush_start: float | None = None,
) -> tuple[bool, bool]:
    """Return (saw_final, saw_post_flush_final)."""
    if item.exc is not None:
        raise item.exc
    if item.payload is None:
        return False, False

    msg = item.payload
    if msg.get("type") == "error":
        raise RuntimeError(f"server error: {msg}")
    if msg.get("type") != "data":
        return False, False

    response_length = _response_length(item.raw)
    _fire_server_processing(msg, response_length)
    if flush_start is not None and item.received_at >= flush_start:
        _fire_elapsed("final", (item.received_at - flush_start) * 1000, response_length=response_length)
        return True, True

    _fire_elapsed(
        "final_before_flush",
        (item.received_at - stream_start) * 1000,
        response_length=response_length,
    )
    return True, False


async def _drain_preflush_messages(
    incoming: asyncio.Queue[IncomingMessage],
    *,
    stream_start: float,
) -> int:
    finals = 0
    while True:
        try:
            item = incoming.get_nowait()
        except asyncio.QueueEmpty:
            return finals
        saw_final, _ = _handle_incoming_message(item, stream_start=stream_start)
        if saw_final:
            finals += 1


async def _wait_for_post_flush_final(
    incoming: asyncio.Queue[IncomingMessage],
    *,
    stream_start: float,
    flush_start: float,
    timeout_seconds: float,
) -> bool:
    deadline = time.perf_counter() + timeout_seconds
    while True:
        timeout = deadline - time.perf_counter()
        if timeout <= 0:
            return False
        item = await asyncio.wait_for(incoming.get(), timeout=timeout)
        _, saw_post_flush_final = _handle_incoming_message(
            item,
            stream_start=stream_start,
            flush_start=flush_start,
        )
        if saw_post_flush_final:
            return True


async def _run_session(url: str) -> None:
    headers = {"Api-Subscription-Key": API_KEY}
    receiver_task: asyncio.Task | None = None

    connect_start = time.perf_counter()
    try:
        ws = await websockets.connect(url, additional_headers=headers, max_size=2**22)
    except Exception as exc:
        _fire("connect", connect_start, exc=exc)
        return
    _fire("connect", connect_start)
    incoming: asyncio.Queue[IncomingMessage] = asyncio.Queue()
    receiver_task = asyncio.create_task(_receive_messages(ws, incoming))

    try:
        stream_start = time.perf_counter()
        bytes_sent = 0
        preflush_finals = 0
        offset = 0
        total_frames = int(STREAM_SECONDS * 1000 / FRAME_MS)
        wall_start = time.perf_counter()
        for frame_idx in range(total_frames):
            try:
                preflush_finals += await _drain_preflush_messages(incoming, stream_start=stream_start)
            except Exception as exc:
                _fire("stream", stream_start, response_length=bytes_sent, exc=exc)
                return
            if offset + BYTES_PER_FRAME > len(_AUDIO_CACHE):
                offset = 0
            chunk = _AUDIO_CACHE[offset : offset + BYTES_PER_FRAME]
            offset += BYTES_PER_FRAME
            if BINARY_AUDIO:
                message = chunk
            else:
                message = json.dumps(
                    {
                        "audio": {
                            "data": base64.b64encode(chunk).decode("ascii"),
                            "sample_rate": SAMPLE_RATE,
                            "encoding": "pcm_s16le",
                        }
                    }
                )
            try:
                await ws.send(message)
            except ConnectionClosed as exc:
                _fire("stream", stream_start, response_length=bytes_sent, exc=_closed_exc("stream", exc))
                return
            bytes_sent += len(chunk)
            # Pace at real time so we exercise the streaming path, not raw throughput.
            target = wall_start + (frame_idx + 1) * (FRAME_MS / 1000)
            sleep_for = target - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        try:
            preflush_finals += await _drain_preflush_messages(incoming, stream_start=stream_start)
        except Exception as exc:
            _fire("stream", stream_start, response_length=bytes_sent, exc=exc)
            return
        _fire("stream", stream_start, response_length=bytes_sent)

        flush_start = time.perf_counter()
        try:
            await ws.send(json.dumps({"type": "flush"}))
        except ConnectionClosed as exc:
            _fire("final", flush_start, exc=_closed_exc("flush", exc))
            return
        try:
            got_final = await _wait_for_post_flush_final(
                incoming,
                stream_start=stream_start,
                flush_start=flush_start,
                timeout_seconds=(
                    POST_FLUSH_GRACE_SECONDS if preflush_finals else FINAL_TIMEOUT_SECONDS
                ),
            )
        except asyncio.TimeoutError as exc:
            if not preflush_finals:
                _fire("final", flush_start, exc=exc)
        except Exception as exc:
            _fire("final", flush_start, exc=exc)
        else:
            if not got_final and not preflush_finals:
                _fire("final", flush_start, exc=asyncio.TimeoutError())
    finally:
        if receiver_task is not None:
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await receiver_task
        try:
            await ws.close()
        except Exception:
            pass


# Locust runs tasks in gevent greenlets that share one OS thread. asyncio's
# running-loop state is per-thread, so two greenlets that call `asyncio.run()`
# concurrently trample each other ("asyncio.run() cannot be called from a
# running event loop"). Run all asyncio work on a single dedicated worker
# thread and submit coroutines to it from the greenlets via
# `run_coroutine_threadsafe`; the returned `concurrent.futures.Future` is what
# we block on (gevent monkey-patches threading primitives, so the wait yields
# to the hub instead of stalling other users).
_ASYNCIO_LOOP: asyncio.AbstractEventLoop | None = None
_ASYNCIO_LOOP_LOCK = threading.Lock()


def _ensure_asyncio_loop() -> asyncio.AbstractEventLoop:
    global _ASYNCIO_LOOP
    if _ASYNCIO_LOOP is not None and _ASYNCIO_LOOP.is_running():
        return _ASYNCIO_LOOP
    with _ASYNCIO_LOOP_LOCK:
        if _ASYNCIO_LOOP is not None and _ASYNCIO_LOOP.is_running():
            return _ASYNCIO_LOOP
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        threading.Thread(target=_runner, name="locust-asyncio", daemon=True).start()
        ready.wait()
        _ASYNCIO_LOOP = loop
        return loop


class WSStreamUser(User):
    wait_time = between(0.5, 1.5)

    @task
    def stream_session(self) -> None:
        url = _build_ws_url(self.host or "ws://localhost:8080")
        loop = _ensure_asyncio_loop()
        future = asyncio.run_coroutine_threadsafe(_run_session(url), loop)
        future.result()
