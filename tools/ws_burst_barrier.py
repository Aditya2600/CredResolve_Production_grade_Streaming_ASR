#!/usr/bin/env python3
"""Barrier-synchronized asyncio burst test for the gateway WebSocket STT API.

The script starts N clients, waits until every client is connected, releases
audio streaming at the same time, waits again after all clients finish sending
audio, then sends `flush` from every client at the same time. That second
barrier is the important bit for testing backend transcription concurrency.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import random
import statistics
import sys
import time
import wave
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, ConnectionClosedOK
except ImportError:
    print("Error: 'websockets' module not found. Install it with: pip install websockets", file=sys.stderr)
    sys.exit(1)


DEFAULT_WS_URL = "ws://localhost/ws/stt"
DEFAULT_WAV_PATH = "recordings/new_test_recording_16k.wav"


@dataclass
class IncomingMessage:
    received_at: float
    raw_len: int
    payload: dict | None = None
    error: str | None = None


@dataclass(frozen=True)
class AudioPayload:
    data: bytes
    source: str


@dataclass(frozen=True)
class WavCandidate:
    path: Path
    duration_sec: float


@dataclass
class ClientResult:
    client_id: int
    ok: bool = False
    connected: bool = False
    error: str = ""
    audio_source: str = ""
    bytes_sent: int = 0
    messages: int = 0
    data_messages: int = 0
    preflush_data_messages: int = 0
    postflush_data_messages: int = 0
    vad_messages: int = 0
    connect_ms: float = 0.0
    start_wait_ms: float = 0.0
    stream_ms: float = 0.0
    flush_wait_ms: float = 0.0
    response_wait_ms: float = 0.0
    total_ms: float = 0.0
    first_audio_at: float | None = None
    flush_sent_at: float | None = None


def parse_bool(value: bool) -> str:
    return "true" if value else "false"


def update_ws_query(base_ws_url: str, params: dict[str, str]) -> str:
    parsed = urlparse(base_ws_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    return urlunparse(parsed._replace(query=urlencode(query)))


def parse_extra_query(values: list[str]) -> dict[str, str]:
    extras: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--query-param must be KEY=VALUE, got: {value}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("--query-param key cannot be empty")
        extras[key] = raw
    return extras


def build_ws_url(args: argparse.Namespace, binary_audio: bool) -> str:
    params = {
        "language-code": args.language_code,
        "model": args.model,
        "mode": args.mode,
        "sample_rate": str(args.sample_rate),
        "high_vad_sensitivity": parse_bool(args.high_vad_sensitivity),
        "vad_signals": parse_bool(args.vad_signals),
        "flush_signal": parse_bool(args.flush_signal),
        "input_audio_codec": args.input_audio_codec,
        "binary_audio": "1" if binary_audio else "0",
    }
    params.update(parse_extra_query(args.query_param))
    return update_ws_query(args.ws, params)


def pcm_l16_bytes(pcm_s16le: bytes) -> bytes:
    if len(pcm_s16le) % 2 != 0:
        raise ValueError("PCM payload must contain an even number of bytes")
    swapped = bytearray(len(pcm_s16le))
    for index in range(0, len(pcm_s16le), 2):
        swapped[index] = pcm_s16le[index + 1]
        swapped[index + 1] = pcm_s16le[index]
    return bytes(swapped)


def prepare_audio_chunk(raw_audio: bytes, encoding: str) -> bytes:
    if encoding == "pcm_l16":
        return pcm_l16_bytes(raw_audio)
    return raw_audio


def encode_audio_chunk(raw_audio: bytes, encoding: str) -> str:
    return base64.b64encode(prepare_audio_chunk(raw_audio, encoding)).decode("ascii")


def synthesize_pcm(sample_rate: int, seconds: float) -> bytes:
    total_samples = max(1, int(sample_rate * seconds))
    output = bytearray(total_samples * 2)
    for idx in range(total_samples):
        envelope = 0.55 + 0.45 * math.sin(2.0 * math.pi * 3.0 * idx / sample_rate)
        value = int(
            envelope
            * (
                2600.0 * math.sin(2.0 * math.pi * 190.0 * idx / sample_rate)
                + 1800.0 * math.sin(2.0 * math.pi * 470.0 * idx / sample_rate)
                + 700.0 * math.sin(2.0 * math.pi * 1100.0 * idx / sample_rate)
            )
        )
        value = max(-32768, min(32767, value))
        output[2 * idx] = value & 0xFF
        output[2 * idx + 1] = (value >> 8) & 0xFF
    return bytes(output)


def synthesize_locust_noise(sample_rate: int, seconds: float) -> bytes:
    if sample_rate != 16000:
        raise SystemExit("--locust-noise matches Run C and requires --sample-rate 16000")
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("--locust-noise requires numpy, matching loadtest/ws_locustfile.py") from exc

    rng = np.random.default_rng(seed=42)
    n_samples = int(sample_rate * seconds)
    pcm = (rng.standard_normal(n_samples) * 800).astype(np.int16)
    return pcm.tobytes()


def pcm_to_wav_blob(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def load_pcm_wav(path: Path, expected_sample_rate: int, max_sec: float | None) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError("WAV must be mono PCM16")
        if wav_file.getframerate() != expected_sample_rate:
            raise ValueError(
                f"WAV sample rate {wav_file.getframerate()}Hz does not match --sample-rate {expected_sample_rate}Hz"
            )
        max_frames = wav_file.getnframes()
        if max_sec is not None and max_sec > 0:
            max_frames = min(max_frames, int(expected_sample_rate * max_sec))
        return wav_file.readframes(max_frames)


def wrap_audio_payload(pcm: bytes, args: argparse.Namespace, source: str) -> AudioPayload:
    if args.input_audio_codec == "wav":
        return AudioPayload(pcm_to_wav_blob(pcm, args.sample_rate), f"{source} as clipped WAV")
    return AudioPayload(pcm, f"{source} as PCM")


def discover_wav_candidates(wav_dir: Path, expected_sample_rate: int, min_sec: float) -> list[WavCandidate]:
    if not wav_dir.exists():
        raise SystemExit(f"--wav-dir not found: {wav_dir}")
    if not wav_dir.is_dir():
        raise SystemExit(f"--wav-dir must be a directory: {wav_dir}")

    candidates: list[WavCandidate] = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                    continue
                sample_rate = wav_file.getframerate()
                if sample_rate != expected_sample_rate:
                    continue
                duration_sec = wav_file.getnframes() / float(sample_rate)
                if min_sec > 0 and duration_sec + 1e-9 < min_sec:
                    continue
        except (EOFError, OSError, wave.Error):
            continue
        candidates.append(WavCandidate(path=wav_path, duration_sec=duration_sec))
    return candidates


def load_audio_payload(args: argparse.Namespace) -> AudioPayload:
    if args.locust_noise:
        seconds = args.max_audio_sec if args.max_audio_sec and args.max_audio_sec > 0 else 10.0
        pcm = synthesize_locust_noise(args.sample_rate, seconds)
        return wrap_audio_payload(pcm, args, f"locust Run C noise {seconds:.2f}s")

    wav_arg = (args.wav or "").strip()
    if wav_arg:
        wav_path = Path(wav_arg)
    else:
        wav_path = None

    if wav_path is not None and wav_path.exists():
        pcm = load_pcm_wav(wav_path, args.sample_rate, args.max_audio_sec)
        return wrap_audio_payload(pcm, args, str(wav_path))

    if args.synthetic_sec <= 0:
        raise SystemExit(f"WAV not found: {args.wav}")

    pcm = synthesize_pcm(args.sample_rate, args.synthetic_sec)
    return wrap_audio_payload(pcm, args, f"synthetic {args.synthetic_sec:.2f}s")


def load_audio_payloads(args: argparse.Namespace, client_count: int) -> list[AudioPayload]:
    wav_dir_arg = (args.wav_dir or "").strip()
    if not wav_dir_arg:
        payload = load_audio_payload(args)
        return [payload for _ in range(client_count)]

    if args.locust_noise:
        raise SystemExit("--wav-dir cannot be combined with --locust-noise")

    min_sec = args.max_audio_sec if args.max_audio_sec and args.max_audio_sec > 0 else 0.0
    candidates = discover_wav_candidates(Path(wav_dir_arg), args.sample_rate, min_sec)
    if len(candidates) < client_count:
        raise SystemExit(
            f"--wav-dir has only {len(candidates)} eligible WAVs for {client_count} clients "
            f"(mono PCM16, {args.sample_rate}Hz, >= {min_sec:.2f}s)"
        )

    rng = random.Random(args.audio_seed) if args.audio_seed is not None else random.SystemRandom()
    selected = rng.sample(candidates, client_count)
    payloads: list[AudioPayload] = []
    for candidate in selected:
        pcm = load_pcm_wav(candidate.path, args.sample_rate, args.max_audio_sec)
        source = f"{candidate.path} ({candidate.duration_sec:.3f}s source)"
        payloads.append(wrap_audio_payload(pcm, args, source))
    return payloads


async def abort_barrier(barrier: asyncio.Barrier) -> None:
    with suppress(asyncio.BrokenBarrierError):
        if not barrier.broken:
            await barrier.abort()


async def wait_at_barrier(barrier: asyncio.Barrier, timeout_sec: float, name: str) -> None:
    try:
        await asyncio.wait_for(barrier.wait(), timeout=timeout_sec)
    except TimeoutError as exc:
        await abort_barrier(barrier)
        raise TimeoutError(f"{name} barrier timed out after {timeout_sec:.1f}s") from exc
    except asyncio.BrokenBarrierError as exc:
        raise RuntimeError(f"{name} barrier was broken") from exc


async def receive_messages(ws, queue: asyncio.Queue[IncomingMessage]) -> None:
    while True:
        try:
            raw = await ws.recv()
        except asyncio.CancelledError:
            raise
        except ConnectionClosedOK:
            return
        except ConnectionClosed as exc:
            await queue.put(IncomingMessage(time.perf_counter(), 0, error=f"connection closed: {exc}"))
            return
        except Exception as exc:
            await queue.put(IncomingMessage(time.perf_counter(), 0, error=f"{type(exc).__name__}: {exc}"))
            return

        received_at = time.perf_counter()
        if isinstance(raw, bytes):
            await queue.put(IncomingMessage(received_at, len(raw), payload=None))
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            await queue.put(IncomingMessage(received_at, len(raw.encode("utf-8")), error=f"bad JSON: {exc}"))
            continue

        if not isinstance(payload, dict):
            await queue.put(IncomingMessage(received_at, len(raw.encode("utf-8")), error="server message was not an object"))
            continue

        await queue.put(IncomingMessage(received_at, len(raw.encode("utf-8")), payload=payload))


def handle_message(result: ClientResult, item: IncomingMessage, flush_sent_at: float | None) -> None:
    if item.error:
        raise RuntimeError(item.error)

    result.messages += 1
    if item.payload is None:
        return

    msg_type = item.payload.get("type")
    if msg_type == "error":
        raise RuntimeError(f"server error: {item.payload}")
    if msg_type == "vad":
        result.vad_messages += 1
    if msg_type != "data":
        return

    result.data_messages += 1
    if flush_sent_at is not None and item.received_at >= flush_sent_at:
        result.postflush_data_messages += 1
    else:
        result.preflush_data_messages += 1


def drain_queued_messages(
    queue: asyncio.Queue[IncomingMessage],
    result: ClientResult,
    flush_sent_at: float | None,
) -> None:
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        handle_message(result, item, flush_sent_at)


async def collect_responses(
    queue: asyncio.Queue[IncomingMessage],
    result: ClientResult,
    flush_sent_at: float,
    *,
    idle_timeout_sec: float,
    max_wait_sec: float,
) -> None:
    deadline = time.perf_counter() + max_wait_sec
    idle_deadline = time.perf_counter() + idle_timeout_sec

    while True:
        drain_queued_messages(queue, result, flush_sent_at)
        now = time.perf_counter()
        wait_for = min(deadline, idle_deadline) - now
        if wait_for <= 0:
            return
        try:
            item = await asyncio.wait_for(queue.get(), timeout=wait_for)
        except TimeoutError:
            return
        handle_message(result, item, flush_sent_at)
        idle_deadline = time.perf_counter() + idle_timeout_sec


async def send_audio_frames(ws, args: argparse.Namespace, audio_payload: bytes, binary_audio: bool) -> int:
    if args.input_audio_codec == "wav":
        await ws.send(
            json.dumps(
                {
                    "audio": {
                        "data": base64.b64encode(audio_payload).decode("ascii"),
                        "sample_rate": str(args.sample_rate),
                        "encoding": "wav",
                    }
                }
            )
        )
        return len(audio_payload)

    frame_bytes = int(args.sample_rate * (args.frame_ms / 1000.0) * 2)
    if frame_bytes <= 0:
        raise ValueError("--frame-ms produced an empty audio frame")

    sent = 0
    offset = 0
    wall_start = time.perf_counter()
    frame_idx = 0

    while offset < len(audio_payload):
        chunk = audio_payload[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk += b"\x00" * (frame_bytes - len(chunk))

        if binary_audio:
            await ws.send(prepare_audio_chunk(chunk, args.input_audio_codec))
        else:
            await ws.send(
                json.dumps(
                    {
                        "audio": {
                            "data": encode_audio_chunk(chunk, args.input_audio_codec),
                            "sample_rate": str(args.sample_rate),
                            "encoding": args.input_audio_codec,
                        }
                    }
                )
            )

        offset += frame_bytes
        sent += len(chunk)
        frame_idx += 1

        if args.send_mode == "realtime":
            target = wall_start + frame_idx * (args.frame_ms / 1000.0)
            sleep_for = target - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        elif args.yield_every > 0 and frame_idx % args.yield_every == 0:
            await asyncio.sleep(0)

    return sent


async def cleanup_receiver(task: asyncio.Task | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, ConnectionClosed, RuntimeError):
        await task


async def run_client(
    client_id: int,
    args: argparse.Namespace,
    audio_payload: AudioPayload,
    start_barrier: asyncio.Barrier,
    flush_barrier: asyncio.Barrier,
) -> ClientResult:
    result = ClientResult(client_id=client_id)
    result.audio_source = audio_payload.source
    total_start = time.perf_counter()
    receiver_task: asyncio.Task | None = None

    try:
        if args.connect_stagger_ms > 0:
            await asyncio.sleep(client_id * args.connect_stagger_ms / 1000.0)

        binary_audio = args.input_audio_codec != "wav" and not args.json_base64
        ws_url = build_ws_url(args, binary_audio)
        headers = {"Api-Subscription-Key": args.api_key}

        ping_interval = None if args.ping_interval <= 0 else args.ping_interval
        ping_timeout = None if args.ping_timeout <= 0 else args.ping_timeout

        connect_start = time.perf_counter()
        async with websockets.connect(
            ws_url,
            additional_headers=headers,
            max_size=args.max_message_size,
            open_timeout=args.open_timeout,
            close_timeout=args.close_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        ) as ws:
            result.connected = True
            result.connect_ms = (time.perf_counter() - connect_start) * 1000.0

            incoming: asyncio.Queue[IncomingMessage] = asyncio.Queue()
            receiver_task = asyncio.create_task(receive_messages(ws, incoming))

            wait_started = time.perf_counter()
            await wait_at_barrier(start_barrier, args.barrier_timeout, "start")
            result.start_wait_ms = (time.perf_counter() - wait_started) * 1000.0

            stream_started = time.perf_counter()
            result.first_audio_at = stream_started
            result.bytes_sent = await send_audio_frames(ws, args, audio_payload.data, binary_audio)
            result.stream_ms = (time.perf_counter() - stream_started) * 1000.0

            drain_queued_messages(incoming, result, flush_sent_at=None)

            flush_wait_started = time.perf_counter()
            await wait_at_barrier(flush_barrier, args.barrier_timeout, "flush")
            result.flush_wait_ms = (time.perf_counter() - flush_wait_started) * 1000.0

            result.flush_sent_at = time.perf_counter()
            await ws.send(json.dumps({"type": "flush"}))

            response_started = time.perf_counter()
            await collect_responses(
                incoming,
                result,
                result.flush_sent_at,
                idle_timeout_sec=args.idle_timeout,
                max_wait_sec=args.response_timeout,
            )
            result.response_wait_ms = (time.perf_counter() - response_started) * 1000.0

            if args.expect_data and result.data_messages == 0:
                raise RuntimeError("missing data response")

            result.ok = True
    except Exception as exc:
        await abort_barrier(start_barrier)
        await abort_barrier(flush_barrier)
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        await cleanup_receiver(receiver_task)
        result.total_ms = (time.perf_counter() - total_start) * 1000.0

    return result


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_line(label: str, values: list[float], unit: str = "ms") -> str:
    if not values:
        return f"{label}: n=0"
    return (
        f"{label}: avg={statistics.mean(values):.1f}{unit} "
        f"p50={percentile(values, 0.50):.1f}{unit} "
        f"p95={percentile(values, 0.95):.1f}{unit} "
        f"min={min(values):.1f}{unit} max={max(values):.1f}{unit}"
    )


def print_summary(results: list[ClientResult]) -> None:
    ok = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]

    print("\n--- Barrier Burst Summary ---")
    print(f"Clients: {len(results)}")
    print(f"Connected: {sum(1 for result in results if result.connected)}")
    print(f"Successful: {len(ok)}")
    print(f"Failures: {len(failed)}")

    if ok:
        first_audio = [result.first_audio_at for result in ok if result.first_audio_at is not None]
        flush_sent = [result.flush_sent_at for result in ok if result.flush_sent_at is not None]
        if first_audio:
            print(f"First-audio release spread: {(max(first_audio) - min(first_audio)) * 1000.0:.2f}ms")
        if flush_sent:
            print(f"Flush send spread: {(max(flush_sent) - min(flush_sent)) * 1000.0:.2f}ms")

        print(f"Bytes sent total: {sum(result.bytes_sent for result in ok)}")
        print(f"Data messages: {sum(result.data_messages for result in ok)}")
        print(f"Pre-flush data messages: {sum(result.preflush_data_messages for result in ok)}")
        print(f"Post-flush data messages: {sum(result.postflush_data_messages for result in ok)}")
        print(f"VAD messages: {sum(result.vad_messages for result in ok)}")
        print(metric_line("Connect", [result.connect_ms for result in ok]))
        print(metric_line("Stream send", [result.stream_ms for result in ok]))
        print(metric_line("Flush barrier wait", [result.flush_wait_ms for result in ok]))
        print(metric_line("Response wait", [result.response_wait_ms for result in ok]))
        print(metric_line("Total session", [result.total_ms for result in ok]))

    if failed:
        print("\nErrors:")
        for error, count in Counter(result.error for result in failed).most_common():
            print(f"  {count}x {error}")


def print_json_summary(results: list[ClientResult]) -> None:
    payload = {
        "clients": len(results),
        "connected": sum(1 for result in results if result.connected),
        "successful": sum(1 for result in results if result.ok),
        "failures": sum(1 for result in results if not result.ok),
        "results": [result.__dict__ for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Barrier-synchronized asyncio burst test for /ws/stt concurrency"
    )
    parser.add_argument("--ws", default=DEFAULT_WS_URL, help="WebSocket URL")
    parser.add_argument("--wav", default=DEFAULT_WAV_PATH, help="Path to mono PCM16 WAV for all clients")
    parser.add_argument(
        "--wav-dir",
        default="",
        help="Directory of mono PCM16 WAVs; samples one distinct eligible WAV per client",
    )
    parser.add_argument(
        "--audio-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible --wav-dir selection",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of concurrent clients")
    parser.add_argument("--api-key", default="dev", help="Api-Subscription-Key")
    parser.add_argument("--language-code", default="hi", help="Negotiated language-code query param")
    parser.add_argument("--model", default="credresolve:v1", help="Public model query param")
    parser.add_argument("--mode", default="transcribe", help="Public mode query param")
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000])
    parser.add_argument("--frame-ms", type=int, default=20, help="PCM frame size in ms")
    parser.add_argument(
        "--input-audio-codec",
        default="pcm_s16le",
        choices=["wav", "pcm_s16le", "pcm_l16", "pcm_raw"],
    )
    parser.add_argument(
        "--send-mode",
        choices=["realtime", "burst"],
        default="realtime",
        help="Pace audio at frame rate or send frames as fast as possible",
    )
    parser.add_argument(
        "--json-base64",
        action="store_true",
        help="Send legacy JSON/base64 PCM frames instead of binary websocket frames",
    )
    parser.add_argument("--vad-signals", action="store_true", help="Ask gateway to emit VAD events")
    parser.add_argument("--high-vad-sensitivity", action="store_true")
    parser.add_argument("--flush-signal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--query-param",
        action="append",
        default=[],
        help="Additional websocket query param as KEY=VALUE; may be repeated",
    )
    parser.add_argument("--max-audio-sec", type=float, default=10.0, help="Max WAV duration to send")
    parser.add_argument(
        "--synthetic-sec",
        type=float,
        default=5.0,
        help="Synthetic audio duration if --wav is missing or empty; <=0 disables fallback",
    )
    parser.add_argument(
        "--locust-noise",
        action="store_true",
        help="Use the exact no-AUDIO_FILE payload shape from loadtest/ws_locustfile.py: numpy noise seed 42",
    )
    parser.add_argument(
        "--connect-stagger-ms",
        type=float,
        default=0.0,
        help="Optional delay per client before opening the websocket",
    )
    parser.add_argument(
        "--barrier-timeout",
        type=float,
        default=60.0,
        help="Seconds each barrier can wait before failing the burst",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=2.0,
        help="Stop collecting responses after this many idle seconds",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=45.0,
        help="Maximum seconds to collect responses after flush",
    )
    parser.add_argument("--expect-data", action="store_true", help="Fail clients that receive no data messages")
    parser.add_argument(
        "--ping-interval",
        type=float,
        default=60.0,
        help="WebSocket ping interval in seconds; <=0 disables client pings",
    )
    parser.add_argument(
        "--ping-timeout",
        type=float,
        default=60.0,
        help="WebSocket ping timeout in seconds; <=0 disables ping timeout",
    )
    parser.add_argument("--open-timeout", type=float, default=30.0)
    parser.add_argument("--close-timeout", type=float, default=10.0)
    parser.add_argument("--max-message-size", type=int, default=20_000_000)
    parser.add_argument(
        "--yield-every",
        type=int,
        default=25,
        help="In burst send-mode, yield to the event loop every N frames; <=0 disables",
    )
    parser.add_argument("--json", action="store_true", help="Print per-client JSON instead of text summary")
    return parser


async def amain(args: argparse.Namespace) -> int:
    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if args.response_timeout <= 0:
        raise SystemExit("--response-timeout must be positive")
    if args.idle_timeout <= 0:
        raise SystemExit("--idle-timeout must be positive")

    audio_payloads = load_audio_payloads(args, args.n)
    binary_audio = args.input_audio_codec != "wav" and not args.json_base64
    ws_url = build_ws_url(args, binary_audio)

    banner_stream = sys.stderr if args.json else sys.stdout
    print(f"WebSocket: {ws_url}", file=banner_stream)
    unique_sources = {payload.source for payload in audio_payloads}
    if len(unique_sources) == 1:
        print(f"Audio: {audio_payloads[0].source}, {len(audio_payloads[0].data)} bytes", file=banner_stream)
    else:
        byte_counts = sorted({len(payload.data) for payload in audio_payloads})
        if len(byte_counts) == 1:
            byte_summary = f"{byte_counts[0]} bytes/client"
        else:
            byte_summary = f"{byte_counts[0]}-{byte_counts[-1]} bytes/client"
        seed_label = str(args.audio_seed) if args.audio_seed is not None else "system-random"
        print(
            f"Audio: {len(unique_sources)} distinct WAVs from {args.wav_dir}, {byte_summary}, seed={seed_label}",
            file=banner_stream,
        )
        print("Selected audio:", file=banner_stream)
        for client_id, payload in enumerate(audio_payloads):
            print(f"  client {client_id:02d}: {payload.source}", file=banner_stream)
    print(
        f"Clients: {args.n}, send_mode={args.send_mode}, barrier_timeout={args.barrier_timeout:.1f}s",
        file=banner_stream,
    )

    start_barrier = asyncio.Barrier(args.n)
    flush_barrier = asyncio.Barrier(args.n)
    started_at = time.perf_counter()

    tasks = [
        asyncio.create_task(run_client(client_id, args, audio_payloads[client_id], start_barrier, flush_barrier))
        for client_id in range(args.n)
    ]
    results = await asyncio.gather(*tasks)
    total_ms = (time.perf_counter() - started_at) * 1000.0

    if args.json:
        print_json_summary(results)
    else:
        print_summary(results)
        print(f"\nWall time: {total_ms:.1f}ms")

    return 0 if all(result.ok for result in results) else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nTest cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
