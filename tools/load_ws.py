#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
import wave
from contextlib import suppress
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    import websockets
except ImportError:
    print("Error: 'websockets' module not found. Install it with: pip install websockets")
    sys.exit(1)


DEFAULT_WS_URL = "ws://localhost/ws/stt"
DEFAULT_WAV_PATH = "sample_data/sample_16k_mono.wav"


def update_ws_query(base_ws_url: str, params: dict[str, str]) -> str:
    parsed = urlparse(base_ws_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    return urlunparse(parsed._replace(query=urlencode(query)))


def pcm_l16_bytes(pcm_s16le: bytes) -> bytes:
    if len(pcm_s16le) % 2 != 0:
        raise ValueError("PCM payload must contain an even number of bytes")
    swapped = bytearray(len(pcm_s16le))
    for index in range(0, len(pcm_s16le), 2):
        swapped[index] = pcm_s16le[index + 1]
        swapped[index + 1] = pcm_s16le[index]
    return bytes(swapped)


def encode_audio_chunk(raw_audio: bytes, encoding: str) -> str:
    if encoding == "pcm_l16":
        raw_audio = pcm_l16_bytes(raw_audio)
    return base64.b64encode(raw_audio).decode("ascii")


def load_pcm_wav(path: str, expected_sample_rate: int, max_sec: float | None) -> bytes:
    try:
        with wave.open(path, "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("WAV must be mono PCM16")
            if wav_file.getframerate() != expected_sample_rate:
                raise ValueError(
                    f"WAV sample rate {wav_file.getframerate()}Hz does not match --sample-rate {expected_sample_rate}Hz"
                )
            frames = wav_file.readframes(wav_file.getnframes())
            if max_sec:
                max_bytes = int(expected_sample_rate * 2 * max_sec)
                frames = frames[:max_bytes]
            return frames
    except Exception as exc:
        raise SystemExit(f"Error loading WAV: {exc}") from exc


class LoadTestStats:
    def __init__(self):
        self.connected = 0
        self.completed_ok = 0
        self.errors = 0
        self.latencies: list[float] = []
        self.error_types: dict[str, int] = {}

    def add_error(self, err: Exception):
        self.errors += 1
        msg = str(err)
        if "Protocol Error" in msg:
            err_type = "ProtocolError"
        elif "Server Error" in msg:
            err_type = f"ServerError: {msg.split(': ')[-1]}"
        elif "Timeout" in msg:
            err_type = "Timeout"
        elif "Connection Closed" in msg:
            code = msg.split(":")[1].strip().split(" ")[0]
            err_type = f"ConnectionClosed: {code}"
        else:
            err_type = type(err).__name__
        self.error_types[err_type] = self.error_types.get(err_type, 0) + 1

    def add_latency(self, latency: float):
        self.latencies.append(latency)

    def print_summary(self):
        print("\n--- Load Test Summary ---")
        print(f"Sessions Established: {self.connected}")
        print(f"Successful Completions: {self.completed_ok}")
        print(f"Failures: {self.errors}")

        if self.latencies:
            print(f"Avg Session Duration: {statistics.mean(self.latencies):.2f}s")
            if len(self.latencies) > 1:
                print(f"P95 Session Duration: {statistics.quantiles(self.latencies, n=20)[18]:.2f}s")
            print(f"Min/Max Duration: {min(self.latencies):.2f}s / {max(self.latencies):.2f}s")

        if self.errors > 0:
            print("\nError Breakdown:")
            for err_type, count in self.error_types.items():
                print(f"  {err_type}: {count}")


stats = LoadTestStats()


async def cleanup_stream_task(task: asyncio.Task | None):
    if task is None:
        return
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, websockets.exceptions.ConnectionClosed, TimeoutError):
        await task


async def drain_messages(websocket, timeout_sec: float) -> int:
    data_messages = 0
    while True:
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            break

        payload = json.loads(raw)
        if payload.get("type") == "error":
            message = payload.get("message", "")
            raise Exception(f"Server Error: {payload.get('code')} - {message}".strip())
        if payload.get("type") == "data":
            data_messages += 1
    return data_messages


async def run_client(client_id, args, audio_payload: bytes):
    if args.ramp_ms > 0:
        await asyncio.sleep((client_id * args.ramp_ms) / 1000.0)

    ws_url = update_ws_query(
        args.ws,
        {
            "language-code": args.language_code,
            "model": args.model,
            "mode": args.mode,
            "sample_rate": str(args.sample_rate),
            "high_vad_sensitivity": str(args.high_vad_sensitivity).lower(),
            "vad_signals": str(args.vad_signals).lower(),
            "flush_signal": str(args.flush_signal).lower(),
            "input_audio_codec": args.input_audio_codec,
        },
    )
    auth_headers = {"Api-Subscription-Key": args.api_key}
    start_time = time.time()

    stream_task: asyncio.Task | None = None
    try:
        ping_interval = None if args.ping_interval <= 0 else args.ping_interval
        ping_timeout = None if args.ping_timeout <= 0 else args.ping_timeout

        async with websockets.connect(
            ws_url,
            max_size=None,
            open_timeout=args.open_timeout,
            close_timeout=args.close_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            additional_headers=auth_headers,
        ) as websocket:
            stats.connected += 1

            if args.input_audio_codec == "wav":
                await websocket.send(
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
            else:
                chunk_size = int(args.sample_rate * (args.frame_ms / 1000.0) * 2)
                stream_task = asyncio.create_task(
                    stream_audio(
                        websocket,
                        audio_payload,
                        chunk_size,
                        args.frame_ms,
                        args.input_audio_codec,
                        args.sample_rate,
                    )
                )
                await stream_task
                stream_task = None

            await websocket.send(json.dumps({"type": "flush"}))
            data_messages = await drain_messages(websocket, timeout_sec=args.timeout_sec)

            if args.expect_data and data_messages == 0:
                raise Exception("Missing data response")

            duration = time.time() - start_time
            stats.add_latency(duration)
            stats.completed_ok += 1
    except Exception as exc:
        stats.add_error(exc)
        print(f"Client {client_id} Error: {exc}")
    finally:
        await cleanup_stream_task(stream_task)


async def stream_audio(
    ws,
    audio_data: bytes,
    chunk_size: int,
    frame_ms: int,
    encoding: str,
    sample_rate: int,
):
    total_len = len(audio_data)
    offset = 0
    sleep_time = frame_ms / 1000.0

    try:
        while offset < total_len:
            chunk = audio_data[offset : offset + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b"\x00" * (chunk_size - len(chunk))
            await ws.send(
                json.dumps(
                    {
                        "audio": {
                            "data": encode_audio_chunk(chunk, encoding),
                            "sample_rate": str(sample_rate),
                            "encoding": encoding,
                        }
                    }
                )
            )
            offset += chunk_size
            await asyncio.sleep(sleep_time)
    except (websockets.exceptions.ConnectionClosed, TimeoutError):
        return


async def main():
    parser = argparse.ArgumentParser(description="Sarvam-like WebSocket STT load generator")
    parser.add_argument("--ws", default=DEFAULT_WS_URL, help="WebSocket URL")
    parser.add_argument("--wav", default=DEFAULT_WAV_PATH, help="Path to mono PCM16 wav")
    parser.add_argument("--n", type=int, default=10, help="Number of concurrent clients")
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
    parser.add_argument("--vad-signals", action="store_true")
    parser.add_argument("--high-vad-sensitivity", action="store_true")
    parser.add_argument("--flush-signal", action="store_true")
    parser.add_argument("--ramp-ms", type=int, default=100, help="Ramp-up delay per client (ms)")
    parser.add_argument("--timeout-sec", type=float, default=2.0, help="Inactivity timeout after flush")
    parser.add_argument("--max-audio-sec", type=float, default=30.0, help="Max audio duration to send")
    parser.add_argument("--expect-data", action="store_true", help="Fail if no data response is received")
    parser.add_argument(
        "--ping-interval",
        type=float,
        default=60.0,
        help="WebSocket ping interval in seconds (<=0 disables client pings)",
    )
    parser.add_argument(
        "--ping-timeout",
        type=float,
        default=60.0,
        help="WebSocket ping timeout in seconds (<=0 disables ping timeout)",
    )
    parser.add_argument(
        "--open-timeout",
        type=float,
        default=30.0,
        help="WebSocket handshake timeout in seconds",
    )
    parser.add_argument(
        "--close-timeout",
        type=float,
        default=10.0,
        help="WebSocket close timeout in seconds",
    )
    args = parser.parse_args()

    print(f"Loading WAV: {args.wav}")
    if args.input_audio_codec == "wav":
        with wave.open(args.wav, "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise SystemExit("WAV must be mono PCM16")
            if wav_file.getframerate() != args.sample_rate:
                raise SystemExit(
                    f"WAV sample rate {wav_file.getframerate()}Hz does not match --sample-rate {args.sample_rate}Hz"
                )
        with open(args.wav, "rb") as wav_file:
            audio_payload = wav_file.read()
        print(f"WAV blob size: {len(audio_payload)} bytes")
    else:
        audio_payload = load_pcm_wav(args.wav, args.sample_rate, args.max_audio_sec)
        print(f"PCM payload size: {len(audio_payload)} bytes ({len(audio_payload)/(args.sample_rate*2):.2f}s)")

    print(f"Starting {args.n} clients...")
    tasks = [run_client(i, args, audio_payload) for i in range(args.n)]

    start_global = time.time()
    await asyncio.gather(*tasks)
    total_time = time.time() - start_global

    print(f"\nTotal Test Time: {total_time:.2f}s")
    stats.print_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest Cancelled")
