#!/usr/bin/env python3
"""
Simplismart Streaming ASR - WebSocket Test Client

Connects to a Simplismart WebSocket ASR endpoint, streams an audio file,
and prints transcription results with latency stats.

pip install numpy soundfile websockets httpx scipy

Then just run:   python benchmark_asr.py
"""

import asyncio
import argparse
import csv
import itertools
import json
import os
import random
import statistics
import tempfile
import time
import urllib.parse
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit these values and run the script                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

WS_URL     = "ws://localhost:8765"
AUDIO_FILE = "YOUR_AUDIO_FILE.raw"
LANGUAGE   = "hi"
REALTIME   = True   # True = stream at real-time speed, False = as fast as possible

# ── Advanced (usually no need to change) ──────────────────────────────────
SAMPLE_RATE         = 16000
RAW_SAMPLE_RATE     = 8000     # assumed sample rate for .raw files (8kHz or 16kHz)
CHUNK_SIZE          = 512      # 512 samples at 16kHz = 32ms per chunk
CHUNK_DURATION      = CHUNK_SIZE / SAMPLE_RATE
WS_OPEN_TIMEOUT     = 120.0
WS_RECV_TIMEOUT     = 30.0
HTTP_DOWNLOAD_TIMEOUT = 30.0
AUDIO_EXTENSIONS = {".wav", ".raw", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif", ".opus"}


# ── Audio loading ─────────────────────────────────────────────────────────

async def load_audio(path: str, verbose: bool = True) -> np.ndarray:
    """Load a local file (.wav, .raw, etc.) or URL, convert to 16kHz mono float32."""
    if path.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=HTTP_DOWNLOAD_TIMEOUT) as client:
            resp = await client.get(path)
            resp.raise_for_status()
        parsed = urllib.parse.urlparse(path)
        _, ext = os.path.splitext(parsed.path)
        ext = (ext or ".wav").lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        try:
            if ext == ".raw":
                audio = np.fromfile(tmp_path, dtype=np.int16).astype(np.float32) / 32768.0
                sr = RAW_SAMPLE_RATE
            else:
                audio, sr = sf.read(tmp_path)
        finally:
            os.unlink(tmp_path)
    elif path.lower().endswith(".raw"):
        audio = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
        sr = RAW_SAMPLE_RATE
        if verbose:
            print(f"  Raw PCM file detected, assuming {sr}Hz mono int16")
    else:
        audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr))
        if verbose:
            print(f"  Resampled from {sr}Hz to {SAMPLE_RATE}Hz")

    return audio


# ── WebSocket streaming ──────────────────────────────────────────────────

async def send_audio(ws, audio: np.ndarray, trace_id: str, realtime: bool = REALTIME,
                     verbose: bool = True, prefix: str = ""):
    """Send audio chunks + end_audio signal."""
    total = (len(audio) + CHUNK_SIZE - 1) // CHUNK_SIZE
    sent = 0

    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i : i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
        await ws.send((chunk * 32767).astype(np.int16).tobytes())
        sent += 1
        if verbose and sent % 100 == 0:
            print(f"{prefix}  Sent {sent}/{total} chunks ({sent/total*100:.0f}%)", end="\r")
        if realtime:
            await asyncio.sleep(CHUNK_DURATION)

    if verbose:
        print(f"{prefix}  Sent {total}/{total} chunks (100%)          ")
    await ws.send(json.dumps({"type": "end_audio", "trace_id": trace_id}))


async def receive_results(ws, start_ms: float, verbose: bool = True,
                          prefix: str = "", capture_rows: bool = True):
    """Receive transcription messages until finished. Returns (partials, finals, final_texts, rows)."""
    partials = 0
    finals = 0
    final_texts = []
    rows = []

    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
            msg = json.loads(raw)
            msg_type = msg.get("type", "transcription")

            if msg_type == "end_audio":
                status = msg.get("body", {}).get("status", "")
                if verbose:
                    print(f"{prefix}  << end_audio ({status})")
                if status == "finished":
                    break
                continue

            if msg_type != "transcription":
                continue

            latency = time.time() * 1000 - start_ms
            is_final = msg.get("is_final", False)
            segments = msg.get("segments", [])
            text = segments[0].get("text", "") if segments else ""
            t0 = segments[0].get("start_time", 0) if segments else 0
            t1 = segments[0].get("end_time", 0) if segments else 0

            kind = "final" if is_final else "partial"
            if capture_rows:
                rows.append({
                    "type": kind,
                    "start_time_s": round(t0, 3),
                    "end_time_s": round(t1, 3),
                    "latency_ms": round(latency),
                    "text": text,
                })

            if is_final:
                finals += 1
                final_texts.append(text)
                if verbose:
                    print(f"{prefix}  FINAL  [{t0:.2f}s-{t1:.2f}s] ({latency:.0f}ms): {text}")
            else:
                partials += 1
                if verbose:
                    print(f"{prefix}  partial [{t0:.2f}s-{t1:.2f}s] ({latency:.0f}ms): {text}")

    except asyncio.TimeoutError:
        if verbose:
            print(f"{prefix}  Timeout after {WS_RECV_TIMEOUT}s")
    except websockets.exceptions.ConnectionClosed as e:
        if verbose:
            print(f"{prefix}  Connection closed: {e}")

    return partials, finals, final_texts, rows


# ── Main ─────────────────────────────────────────────────────────────────

def build_metadata() -> dict:
    whisper_params = {}
    if LANGUAGE:
        whisper_params["audio_language"] = LANGUAGE
    return {
        "whisper_params": whisper_params,
        "streaming_params": {
            "sample_rate": SAMPLE_RATE,
            "enable_partial_transcripts": True,
        },
    }


def result_csv_for_audio(audio_file: str) -> str:
    if audio_file.startswith(("http://", "https://")):
        return "results.csv"
    return os.path.splitext(audio_file)[0] + "_results.csv"


async def run_request(audio_file: str, request_id: int = 1, verbose: bool = True,
                      capture_rows: bool = True) -> dict:
    prefix = f"[{request_id:03d}]" if not verbose else ""
    result = {
        "request_id": request_id,
        "audio_file": audio_file,
        "status": "error",
        "error": "",
        "duration_s": 0.0,
        "total_ms": 0,
        "rtf": 0.0,
        "partials": 0,
        "finals": 0,
        "text": "",
        "rows": [],
    }

    try:
        if verbose:
            print("  Loading audio...")
        audio = await load_audio(audio_file, verbose=verbose)
        duration = len(audio) / SAMPLE_RATE
        result["duration_s"] = round(duration, 3)
        if verbose:
            print(f"  Loaded: {duration:.2f}s ({len(audio):,} samples at {SAMPLE_RATE}Hz)")

        if verbose:
            print(f"\n  Connecting to {WS_URL}...")
        t0 = time.time() * 1000

        async with websockets.connect(WS_URL, open_timeout=WS_OPEN_TIMEOUT) as ws:
            conn_ms = time.time() * 1000 - t0
            if verbose:
                print(f"  Connected in {conn_ms:.0f}ms")

            await ws.send(json.dumps(build_metadata()))

            if verbose:
                print(f"\n  Streaming audio...\n")
            stream_start = time.time() * 1000

            _, (partials, finals, texts, rows) = await asyncio.gather(
                send_audio(ws, audio, audio_file, realtime=REALTIME, verbose=verbose, prefix=prefix),
                receive_results(ws, stream_start, verbose=verbose, prefix=prefix, capture_rows=capture_rows),
            )

        total_ms = time.time() * 1000 - t0
        rtf = (total_ms / 1000) / duration if duration > 0 else 0
        full_text = " ".join(texts).strip()

        result.update({
            "status": "ok",
            "total_ms": round(total_ms),
            "rtf": round(rtf, 3),
            "partials": partials,
            "finals": finals,
            "text": full_text,
            "rows": rows,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  ERROR: {result['error']}")

    return result


def write_detail_csv(audio_file: str, result: dict):
    csv_file = result_csv_for_audio(audio_file)
    rows = result.get("rows", [])
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "start_time_s", "end_time_s", "latency_ms", "text"])
        w.writeheader()
        w.writerows(rows)
        w.writerow({})
        w.writerow({"type": "SUMMARY", "start_time_s": f"duration={result['duration_s']:.2f}s",
                     "end_time_s": f"total={result['total_ms']}ms", "latency_ms": f"rtf={result['rtf']:.2f}x",
                     "text": result["text"]})
    return csv_file


async def run_single(audio_file: str, verbose: bool = True):
    print("=" * 60)
    print("  Simplismart Streaming ASR - WebSocket Test")
    print("=" * 60)
    print(f"  Endpoint : {WS_URL}")
    print(f"  Audio    : {AUDIO_FILE}")
    print(f"  Language : {LANGUAGE}")
    print(f"  Realtime : {REALTIME}")
    print()

    result = await run_request(audio_file, verbose=verbose, capture_rows=True)

    # Results
    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    if result["status"] != "ok":
        print(f"  Status          : {result['status']}")
        print(f"  Error           : {result['error']}")
        return

    rtf = result["rtf"]
    total_ms = result["total_ms"]
    print(f"  Audio duration  : {result['duration_s']:.2f}s")
    print(f"  Total time      : {total_ms:.0f}ms ({total_ms/1000:.2f}s)")
    print(f"  Real-time factor: {rtf:.2f}x {'(faster)' if rtf < 1 else '(slower)' if rtf > 1 else ''}")
    print(f"  Partials        : {result['partials']}")
    print(f"  Finals          : {result['finals']}")

    if result["text"]:
        print(f"\n  Transcription:\n  {result['text']}")
    else:
        print(f"\n  (No transcription received)")

    # Write CSV
    csv_file = write_detail_csv(audio_file, result)
    print(f"\n  CSV saved to: {csv_file}")
    print()


def collect_audio_files(dataset: str | None, audio_file: str | None) -> list[str]:
    if dataset:
        root = Path(dataset).expanduser()
        files = [
            str(path)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ]
        return sorted(files)

    if audio_file:
        return [audio_file]

    if AUDIO_FILE != "YOUR_AUDIO_FILE.raw":
        return [AUDIO_FILE]

    return []


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


async def run_benchmark(audio_files: list[str], concurrency: int, requests: int | None,
                        random_count: int | None, seed: int | None,
                        output_csv: str, verbose: bool):
    if random_count:
        if random_count < 1:
            raise SystemExit("--random must be at least 1")
        if random_count > len(audio_files):
            raise SystemExit(
                f"--random {random_count} requested, but only {len(audio_files)} audio file(s) were found."
            )
        rng = random.Random(seed)
        selected = rng.sample(audio_files, random_count)
    elif requests:
        selected = list(itertools.islice(itertools.cycle(audio_files), requests))
    elif len(audio_files) == 1 and concurrency > 1:
        selected = [audio_files[0]] * concurrency
    else:
        selected = audio_files

    if len(selected) < concurrency:
        print(f"  Note: only {len(selected)} request(s) queued, so peak concurrency is {len(selected)}.")

    print("=" * 60)
    print("  Simplismart Streaming ASR - Concurrent Benchmark")
    print("=" * 60)
    print(f"  Endpoint    : {WS_URL}")
    print(f"  Audio files : {len(audio_files)} found")
    print(f"  Requests    : {len(selected)}")
    print(f"  Concurrency : {concurrency}")
    print(f"  Language    : {LANGUAGE}")
    print(f"  Realtime    : {REALTIME}")
    if random_count:
        seed_text = seed if seed is not None else "system"
        print(f"  Random pick : {random_count} file(s), seed={seed_text}")
    print(f"  Output CSV  : {output_csv}")
    print()

    semaphore = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()
    completed = 0
    started_at = time.time()

    async def worker(request_id: int, audio_file: str) -> dict:
        nonlocal completed
        async with semaphore:
            result = await run_request(
                audio_file,
                request_id=request_id,
                verbose=verbose,
                capture_rows=False,
            )
        async with progress_lock:
            completed += 1
            status = result["status"]
            elapsed = time.time() - started_at
            print(f"  [{completed}/{len(selected)}] request {request_id:03d} {status} "
                  f"({result['total_ms']}ms, rtf={result['rtf']:.2f}x, elapsed={elapsed:.1f}s)")
            if result["error"]:
                print(f"      {result['error']}")
        return result

    results = await asyncio.gather(
        *(worker(idx, path) for idx, path in enumerate(selected, start=1))
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "request_id", "audio_file", "status", "error", "duration_s",
            "total_ms", "rtf", "partials", "finals", "text",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for result in results:
            w.writerow({key: result[key] for key in fieldnames})

    ok = [result for result in results if result["status"] == "ok"]
    failed = len(results) - len(ok)
    total_wall_s = time.time() - started_at
    total_audio_s = sum(result["duration_s"] for result in ok)
    total_times = [result["total_ms"] for result in ok]
    rtfs = [result["rtf"] for result in ok]

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Completed       : {len(ok)}/{len(results)} ok, {failed} failed")
    print(f"  Wall time       : {total_wall_s:.2f}s")
    print(f"  Audio processed : {total_audio_s:.2f}s")
    if total_wall_s > 0:
        print(f"  Throughput      : {total_audio_s / total_wall_s:.2f} audio-sec/sec")
    if ok:
        print(f"  Request ms      : avg={statistics.mean(total_times):.0f}, "
              f"p50={statistics.median(total_times):.0f}, p95={percentile(total_times, 95):.0f}")
        print(f"  RTF             : avg={statistics.mean(rtfs):.2f}x, "
              f"p50={statistics.median(rtfs):.2f}x, p95={percentile(rtfs, 95):.2f}x")
    print(f"  CSV saved to    : {output_csv}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Simplismart streaming ASR over WebSocket.")
    parser.add_argument("audio", nargs="?", help="Audio file to stream. Repeated when --concurrency > 1.")
    parser.add_argument("--audio", dest="audio_option", help="Audio file to stream.")
    parser.add_argument("--dataset", help="Directory containing audio files to benchmark.")
    parser.add_argument("--ws-url", default=WS_URL, help=f"WebSocket endpoint. Default: {WS_URL}")
    parser.add_argument("--language", default=LANGUAGE, help=f"Audio language. Default: {LANGUAGE}")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum simultaneous WebSocket streams.")
    parser.add_argument("--requests", type=int, help="Total requests to run. Cycles through dataset if needed.")
    parser.add_argument("--random", type=int, dest="random_count",
                        help="Randomly select this many unique audio files from --dataset.")
    parser.add_argument("--seed", type=int, help="Seed for --random selection.")
    parser.add_argument("--output", default="benchmark_summary.csv", help="Summary CSV path for concurrent runs.")
    parser.add_argument("--realtime", action="store_true", help="Stream chunks at real-time speed.")
    parser.add_argument("--raw-sample-rate", type=int, default=RAW_SAMPLE_RATE,
                        help=f"Sample rate to assume for .raw files. Default: {RAW_SAMPLE_RATE}")
    parser.add_argument("--verbose", action="store_true", help="Print per-transcript logs during concurrent runs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-transcript logs for single-file runs.")
    return parser.parse_args()


async def main():
    global WS_URL, LANGUAGE, REALTIME, RAW_SAMPLE_RATE, AUDIO_FILE

    args = parse_args()
    WS_URL = args.ws_url
    LANGUAGE = args.language
    REALTIME = args.realtime
    RAW_SAMPLE_RATE = args.raw_sample_rate
    audio_file = args.audio_option or args.audio
    if audio_file:
        AUDIO_FILE = audio_file

    audio_files = collect_audio_files(args.dataset, audio_file)
    if not audio_files:
        raise SystemExit("No audio files found. Pass --dataset /path/to/vanni or an audio file.")

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.random_count and not args.dataset:
        raise SystemExit("--random requires --dataset")
    if args.random_count and args.requests:
        raise SystemExit("Use either --random or --requests, not both")

    if args.concurrency == 1 and not args.requests and not args.random_count and len(audio_files) == 1:
        await run_single(audio_files[0], verbose=not args.quiet)
    else:
        verbose = args.verbose and not args.quiet
        await run_benchmark(
            audio_files,
            args.concurrency,
            args.requests,
            args.random_count,
            args.seed,
            args.output,
            verbose=verbose,
        )


if __name__ == "__main__":
    asyncio.run(main())
