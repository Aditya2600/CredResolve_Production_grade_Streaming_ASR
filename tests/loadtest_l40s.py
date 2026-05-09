"""L40s concurrency / latency load test.

Replays a directory of pre-cached 8 kHz mono int16 .raw clips against the
Simplismart /v1/predict endpoint with N concurrent in-flight requests.

Prep clips with: python -m tests.fetch_vaani_loadtest_clips --n 50
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from tests.test_client_raw_streaming import API_URL, HEADERS, SAMPLE_RATE

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_connections=128, pool_maxsize=128))


def load_raw_clips(clip_dir: Path) -> list[tuple[str, bytes, float]]:
    clips = []
    for path in sorted(clip_dir.glob("*.raw")):
        data = path.read_bytes()
        n_samples = len(data) // 2  # int16
        clips.append((path.name, data, n_samples / SAMPLE_RATE))
    if not clips:
        raise SystemExit(f"no .raw clips in {clip_dir} — run fetch_vaani_loadtest_clips first")
    return clips


def send_one(raw_bytes: bytes, language: str | None, timeout: int = 60) -> dict:
    data = {"sample_rate": str(SAMPLE_RATE)}
    if language:
        data["language"] = language
    t0 = time.perf_counter()
    try:
        r = SESSION.post(
            API_URL,
            files={"audio": ("chunk.raw", raw_bytes, "application/octet-stream")},
            data=data,
            headers=HEADERS,
            timeout=timeout,
        )
        return {"ok": r.status_code == 200, "status": r.status_code,
                "lat": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "status": -1,
                "lat": time.perf_counter() - t0, "err": repr(e)}


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", default="tests/fixtures/loadtest_vaani_8k")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--total", type=int, default=500)
    ap.add_argument("--language", default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clips = load_raw_clips(Path(args.clip_dir))
    durations = [c[2] for c in clips]
    print(f"[setup] {len(clips)} clips loaded, "
          f"mean={np.mean(durations):.2f}s min={min(durations):.2f}s max={max(durations):.2f}s")

    rng = random.Random(args.seed)
    schedule = [clips[rng.randrange(len(clips))] for _ in range(args.total)]

    if args.warmup:
        print(f"[warmup] {args.warmup} sequential calls")
        for _ in range(args.warmup):
            send_one(clips[0][1], args.language)

    print(f"[run] concurrency={args.concurrency}, total={args.total}")
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(send_one, payload, args.language)
                for _, payload, _ in schedule]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    lats = [r["lat"] for r in ok]
    audio_total = sum(d for _, _, d in schedule)

    print(f"\nsuccess/fail   : {len(ok)} / {len(results) - len(ok)}")
    print(f"wall time      : {wall:.2f}s")
    print(f"throughput rps : {len(ok) / wall:.2f}")
    print(f"audio sent     : {audio_total:.2f}s")
    print(f"aggregate RTF  : {audio_total / wall:.2f}x realtime")
    if lats:
        print(f"latency mean   : {statistics.fmean(lats):.3f}s")
        print(f"p50/p95/p99    : {pct(lats, 50):.3f} / {pct(lats, 95):.3f} / {pct(lats, 99):.3f}s")
        print(f"max            : {max(lats):.3f}s")
    if len(ok) != len(results):
        codes = {}
        for r in results:
            if not r["ok"]:
                codes[r["status"]] = codes.get(r["status"], 0) + 1
        print(f"failure breakdown: {codes}")


if __name__ == "__main__":
    main()
