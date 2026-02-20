from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
import numpy as np

from asr.eval.metrics import latency_stats
from asr.preprocess import float32_to_pcm16le, load_wav_mono


def run_benchmark(
    *,
    url: str,
    wav_path: str,
    runs: int = 20,
    sample_rate: int | None = None,
) -> dict[str, float]:
    audio, sr = load_wav_mono(wav_path)
    sr = sample_rate or sr
    pcm = float32_to_pcm16le(audio)

    lat_ms: list[float] = []
    durations = [len(audio) / float(sr)] * runs

    with httpx.Client(timeout=30.0) as client:
        for _ in range(runs):
            t0 = time.time()
            resp = client.post(
                url,
                content=pcm,
                headers={"X-Sample-Rate": str(sr), "X-Mode": "final"},
            )
            resp.raise_for_status()
            _ = resp.json()
            lat_ms.append((time.time() - t0) * 1000.0)

    stats = latency_stats(lat_ms, durations)
    return {
        "runs": float(runs),
        "latency_p50_ms": stats.p50_ms,
        "latency_p95_ms": stats.p95_ms,
        "rtf": stats.rtf,
        "audio_seconds": float(durations[0] if durations else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency benchmark for streaming_server HTTP endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:8010/v1/transcribe")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out", default="asr/artifacts/eval/latency_report.json")
    args = parser.parse_args()

    report = run_benchmark(url=args.url, wav_path=args.wav, runs=max(1, args.runs))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
