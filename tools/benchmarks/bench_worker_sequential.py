"""
Sequentially POST utterances to the worker /v1/transcribe endpoint and
record per-request timing. No concurrency: one in-flight request at a time
so each request is cleanly attributable in worker / Triton logs.

Manifest format (JSONL, one object per line):
    {"utt_id": "...", "language": "hi", "bucket": "short|medium|long", "wav": "/abs/path.wav"}

Output CSV columns:
    utt_id,language,bucket,audio_s,status,http_ms,worker_audio_s,worker_proc_ms,rtf,hyp_text

Usage:
    python tools/benchmarks/bench_worker_sequential.py \
        --manifest loadtest/bench_60.jsonl \
        --worker-url http://127.0.0.1:9000 \
        --out loadtest/bench_60_results.csv
"""

import argparse
import csv
import json
import sys
import time
import wave
from pathlib import Path

import requests


def read_pcm_s16le(wav_path: str) -> tuple[bytes, int, float]:
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(f"{wav_path}: must be mono PCM16")
        sr = wf.getframerate()
        nframes = wf.getnframes()
        pcm = wf.readframes(nframes)
    return pcm, sr, nframes / float(sr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSONL manifest")
    parser.add_argument("--worker-url", default="http://127.0.0.1:9000")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--decoder", default="ctc")
    parser.add_argument("--mode", default="final")
    parser.add_argument("--timestamp-type", default="none")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Discarded warmup requests before recording")
    parser.add_argument("--sleep-ms", type=int, default=200,
                        help="Pause between requests (lets logs flush)")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    rows = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("Manifest is empty", file=sys.stderr)
        return 2

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow([
        "utt_id", "language", "bucket", "audio_s",
        "status", "http_ms", "worker_audio_s", "worker_proc_ms", "rtf", "hyp_text",
    ])

    url = args.worker_url.rstrip("/") + "/v1/transcribe"

    # Warmup: re-send the first row a few times, results discarded.
    if args.warmup > 0:
        warm = rows[0]
        try:
            pcm, sr, _ = read_pcm_s16le(warm["wav"])
        except Exception as exc:
            print(f"warmup read failed: {exc}", file=sys.stderr)
            return 2
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(sr),
            "X-Decoder": args.decoder,
            "X-Language": warm["language"],
            "X-Mode": args.mode,
            "X-Timestamp-Type": args.timestamp_type,
            "X-Session-Id": "bench-warmup",
        }
        for i in range(args.warmup):
            print(f"[warmup {i+1}/{args.warmup}] {warm['utt_id']}", file=sys.stderr)
            try:
                requests.post(url, data=pcm, headers={**headers, "X-Utterance-Id": f"warmup-{i}"},
                              timeout=args.timeout_s)
            except Exception as exc:
                print(f"  warmup error: {exc}", file=sys.stderr)
            time.sleep(args.sleep_ms / 1000.0)

    for idx, row in enumerate(rows, 1):
        utt_id = row["utt_id"]
        language = row["language"]
        bucket = row["bucket"]
        wav = row["wav"]
        try:
            pcm, sr, audio_s = read_pcm_s16le(wav)
        except Exception as exc:
            writer.writerow([utt_id, language, bucket, "", f"read_error:{exc}", "", "", "", "", ""])
            out_f.flush()
            continue

        headers = {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(sr),
            "X-Decoder": args.decoder,
            "X-Language": language,
            "X-Mode": args.mode,
            "X-Timestamp-Type": args.timestamp_type,
            "X-Session-Id": f"bench-{bucket}-{language}",
            "X-Utterance-Id": utt_id,
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(url, data=pcm, headers=headers, timeout=args.timeout_s)
            http_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            writer.writerow([utt_id, language, bucket, f"{audio_s:.3f}",
                             f"http_error:{exc}", "", "", "", "", ""])
            out_f.flush()
            print(f"[{idx}/{len(rows)}] {utt_id} ERROR {exc}", file=sys.stderr)
            time.sleep(args.sleep_ms / 1000.0)
            continue

        status = f"http_{resp.status_code}"
        worker_audio_s = ""
        worker_proc_ms = ""
        rtf = ""
        hyp_text = ""
        if resp.status_code == 200:
            try:
                body = resp.json()
                hyp_text = (body.get("text") or "").replace("\n", " ").replace("\r", " ")
                metrics = body.get("metrics") or {}
                worker_audio_s = metrics.get("audio_duration", "")
                worker_proc_ms = metrics.get("processing_latency", "")
                if worker_audio_s and float(worker_audio_s) > 0 and worker_proc_ms != "":
                    rtf = f"{(float(worker_proc_ms) / 1000.0) / float(worker_audio_s):.4f}"
            except Exception as exc:
                status = f"parse_error:{exc}"

        writer.writerow([
            utt_id, language, bucket, f"{audio_s:.3f}",
            status, f"{http_ms:.1f}",
            worker_audio_s, worker_proc_ms, rtf, hyp_text,
        ])
        out_f.flush()
        print(f"[{idx}/{len(rows)}] {utt_id} ({bucket}/{language}) "
              f"audio={audio_s:.2f}s http={http_ms:.0f}ms rtf={rtf}", file=sys.stderr)

        time.sleep(args.sleep_ms / 1000.0)

    out_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
