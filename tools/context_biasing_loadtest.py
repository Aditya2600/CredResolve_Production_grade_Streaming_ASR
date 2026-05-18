#!/usr/bin/env python3
"""Operational context-biasing concurrency probe for staging environments.

Run one or more pool-size groups against a worker endpoint. If the same endpoint
is reused between groups, pass --prepare-command so the worker is restarted with
the requested pool/concurrency settings before each group.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import request


BUCKET_RE = re.compile(r'^(?P<name>\w+)_bucket\{le="(?P<le>[^"]+)"\}\s+(?P<value>[-+0-9.eE]+)$')
COUNTER_RE = re.compile(
    r'^asr_worker_context_biasing_fallback_total\{reason="(?P<reason>[^"]+)"\}\s+(?P<value>[-+0-9.eE]+)$'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staging context-biasing concurrency probes.")
    parser.add_argument("--url", default="http://localhost:9000", help="Worker base URL.")
    parser.add_argument("--requests", type=int, default=32, help="Requests per pool-size group.")
    parser.add_argument("--concurrency", type=int, default=8, help="Client-side concurrent requests.")
    parser.add_argument("--pool-sizes", default="1,2,4", help="Comma-separated labels to run.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--audio-seconds", type=float, default=1.0)
    parser.add_argument("--pcm-file", type=Path, help="Optional raw PCM16LE payload.")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--bias-mode", default="shadow", choices=("shadow", "active"))
    parser.add_argument(
        "--biasing-context-json",
        default='{"debtor_name":"Ravi Kumar","lender":"CredResolve"}',
        help="JSON object attached as biasing_context.",
    )
    parser.add_argument(
        "--prepare-command",
        default="",
        help="Optional shell command run before each group; {pool_size} is substituted.",
    )
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def read_payload(args: argparse.Namespace) -> bytes:
    if args.pcm_file:
        return args.pcm_file.read_bytes()
    samples = max(int(args.sample_rate * max(args.audio_seconds, 0.01)), 1)
    return b"\x00\x00" * samples


def fetch_text(url: str, timeout_s: float) -> str:
    with request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def metrics_snapshot(base_url: str, timeout_s: float) -> dict[str, Any]:
    text = fetch_text(f"{base_url.rstrip('/')}/metrics", timeout_s)
    buckets: dict[str, dict[float, float]] = {}
    fallbacks: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = BUCKET_RE.match(line)
        if match:
            le_raw = match.group("le")
            le = math.inf if le_raw == "+Inf" else float(le_raw)
            buckets.setdefault(match.group("name"), {})[le] = float(match.group("value"))
            continue
        match = COUNTER_RE.match(line)
        if match:
            fallbacks[match.group("reason")] = float(match.group("value"))
    return {"buckets": buckets, "fallbacks": fallbacks}


def histogram_delta(before: dict[str, Any], after: dict[str, Any], name: str) -> dict[float, float]:
    before_buckets = before["buckets"].get(name, {})
    after_buckets = after["buckets"].get(name, {})
    return {bucket: after_buckets.get(bucket, 0.0) - before_buckets.get(bucket, 0.0) for bucket in after_buckets}


def approx_quantile(cumulative_buckets: dict[float, float], quantile: float) -> float | None:
    if not cumulative_buckets:
        return None
    total = cumulative_buckets.get(math.inf, max(cumulative_buckets.values(), default=0.0))
    if total <= 0:
        return None
    target = total * quantile
    for upper_bound in sorted(cumulative_buckets):
        if cumulative_buckets[upper_bound] >= target:
            return upper_bound
    return None


def gpu_memory_mib() -> list[int] | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    values = []
    for line in output.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return values or None


def send_request(
    *,
    base_url: str,
    payload: bytes,
    sample_rate: int,
    language: str,
    context_header: str,
    index: int,
    timeout_s: float,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Sample-Rate": str(sample_rate),
        "X-Language": language,
        "X-Session-Id": f"context-loadtest-{index}",
        "X-Utterance-Id": f"context-loadtest-{index}",
        "X-Context-Biasing-Request": context_header,
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/v1/transcribe",
        data=payload,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
            status = response.status
    except Exception as exc:
        return {"status": "error", "error": str(exc), "client_total_ms": (time.perf_counter() - started) * 1000.0}
    context = body.get("context_biasing") or {}
    return {
        "status": status,
        "fallback_reason": context.get("fallback_reason"),
        "bias_latency_ms": context.get("latency_ms"),
        "returned_source": context.get("returned_source"),
        "client_total_ms": (time.perf_counter() - started) * 1000.0,
    }


def summarize_group(pool_size: int, results: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    completed = sum(1 for result in results if result.get("status") == 200 and not result.get("fallback_reason"))
    errors = sum(1 for result in results if result.get("status") == "error")
    response_fallback_reasons: dict[str, int] = {}
    for result in results:
        reason = result.get("fallback_reason")
        if reason:
            response_fallback_reasons[str(reason)] = response_fallback_reasons.get(str(reason), 0) + 1
    metric_fallback_reasons = {
        reason: int(after["fallbacks"].get(reason, 0.0) - before["fallbacks"].get(reason, 0.0))
        for reason in set(before["fallbacks"]) | set(after["fallbacks"])
        if after["fallbacks"].get(reason, 0.0) - before["fallbacks"].get(reason, 0.0) > 0
    }
    queue_timeouts = int(after["fallbacks"].get("queue_timeout", 0.0) - before["fallbacks"].get("queue_timeout", 0.0))
    inference_timeouts = int(
        after["fallbacks"].get("inference_timeout", 0.0) - before["fallbacks"].get("inference_timeout", 0.0)
    )
    queue_hist = histogram_delta(before, after, "asr_worker_context_biasing_queue_wait_ms")
    inference_hist = histogram_delta(before, after, "asr_worker_context_biasing_latency_seconds")
    total_hist = histogram_delta(before, after, "asr_worker_context_biasing_total_latency_seconds")
    client_totals = [float(result["client_total_ms"]) for result in results if "client_total_ms" in result]
    return {
        "pool_size": pool_size,
        "requests": len(results),
        "completed_biased_decodes": completed,
        "queue_timeouts": queue_timeouts,
        "inference_timeouts": inference_timeouts,
        "fallback_rate": round(sum(metric_fallback_reasons.values()) / max(len(results), 1), 4),
        "fallback_reasons": metric_fallback_reasons,
        "response_fallback_reasons": response_fallback_reasons,
        "errors": errors,
        "queue_wait_ms": {"p50": approx_quantile(queue_hist, 0.50), "p95": approx_quantile(queue_hist, 0.95)},
        "inference_latency_s": {
            "p50": approx_quantile(inference_hist, 0.50),
            "p95": approx_quantile(inference_hist, 0.95),
        },
        "total_latency_s": {"p50": approx_quantile(total_hist, 0.50), "p95": approx_quantile(total_hist, 0.95)},
        "client_total_ms": {
            "p50": round(statistics.median(client_totals), 2) if client_totals else None,
            "p95": round(statistics.quantiles(client_totals, n=20)[18], 2) if len(client_totals) >= 20 else None,
        },
        "gpu_memory_mib": gpu_memory_mib(),
    }


def run_group(args: argparse.Namespace, pool_size: int, payload: bytes, context_header: str) -> dict[str, Any]:
    if args.prepare_command:
        env = os.environ.copy()
        env.update(
            {
                "ASR_CONTEXT_BIASING_MODEL_POOL_SIZE": str(pool_size),
                "ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES": str(pool_size),
                "ASR_CONTEXT_BIASING_EXECUTOR_WORKERS": str(pool_size),
            }
        )
        subprocess.run(args.prepare_command.format(pool_size=pool_size), shell=True, check=True, env=env)
        if args.warmup_seconds > 0:
            time.sleep(args.warmup_seconds)
    before = metrics_snapshot(args.url, args.timeout_seconds)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        futures = [
            executor.submit(
                send_request,
                base_url=args.url,
                payload=payload,
                sample_rate=args.sample_rate,
                language=args.language,
                context_header=context_header,
                index=(pool_size * 100000) + index,
                timeout_s=args.timeout_seconds,
            )
            for index in range(max(args.requests, 1))
        ]
        for future in as_completed(futures):
            results.append(future.result())
    after = metrics_snapshot(args.url, args.timeout_seconds)
    return summarize_group(pool_size, results, before, after)


def main() -> None:
    args = parse_args()
    payload = read_payload(args)
    biasing_context = json.loads(args.biasing_context_json)
    if not isinstance(biasing_context, dict):
        raise SystemExit("--biasing-context-json must decode to an object")
    context_header = json.dumps(
        {"context_biasing": {"enabled": True, "mode": args.bias_mode}, "biasing_context": biasing_context},
        separators=(",", ":"),
    )
    pool_sizes = [int(item.strip()) for item in args.pool_sizes.split(",") if item.strip()]
    if len(pool_sizes) > 1 and not args.prepare_command:
        print("warning: --prepare-command not set; pool-size groups are labels only unless the target endpoint changes externally")
    summaries = [run_group(args, pool_size, payload, context_header) for pool_size in pool_sizes]
    print(json.dumps({"results": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
