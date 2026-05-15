#!/usr/bin/env python3
"""Export a Prometheus time window for ASR load-test comparison.

Prometheus keeps the live TSDB; this script copies the run window into ordinary
JSON files under an artifact directory so a benchmark can be reviewed or moved
without depending on the Prometheus volume still being present.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_PROM_URL = "http://localhost:9090"
DEFAULT_STEP_S = 5
DEFAULT_QUERIES = {
    "ws_connections": "sum(asr_ws_connections)",
    "worker_inflight": "sum(asr_worker_inflight_requests)",
    "worker_request_rate_1m": "sum(rate(asr_worker_requests_total[1m]))",
    "worker_latency_p95_1m": (
        "histogram_quantile(0.95, sum(rate(asr_worker_latency_seconds_bucket[1m])) by (le))"
    ),
    "worker_inference_p95_1m": (
        "histogram_quantile(0.95, sum(rate(asr_worker_inference_seconds_bucket[1m])) by (le))"
    ),
    "worker_triton_roundtrip_p95_1m": (
        "histogram_quantile(0.95, "
        "sum(rate(asr_worker_triton_infer_seconds_bucket[1m])) by (le))"
    ),
    "e2e_latency_p95_1m": (
        "histogram_quantile(0.95, sum(rate(asr_e2e_seconds_bucket[1m])) by (le))"
    ),
    "worker_fallback_increase_5m": "sum(increase(asr_worker_fallback_total[5m]))",
    "gpu_utilization": 'max by (gpu) (DCGM_FI_DEV_GPU_UTIL{job="gpu"})',
    "gpu_vram_used_mib": 'max by (gpu) (DCGM_FI_DEV_FB_USED{job="gpu"})',
    "triton_request_rate_1m": 'sum(rate(nv_inference_request_success{job="triton"}[1m]))',
    "triton_avg_queue_us_1m": (
        'sum(rate(nv_inference_queue_duration_us{job="triton"}[1m])) '
        '/ clamp_min(sum(rate(nv_inference_request_success{job="triton"}[1m])), 1)'
    ),
    "triton_avg_compute_infer_us_1m": (
        'sum(rate(nv_inference_compute_infer_duration_us{job="triton"}[1m])) '
        '/ clamp_min(sum(rate(nv_inference_request_success{job="triton"}[1m])), 1)'
    ),
}


def parse_timestamp(value: str) -> float:
    raw = value.strip()
    try:
        return float(raw)
    except ValueError:
        pass

    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def iso_utc(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def query_range(prom_url: str, query: str, start_s: float, end_s: float, step_s: int) -> dict:
    params = urlencode(
        {
            "query": query,
            "start": f"{start_s:.3f}",
            "end": f"{end_s:.3f}",
            "step": step_s,
        }
    )
    with urlopen(f"{prom_url.rstrip('/')}/api/v1/query_range?{params}", timeout=30) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed for {query!r}: {payload}")
    return payload


def parse_query_overrides(values: list[str]) -> dict[str, str]:
    queries = dict(DEFAULT_QUERIES)
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--query must be NAME=EXPR, got {value!r}")
        name, expr = value.split("=", 1)
        name = name.strip()
        expr = expr.strip()
        if not name or not expr:
            raise SystemExit(f"--query must be NAME=EXPR, got {value!r}")
        queries[name] = expr
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC ISO-8601 timestamp or Unix epoch seconds.")
    parser.add_argument("--end", required=True, help="UTC ISO-8601 timestamp or Unix epoch seconds.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prom-url", default=DEFAULT_PROM_URL)
    parser.add_argument("--step-seconds", type=int, default=DEFAULT_STEP_S)
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        metavar="NAME=EXPR",
        help="Add or replace one exported PromQL query.",
    )
    args = parser.parse_args()

    start_s = parse_timestamp(args.start)
    end_s = parse_timestamp(args.end)
    if end_s <= start_s:
        raise SystemExit("--end must be later than --start")
    if args.step_seconds <= 0:
        raise SystemExit("--step-seconds must be > 0")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = parse_query_overrides(args.query)

    metadata = {
        "prom_url": args.prom_url,
        "start": iso_utc(start_s),
        "end": iso_utc(end_s),
        "step_seconds": args.step_seconds,
        "queries": queries,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for name, query in queries.items():
        payload = query_range(args.prom_url, query, start_s, end_s, args.step_seconds)
        (out_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        series_count = len(payload.get("data", {}).get("result", []))
        print(f"{name}: {series_count} series")

    print(f"wrote {len(queries)} query exports to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
