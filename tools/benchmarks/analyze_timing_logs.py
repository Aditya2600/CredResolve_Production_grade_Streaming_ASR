"""
Parse Triton timing logs and aggregate by duration bucket.

Usage:
    docker compose ... logs --since 10m triton 2>&1 \
        | grep "timing audio=" \
        | python tools/benchmarks/analyze_timing_logs.py
"""

import math
import re
import sys
import statistics
from collections import defaultdict

LINE_RE = re.compile(
    r"timing audio=(?P<audio>\d+(?:\.\d+)?)s "
    r"frames=(?P<frames>\d+) "
    r"tokens=(?P<tokens>\d+) "
    r"lang=(?P<lang>\w+) "
    r"total=(?P<total>\d+(?:\.\d+)?)ms"
)

STAGE_RE = re.compile(
    r"(?P<name>preproc|encoder|decode|postproc)="
    r"(?P<ms>\d+(?:\.\d+)?)ms"
)

REQUIRED_STAGES = {"preproc", "encoder", "decode", "postproc"}

BUCKETS = [
    ("0-1s", 0.0, 1.0),
    ("1-3s", 1.0, 3.0),
    ("3-8s", 3.0, 8.0),
    ("8-20s", 8.0, 20.0),
    (">20s", 20.0, float("inf")),
]


def bucket_for(audio_sec):
    for name, lo, hi in BUCKETS:
        if lo <= audio_sec < hi:
            return name
    return None


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None

    idx = math.ceil((p / 100.0) * len(values)) - 1
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


def stat(values):
    values = sorted(values)
    return statistics.median(values), percentile(values, 95)


def parse_line(line):
    m = LINE_RE.search(line)
    if not m:
        return None

    audio = float(m.group("audio"))
    total = float(m.group("total"))

    stages = {}
    for sm in STAGE_RE.finditer(line):
        stages[sm.group("name")] = float(sm.group("ms"))

    # Reject partial or malformed timing lines.
    if set(stages.keys()) != REQUIRED_STAGES:
        return None

    # Reject suspicious records where stage sum does not match total.
    stage_sum = sum(stages.values())
    if total <= 0:
        return None

    drift = abs(stage_sum - total) / total
    if drift > 0.10:
        return None

    return {
        "audio": audio,
        "frames": int(m.group("frames")),
        "tokens": int(m.group("tokens")),
        "lang": m.group("lang"),
        "total": total,
        "stages": stages,
    }


def pct_of_total(record, stage):
    if record["total"] == 0:
        return 0.0
    return 100.0 * record["stages"].get(stage, 0.0) / record["total"]


def summarize(records, label):
    if not records:
        print(f"{label}: no data")
        return

    n = len(records)

    total_med, total_p95 = stat([r["total"] for r in records])

    # RTF = processing_time / audio_duration
    rtf_med, rtf_p95 = stat([
        r["total"] / 1000.0 / r["audio"]
        for r in records
        if r["audio"] > 0
    ])

    ms_per_audio_sec_med, ms_per_audio_sec_p95 = stat([
        r["total"] / r["audio"]
        for r in records
        if r["audio"] > 0
    ])

    print(f"\n=== {label} (n={n}) ===")
    print(f"  total_ms               median={total_med:7.1f}  p95={total_p95:7.1f}")
    print(f"  RTF                    median={rtf_med:7.3f}x p95={rtf_p95:7.3f}x")
    print(f"  processing_ms/audio_s  median={ms_per_audio_sec_med:7.1f}  p95={ms_per_audio_sec_p95:7.1f}")
    print("  stage breakdown (median %, p95 %):")

    for stage in ["preproc", "encoder", "decode", "postproc"]:
        pcts = [pct_of_total(r, stage) for r in records]
        med, p95 = stat(pcts)
        bar = "█" * int(med / 2)
        print(f"    {stage:9} {med:5.1f}%  p95={p95:5.1f}%  {bar}")


def main():
    records = []
    skipped = 0

    for line in sys.stdin:
        rec = parse_line(line)
        if rec is not None:
            records.append(rec)
        elif "timing audio=" in line:
            skipped += 1

    if not records:
        print("No valid timing lines parsed. Check log format.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(records)} timing records.")
    if skipped:
        print(f"Skipped {skipped} malformed timing lines.")

    by_bucket = defaultdict(list)
    for r in records:
        b = bucket_for(r["audio"])
        if b:
            by_bucket[b].append(r)

    for name, _lo, _hi in BUCKETS:
        summarize(by_bucket.get(name, []), name)

    print("\n=== VERDICT ===")
    for name, _lo, _hi in BUCKETS:
        recs = by_bucket.get(name, [])
        if not recs:
            continue

        encoder_med = statistics.median([
            pct_of_total(r, "encoder")
            for r in recs
        ])

        decode_med = statistics.median([
            pct_of_total(r, "decode")
            for r in recs
        ])

        if encoder_med >= 60:
            verdict = "✓ GO  - encoder dominates, RNNT encoder BLS/TRT is worth building"
        elif encoder_med >= 45:
            verdict = "~ MEH - encoder still matters, but decode is becoming significant"
        else:
            verdict = "✗ STOP - decode loop is likely the bottleneck"

        print(
            f"  {name:8} "
            f"encoder_med={encoder_med:5.1f}%  "
            f"decode_med={decode_med:5.1f}%  "
            f"→ {verdict}"
        )


if __name__ == "__main__":
    main()
