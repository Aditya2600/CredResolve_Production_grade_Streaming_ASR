#!/usr/bin/env python3
"""Regenerate audio_bench baseline outputs.

Run this after a deliberate algorithm change in audio_processing.py (or in
the bench harness) when you want a fresh baseline. It is NOT run in CI; the
schema test only validates the JSON shape.

Defaults:
    iterations per fixture : 20
    baseline output        : tests/fixtures/audio_bench/baseline/
    STT                    : enabled (uses worker.app.main_v2.build_worker_model)
    VAD + denoise          : both enabled (production defaults)

Pass extra flags through to tools/benchmarks/audio_bench.py via ``--`` if needed,
e.g. ``--iterations 40``.

Update tests/fixtures/audio_bench/baseline/README.md afterwards with the
date / machine / model version captured.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WAV_DIR = Path(__file__).resolve().parent
BASELINE_DIR = WAV_DIR / "baseline"
BENCH_SCRIPT = REPO_ROOT / "tools" / "benchmarks" / "audio_bench.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--no-stt",
        action="store_true",
        help="Skip STT (latency-only baseline, leaves WER fields null).",
    )
    parser.add_argument(
        "--baseline-out",
        type=Path,
        default=BASELINE_DIR,
        help=(
            "Directory for baseline.csv, baseline.json, audio_bench_results.csv, "
            "and audio_bench_summary.txt. Defaults to tests/fixtures/audio_bench/baseline."
        ),
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to tools/benchmarks/audio_bench.py (place after --).",
    )
    args = parser.parse_args()

    if not BENCH_SCRIPT.exists():
        print(f"error: {BENCH_SCRIPT} not found", file=sys.stderr)
        return 2

    baseline_dir = args.baseline_out
    if not baseline_dir.is_absolute():
        baseline_dir = REPO_ROOT / baseline_dir
    baseline_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        str(WAV_DIR),
        "--iterations",
        str(args.iterations),
        "--baseline-out",
        str(baseline_dir),
        "--out-dir",
        str(baseline_dir),
    ]
    if args.no_stt:
        cmd.append("--no-stt")
    forward = [a for a in args.extra if a != "--"]
    cmd.extend(forward)

    print("Regenerating baseline:", " ".join(cmd))
    env = os.environ.copy()
    return subprocess.call(cmd, env=env, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    sys.exit(main())
