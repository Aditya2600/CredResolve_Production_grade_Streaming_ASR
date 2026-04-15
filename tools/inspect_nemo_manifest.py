#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional for read-only inspection
    sf = None


TEXT_KEY_CANDIDATES = ("text", "reference", "normalized_text", "transcript", "sentence")
AUDIO_KEY_CANDIDATES = ("audio_filepath", "audio_path", "audio", "path")
DURATION_KEY_CANDIDATES = ("duration", "audio_duration", "audio_duration_sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a NeMo-style JSONL manifest for duration, transcript-length, "
            "duplicate, and suspicious text/audio mismatch outliers."
        )
    )
    parser.add_argument("manifest", type=Path, help="Input JSONL manifest.")
    parser.add_argument("--text-key", help="Transcript field name. Auto-detected by default.")
    parser.add_argument("--audio-key", help="Audio path field name. Auto-detected by default.")
    parser.add_argument("--duration-key", help="Duration field name. Auto-detected by default.")
    parser.add_argument(
        "--probe-audio",
        action="store_true",
        help="Open audio files with soundfile to validate them and fill missing durations.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many longest and suspicious samples to print. Default: 10",
    )
    parser.add_argument(
        "--max-words-per-sec",
        type=float,
        default=4.5,
        help="Rows above this ratio are flagged as suspicious. Default: 4.5",
    )
    parser.add_argument(
        "--min-words-per-sec",
        type=float,
        default=0.6,
        help="Rows below this ratio are flagged as suspicious when long enough. Default: 0.6",
    )
    parser.add_argument(
        "--suspicious-min-duration",
        type=float,
        default=4.0,
        help="Minimum duration for low-words-per-second mismatch checks. Default: 4.0",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary as JSON instead of a human-readable report.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{lineno}: expected a JSON object")
        rows.append((lineno, row))
    if not rows:
        raise SystemExit(f"{path}: manifest is empty")
    return rows


def detect_key(candidates: tuple[str, ...], rows: list[tuple[int, dict[str, Any]]], explicit: str | None) -> str:
    if explicit:
        return explicit
    sample = rows[0][1]
    for key in candidates:
        if key in sample:
            return key
    if not sample:
        raise SystemExit("cannot auto-detect key from an empty sample row")
    raise SystemExit(f"could not auto-detect key; tried {', '.join(candidates)}")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).strip()


def quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = int(round((len(ordered) - 1) * percentile))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def round_summary(payload: dict[str, float | None]) -> dict[str, float | int | None]:
    rounded: dict[str, float | int | None] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        else:
            rounded[key] = value
    return rounded


def main() -> int:
    args = parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    rows = load_rows(manifest_path)
    text_key = detect_key(TEXT_KEY_CANDIDATES, rows, args.text_key)
    audio_key = detect_key(AUDIO_KEY_CANDIDATES, rows, args.audio_key)
    duration_key = detect_key(DURATION_KEY_CANDIDATES, rows, args.duration_key)

    if args.probe_audio and sf is None:
        raise SystemExit("soundfile is required for --probe-audio")

    duration_values: list[float] = []
    word_values: list[float] = []
    char_values: list[float] = []
    words_per_sec_values: list[float] = []

    invalid_counts: Counter[str] = Counter()
    duplicate_audio: Counter[str] = Counter()
    duplicate_text: Counter[str] = Counter()
    longest_rows: list[dict[str, Any]] = []
    suspicious_rows: list[dict[str, Any]] = []

    for lineno, row in rows:
        text = normalize_text(row.get(text_key, ""))
        audio_path = Path(str(row.get(audio_key, "") or "")).expanduser()
        audio_text = str(audio_path)
        if audio_text:
            duplicate_audio[audio_text] += 1
        if text:
            duplicate_text[text] += 1

        if not text:
            invalid_counts["empty_text"] += 1

        duration = None
        raw_duration = row.get(duration_key)
        if isinstance(raw_duration, (int, float)) and float(raw_duration) > 0:
            duration = float(raw_duration)

        if args.probe_audio:
            if not audio_text:
                invalid_counts["missing_audio_path"] += 1
            elif not audio_path.exists():
                invalid_counts["missing_audio_file"] += 1
            else:
                try:
                    info = sf.info(str(audio_path))
                except Exception:
                    invalid_counts["corrupt_audio"] += 1
                else:
                    if duration is None:
                        duration = float(info.duration)
        elif not audio_text:
            invalid_counts["missing_audio_path"] += 1

        if duration is None:
            invalid_counts["missing_duration"] += 1
            continue

        words = len(text.split())
        chars = len(text)
        words_per_sec = words / duration if duration > 0 else math.inf

        duration_values.append(duration)
        word_values.append(float(words))
        char_values.append(float(chars))
        if math.isfinite(words_per_sec):
            words_per_sec_values.append(words_per_sec)

        sample_info = {
            "lineno": lineno,
            "duration": round(duration, 4),
            "words": words,
            "chars": chars,
            "words_per_sec": round(words_per_sec, 4) if math.isfinite(words_per_sec) else None,
            "audio_path": audio_text,
            "text": text,
        }
        longest_rows.append(sample_info)

        mismatch_high = math.isfinite(words_per_sec) and words_per_sec > args.max_words_per_sec
        mismatch_low = duration >= args.suspicious_min_duration and math.isfinite(words_per_sec) and words_per_sec < args.min_words_per_sec
        too_many_words = words >= 40
        too_many_chars = chars >= 220
        if mismatch_high or mismatch_low or too_many_words or too_many_chars:
            suspicious_rows.append(sample_info)

    longest_rows.sort(key=lambda item: item["duration"], reverse=True)
    suspicious_rows.sort(key=lambda item: (item["duration"], item["words"]), reverse=True)

    summary = {
        "manifest": str(manifest_path),
        "detected_keys": {
            "text_key": text_key,
            "audio_key": audio_key,
            "duration_key": duration_key,
        },
        "row_count": len(rows),
        "invalid_counts": dict(invalid_counts),
        "duplicate_audio_paths": {
            "rows_with_duplicate_audio_path": sum(count for count in duplicate_audio.values() if count > 1),
            "unique_duplicate_audio_paths": sum(1 for count in duplicate_audio.values() if count > 1),
        },
        "duplicate_texts": {
            "rows_with_duplicate_text": sum(count for count in duplicate_text.values() if count > 1),
            "unique_duplicate_texts": sum(1 for count in duplicate_text.values() if count > 1),
        },
        "duration_seconds": round_summary(summarize(duration_values)),
        "transcript_words": round_summary(summarize(word_values)),
        "transcript_chars": round_summary(summarize(char_values)),
        "words_per_sec": round_summary(summarize(words_per_sec_values)),
        "threshold_counts": {
            "duration_lt_0_3": sum(1 for value in duration_values if value < 0.3),
            "duration_lt_0_4": sum(1 for value in duration_values if value < 0.4),
            "duration_lt_0_5": sum(1 for value in duration_values if value < 0.5),
            "duration_gt_6": sum(1 for value in duration_values if value > 6.0),
            "duration_gt_8": sum(1 for value in duration_values if value > 8.0),
            "duration_gt_10": sum(1 for value in duration_values if value > 10.0),
            "duration_gt_12": sum(1 for value in duration_values if value > 12.0),
            "words_gt_20": sum(1 for value in word_values if value > 20),
            "words_gt_30": sum(1 for value in word_values if value > 30),
            "words_gt_40": sum(1 for value in word_values if value > 40),
            "words_gt_50": sum(1 for value in word_values if value > 50),
        },
        "longest_rows": longest_rows[: max(args.top_n, 0)],
        "suspicious_rows": suspicious_rows[: max(args.top_n, 0)],
    }

    if args.json:
        json.dump(summary, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print(f"Manifest: {manifest_path}")
    print(f"Detected keys: text={text_key} audio={audio_key} duration={duration_key}")
    print(f"Rows: {len(rows)}")
    print("Invalid rows: " + json.dumps(summary["invalid_counts"], ensure_ascii=False, sort_keys=True))
    print("Duplicate audio paths: " + json.dumps(summary["duplicate_audio_paths"], ensure_ascii=False, sort_keys=True))
    print("Duplicate texts: " + json.dumps(summary["duplicate_texts"], ensure_ascii=False, sort_keys=True))
    print("Duration stats (s): " + json.dumps(summary["duration_seconds"], ensure_ascii=False, sort_keys=True))
    print("Transcript stats (words): " + json.dumps(summary["transcript_words"], ensure_ascii=False, sort_keys=True))
    print("Transcript stats (chars): " + json.dumps(summary["transcript_chars"], ensure_ascii=False, sort_keys=True))
    print("Words/sec stats: " + json.dumps(summary["words_per_sec"], ensure_ascii=False, sort_keys=True))
    print("Threshold counts: " + json.dumps(summary["threshold_counts"], ensure_ascii=False, sort_keys=True))

    if summary["longest_rows"]:
        print("\nLongest rows:")
        for item in summary["longest_rows"]:
            print(json.dumps(item, ensure_ascii=False))

    if summary["suspicious_rows"]:
        print("\nSuspicious rows:")
        for item in summary["suspicious_rows"]:
            print(json.dumps(item, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
