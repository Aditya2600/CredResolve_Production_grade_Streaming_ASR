#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional for audio probing
    sf = None


TEXT_KEY_CANDIDATES = ("text", "reference", "normalized_text", "transcript", "sentence")
AUDIO_KEY_CANDIDATES = ("audio_filepath", "audio_path", "audio", "path")
DURATION_KEY_CANDIDATES = ("duration", "audio_duration", "audio_duration_sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a JSONL speech manifest for empty text, missing/corrupt audio, "
            "duration outliers, transcript-length outliers, suspicious text/audio mismatches, "
            "and optional duplicates."
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True, help="Input JSONL manifest.")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Cleaned output JSONL manifest.")
    parser.add_argument("--reject-manifest", type=Path, help="Optional JSONL file with rejected rows and reasons.")
    parser.add_argument("--summary-json", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--text-key", help="Transcript field name. Auto-detected by default.")
    parser.add_argument("--audio-key", help="Audio path field name. Auto-detected by default.")
    parser.add_argument("--duration-key", help="Duration field name. Auto-detected by default.")
    parser.add_argument(
        "--probe-audio",
        action="store_true",
        help="Open audio files with soundfile to validate them and fill missing durations.",
    )
    parser.add_argument(
        "--output-schema",
        choices=("same", "nemo"),
        default="same",
        help="Write rows in the original schema or normalized NeMo schema. Default: same",
    )
    parser.add_argument("--min-duration", type=float, default=0.0, help="Reject rows shorter than this many seconds.")
    parser.add_argument("--max-duration", type=float, help="Reject rows longer than this many seconds.")
    parser.add_argument("--max-words", type=int, help="Reject rows with more than this many words.")
    parser.add_argument("--max-chars", type=int, help="Reject rows with more than this many characters.")
    parser.add_argument(
        "--mismatch-min-duration",
        type=float,
        default=4.0,
        help="Only apply low-words-per-second mismatch checks above this duration. Default: 4.0",
    )
    parser.add_argument(
        "--min-words-per-sec",
        type=float,
        default=0.6,
        help="Reject long rows below this transcript-density threshold. Default: 0.6",
    )
    parser.add_argument(
        "--max-words-per-sec",
        type=float,
        default=4.5,
        help="Reject rows above this transcript-density threshold. Default: 4.5",
    )
    parser.add_argument(
        "--drop-unintelligible",
        action="store_true",
        help="Reject rows whose normalized transcript contains <unintelligible>.",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=("none", "audio", "text", "audio_text"),
        default="none",
        help="Optional deduplication strategy. Default: none",
    )
    parser.add_argument(
        "--duration-buckets",
        help=(
            "Optional comma-separated upper bounds used to emit duration-sliced manifests "
            "for manual curriculum or bucketing, for example: 2,4,6,8"
        ),
    )
    parser.add_argument(
        "--bucket-output-dir",
        type=Path,
        help="Directory for optional duration-sliced manifests created by --duration-buckets.",
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
    raise SystemExit(f"could not auto-detect key; tried {', '.join(candidates)}")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def summarize_durations(rows: list[dict[str, Any]], duration_key: str) -> dict[str, float | int | None]:
    values = [float(row[duration_key]) for row in rows if isinstance(row.get(duration_key), (int, float))]
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "max": round(max(values), 4),
    }


def parse_duration_buckets(raw: str | None) -> list[float]:
    if not raw:
        return []
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if any(value <= 0 for value in values):
        raise SystemExit("duration buckets must be positive")
    return values


def bucket_name(lower: float | None, upper: float | None) -> str:
    if lower is None and upper is not None:
        return f"duration_le_{upper:g}s"
    if lower is not None and upper is not None:
        return f"duration_{lower:g}s_to_{upper:g}s"
    if lower is not None:
        return f"duration_gt_{lower:g}s"
    return "duration_all"


def main() -> int:
    args = parse_args()

    input_manifest = args.input_manifest.expanduser().resolve()
    rows = load_rows(input_manifest)
    text_key = detect_key(TEXT_KEY_CANDIDATES, rows, args.text_key)
    audio_key = detect_key(AUDIO_KEY_CANDIDATES, rows, args.audio_key)
    duration_key = detect_key(DURATION_KEY_CANDIDATES, rows, args.duration_key)
    duration_buckets = parse_duration_buckets(args.duration_buckets)

    if args.probe_audio and sf is None:
        raise SystemExit("soundfile is required for --probe-audio")
    if duration_buckets and args.bucket_output_dir is None:
        raise SystemExit("--bucket-output-dir is required when --duration-buckets is used")

    reject_counts: Counter[str] = Counter()
    duplicate_keys: set[str] = set()
    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for lineno, row in rows:
        raw_row = dict(row)
        text = normalize_text(raw_row.get(text_key, ""))
        audio_path = Path(str(raw_row.get(audio_key, "") or "")).expanduser()
        audio_text = str(audio_path)

        duration = None
        raw_duration = raw_row.get(duration_key)
        if isinstance(raw_duration, (int, float)) and float(raw_duration) > 0:
            duration = float(raw_duration)

        reason = None
        if not text:
            reason = "empty_text"
        elif args.drop_unintelligible and "<unintelligible>" in text.casefold():
            reason = "unintelligible_text"
        elif not audio_text:
            reason = "missing_audio_path"
        elif args.probe_audio and not audio_path.exists():
            reason = "missing_audio_file"
        elif args.probe_audio:
            try:
                info = sf.info(str(audio_path))
            except Exception:
                reason = "corrupt_audio"
            else:
                if duration is None:
                    duration = float(info.duration)
        if reason is None and duration is None:
            reason = "missing_duration"
        if reason is None and duration is not None and duration < float(args.min_duration or 0.0):
            reason = "too_short"
        if reason is None and duration is not None and args.max_duration is not None and duration > float(args.max_duration):
            reason = "too_long"

        words = len(text.split())
        chars = len(text)
        words_per_sec = (words / duration) if duration and duration > 0 else None

        if reason is None and args.max_words is not None and words > int(args.max_words):
            reason = "too_many_words"
        if reason is None and args.max_chars is not None and chars > int(args.max_chars):
            reason = "too_many_chars"
        if reason is None and words_per_sec is not None and words_per_sec > float(args.max_words_per_sec):
            reason = "mismatch_high_words_per_sec"
        if (
            reason is None
            and duration is not None
            and duration >= float(args.mismatch_min_duration)
            and words_per_sec is not None
            and words_per_sec < float(args.min_words_per_sec)
        ):
            reason = "mismatch_low_words_per_sec"

        dedupe_key = None
        if reason is None and args.dedupe_mode != "none":
            if args.dedupe_mode == "audio":
                dedupe_key = f"audio::{audio_text}"
            elif args.dedupe_mode == "text":
                dedupe_key = f"text::{text.casefold()}"
            else:
                dedupe_key = f"audio_text::{audio_text}::{text.casefold()}"
            if dedupe_key in duplicate_keys:
                reason = "duplicate_row"

        output_row = dict(raw_row)
        if args.output_schema == "nemo":
            output_row["audio_filepath"] = audio_text
            output_row["text"] = text
            output_row["duration"] = float(duration) if duration is not None else None
        else:
            output_row[text_key] = text
            output_row[audio_key] = audio_text
            output_row[duration_key] = float(duration) if duration is not None else None

        if reason is not None:
            reject_counts[reason] += 1
            output_row["reject_reason"] = reason
            output_row["source_lineno"] = lineno
            rejected_rows.append(output_row)
            continue

        if dedupe_key is not None:
            duplicate_keys.add(dedupe_key)

        output_row["source_lineno"] = lineno
        kept_rows.append(output_row)

    output_manifest = args.output_manifest.expanduser().resolve()
    write_jsonl(output_manifest, kept_rows)

    if args.reject_manifest:
        write_jsonl(args.reject_manifest.expanduser().resolve(), rejected_rows)

    bucket_summary: dict[str, int] = {}
    if duration_buckets:
        bucket_dir = args.bucket_output_dir.expanduser().resolve()
        bucket_dir.mkdir(parents=True, exist_ok=True)
        previous = None
        remaining_rows = kept_rows
        for upper in duration_buckets:
            bucket_rows = [
                row
                for row in remaining_rows
                if isinstance(row.get("duration" if args.output_schema == "nemo" else duration_key), (int, float))
                and float(row.get("duration" if args.output_schema == "nemo" else duration_key)) <= upper
                and (previous is None or float(row.get("duration" if args.output_schema == "nemo" else duration_key)) > previous)
            ]
            name = bucket_name(previous, upper)
            write_jsonl(bucket_dir / f"{name}.jsonl", bucket_rows)
            bucket_summary[name] = len(bucket_rows)
            previous = upper
        tail_rows = [
            row
            for row in kept_rows
            if isinstance(row.get("duration" if args.output_schema == "nemo" else duration_key), (int, float))
            and float(row.get("duration" if args.output_schema == "nemo" else duration_key)) > duration_buckets[-1]
        ]
        tail_name = bucket_name(duration_buckets[-1], None)
        write_jsonl(bucket_dir / f"{tail_name}.jsonl", tail_rows)
        bucket_summary[tail_name] = len(tail_rows)

    summary = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "output_schema": args.output_schema,
        "detected_keys": {
            "text_key": text_key,
            "audio_key": audio_key,
            "duration_key": duration_key,
        },
        "rows_in": len(rows),
        "rows_kept": len(kept_rows),
        "rows_rejected": len(rejected_rows),
        "reject_counts": dict(reject_counts),
        "duration_stats_kept": summarize_durations(
            kept_rows,
            "duration" if args.output_schema == "nemo" else duration_key,
        ),
        "filters": {
            "probe_audio": args.probe_audio,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "max_words": args.max_words,
            "max_chars": args.max_chars,
            "mismatch_min_duration": args.mismatch_min_duration,
            "min_words_per_sec": args.min_words_per_sec,
            "max_words_per_sec": args.max_words_per_sec,
            "drop_unintelligible": args.drop_unintelligible,
            "dedupe_mode": args.dedupe_mode,
        },
        "duration_bucket_counts": bucket_summary,
    }

    if args.summary_json:
        summary_path = args.summary_json.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
