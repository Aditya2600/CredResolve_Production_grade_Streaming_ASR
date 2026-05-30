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
except ModuleNotFoundError:  # pragma: no cover - reported cleanly by main
    sf = None


TEXT_KEY_CANDIDATES = ("text", "reference", "normalized_text", "transcript", "sentence")
AUDIO_KEY_CANDIDATES = ("audio_filepath", "audio_path", "audio", "path")
DURATION_KEY_CANDIDATES = ("duration", "audio_duration", "audio_duration_sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate NeMo JSONL manifest audio/transcript pairs, write a filtered "
            "manifest, rejected rows with reasons, and a JSON summary."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input NeMo JSONL manifest.")
    parser.add_argument("--output", type=Path, required=True, help="Filtered output JSONL manifest.")
    parser.add_argument("--rejects", type=Path, required=True, help="Rejected rows JSONL manifest.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Validation summary JSON.")
    parser.add_argument("--text-key", help="Transcript field name. Auto-detected per row by default.")
    parser.add_argument("--audio-key", help="Audio path field name. Auto-detected per row by default.")
    parser.add_argument("--duration-key", help="Duration field name. Auto-detected per row by default.")
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=20.0)
    parser.add_argument("--max-chars-per-sec", type=float, default=30.0)
    parser.add_argument("--max-words-per-sec", type=float, default=4.5)
    parser.add_argument("--expected-sample-rate", type=int, default=16000)
    parser.add_argument(
        "--require-mono",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require one audio channel. Use --no-require-mono to allow stereo/multichannel.",
    )
    parser.add_argument(
        "--max-duration-delta-sec",
        type=float,
        default=0.25,
        help="Reject rows when manifest and real duration differ by more than this. Default: 0.25.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def detect_key(row: dict[str, Any], explicit: str | None, candidates: tuple[str, ...]) -> str | None:
    if explicit:
        return explicit
    for key in candidates:
        if key in row:
            return key
    return None


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).strip()


def parse_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def resolve_audio_path(value: Any, manifest_path: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def probe_audio(path: Path) -> dict[str, Any]:
    if sf is None:
        raise RuntimeError("soundfile is required")
    with sf.SoundFile(str(path)) as handle:
        frames = int(handle.frames)
        samplerate = int(handle.samplerate)
        channels = int(handle.channels)
        read_frames = min(frames, 1024)
        if read_frames > 0:
            handle.read(frames=read_frames, dtype="float32", always_2d=True)
    duration = (frames / float(samplerate)) if samplerate > 0 else 0.0
    return {
        "frames": frames,
        "sample_rate": samplerate,
        "channels": channels,
        "duration": duration,
    }


def quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "median": round(float(quantile(values, 0.5) or 0.0), 6),
        "p95": round(float(quantile(values, 0.95) or 0.0), 6),
        "p99": round(float(quantile(values, 0.99) or 0.0), 6),
        "max": round(max(values), 6),
    }


def reject_payload(
    row: dict[str, Any],
    *,
    line_num: int,
    reasons: list[str],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(row)
    payload["validation_source_lineno"] = int(line_num)
    payload["reject_reasons"] = reasons
    if metrics:
        payload["validation_metrics"] = metrics
    return payload


def validate_manifest_file(
    input_path: Path,
    output_path: Path,
    rejects_path: Path,
    summary_path: Path,
    *,
    text_key: str | None = None,
    audio_key: str | None = None,
    duration_key: str | None = None,
    min_duration_sec: float = 1.0,
    max_duration_sec: float = 20.0,
    max_chars_per_sec: float = 30.0,
    max_words_per_sec: float = 4.5,
    expected_sample_rate: int = 16000,
    require_mono: bool = True,
    max_duration_delta_sec: float = 0.25,
    progress_every: int = 1000,
) -> dict[str, Any]:
    if sf is None:
        raise SystemExit("soundfile is required: pip install soundfile")

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    rejects_path = rejects_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    rejected = 0
    reject_counts: Counter[str] = Counter()
    sample_rates: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    duration_values: list[float] = []
    chars_per_sec_values: list[float] = []
    words_per_sec_values: list[float] = []
    reject_examples: list[dict[str, Any]] = []

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
        rejects_path.open("w", encoding="utf-8") as frej,
    ):
        for line_num, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            if progress_every > 0 and total % progress_every == 0:
                print(f"validated {total} rows", file=sys.stderr, flush=True)

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                reasons = ["invalid_json"]
                reject_counts.update(reasons)
                rejected += 1
                payload = {
                    "validation_source_lineno": line_num,
                    "reject_reasons": reasons,
                    "reject_detail": str(exc),
                    "raw_line": raw_line.rstrip("\n"),
                }
                frej.write(json.dumps(payload, ensure_ascii=False) + "\n")
                if len(reject_examples) < 20:
                    reject_examples.append(payload)
                continue

            if not isinstance(row, dict):
                reasons = ["non_object_row"]
                reject_counts.update(reasons)
                rejected += 1
                payload = {
                    "validation_source_lineno": line_num,
                    "reject_reasons": reasons,
                    "row": row,
                }
                frej.write(json.dumps(payload, ensure_ascii=False) + "\n")
                if len(reject_examples) < 20:
                    reject_examples.append(payload)
                continue

            reasons: list[str] = []
            metrics: dict[str, Any] = {}

            resolved_text_key = detect_key(row, text_key, TEXT_KEY_CANDIDATES)
            text = normalize_text(row.get(resolved_text_key)) if resolved_text_key else ""
            if not text:
                reasons.append("empty_transcript")

            resolved_audio_key = detect_key(row, audio_key, AUDIO_KEY_CANDIDATES)
            audio_path = resolve_audio_path(row.get(resolved_audio_key), input_path) if resolved_audio_key else None
            if audio_path is None:
                reasons.append("missing_audio_path")
            elif not audio_path.exists():
                reasons.append("missing_audio_file")
                metrics["audio_filepath"] = str(audio_path)
            else:
                metrics["audio_filepath"] = str(audio_path)
                try:
                    audio_info = probe_audio(audio_path)
                except Exception as exc:
                    reasons.append("unreadable_audio")
                    metrics["audio_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    sample_rate = int(audio_info["sample_rate"])
                    channel_count = int(audio_info["channels"])
                    real_duration = float(audio_info["duration"])
                    sample_rates[str(sample_rate)] += 1
                    channels[str(channel_count)] += 1
                    metrics.update(
                        {
                            "real_duration": round(real_duration, 6),
                            "sample_rate": sample_rate,
                            "channels": channel_count,
                            "frames": int(audio_info["frames"]),
                        }
                    )
                    duration_values.append(real_duration)

                    if real_duration < float(min_duration_sec):
                        reasons.append("too_short_audio")
                    if real_duration > float(max_duration_sec):
                        reasons.append("too_long_audio")
                    if expected_sample_rate > 0 and sample_rate != int(expected_sample_rate):
                        reasons.append("wrong_sample_rate")
                    if require_mono and channel_count != 1:
                        reasons.append("non_mono_audio")

                    if real_duration > 0:
                        chars_per_sec = len(text) / real_duration
                        words_per_sec = len(text.split()) / real_duration
                    else:
                        chars_per_sec = math.inf
                        words_per_sec = math.inf
                    metrics["chars_per_sec"] = (
                        round(chars_per_sec, 6) if math.isfinite(chars_per_sec) else None
                    )
                    metrics["words_per_sec"] = (
                        round(words_per_sec, 6) if math.isfinite(words_per_sec) else None
                    )
                    if math.isfinite(chars_per_sec):
                        chars_per_sec_values.append(chars_per_sec)
                    if math.isfinite(words_per_sec):
                        words_per_sec_values.append(words_per_sec)
                    if chars_per_sec > float(max_chars_per_sec):
                        reasons.append("high_chars_per_sec")
                    if words_per_sec > float(max_words_per_sec):
                        reasons.append("high_words_per_sec")

                    resolved_duration_key = detect_key(row, duration_key, DURATION_KEY_CANDIDATES)
                    manifest_duration = (
                        parse_positive_float(row.get(resolved_duration_key))
                        if resolved_duration_key
                        else None
                    )
                    metrics["manifest_duration"] = manifest_duration
                    if manifest_duration is None:
                        reasons.append("invalid_manifest_duration")
                    else:
                        delta = abs(manifest_duration - real_duration)
                        metrics["duration_delta"] = round(delta, 6)
                        if delta > float(max_duration_delta_sec):
                            reasons.append("duration_mismatch")

            if reasons:
                reject_counts.update(reasons)
                rejected += 1
                payload = reject_payload(row, line_num=line_num, reasons=reasons, metrics=metrics)
                frej.write(json.dumps(payload, ensure_ascii=False) + "\n")
                if len(reject_examples) < 20:
                    reject_examples.append(payload)
                continue

            output_row = dict(row)
            output_row["validation_source_manifest"] = str(input_path)
            output_row["validation_source_lineno"] = int(line_num)
            fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            kept += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "rejects": str(rejects_path),
        "summary_json": str(summary_path),
        "total": total,
        "kept": kept,
        "rejected": rejected,
        "reject_counts": dict(sorted(reject_counts.items())),
        "thresholds": {
            "min_duration_sec": float(min_duration_sec),
            "max_duration_sec": float(max_duration_sec),
            "max_chars_per_sec": float(max_chars_per_sec),
            "max_words_per_sec": float(max_words_per_sec),
            "expected_sample_rate": int(expected_sample_rate),
            "require_mono": bool(require_mono),
            "max_duration_delta_sec": float(max_duration_delta_sec),
        },
        "audio": {
            "sample_rates": dict(sorted(sample_rates.items())),
            "channels": dict(sorted(channels.items())),
            "duration_sec": summarize_values(duration_values),
        },
        "transcript_rates": {
            "chars_per_sec": summarize_values(chars_per_sec_values),
            "words_per_sec": summarize_values(words_per_sec_values),
        },
        "reject_examples": reject_examples,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = validate_manifest_file(
        args.input,
        args.output,
        args.rejects,
        args.summary_json,
        text_key=args.text_key,
        audio_key=args.audio_key,
        duration_key=args.duration_key,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        max_chars_per_sec=args.max_chars_per_sec,
        max_words_per_sec=args.max_words_per_sec,
        expected_sample_rate=args.expected_sample_rate,
        require_mono=args.require_mono,
        max_duration_delta_sec=args.max_duration_delta_sec,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
