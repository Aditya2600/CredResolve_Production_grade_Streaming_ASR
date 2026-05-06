#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_vaani_hindi_to_nemo import (
    DEFAULT_FILTERS,
    VaaniExportFilters,
    detect_audio_field,
    detect_text_field,
    export_samples_to_nemo,
    prepend_first_sample,
    resolve_cache_dir,
    resolve_hf_token,
    sanitize_component,
)
from tools.indicvoices_dataset import VAANI_DATASET_ID, load_vaani_stream


DEFAULT_LANGUAGE_HOURS: OrderedDict[str, float] = OrderedDict(
    [
        ("Hindi", 27.4),
        ("Gujarati", 4.2),
        ("Marathi", 4.0),
        ("Telugu", 3.9),
        ("Tamil", 3.8),
        ("Kannada", 2.9),
        ("Bengali", 2.6),
        ("Malayalam", 1.2),
    ]
)
VAANI_LANGUAGE_CODES: dict[str, str] = {
    "Bengali": "bn",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
}
LANGUAGE_ALIASES: dict[str, str] = {
    "bn": "Bengali",
    "ben": "Bengali",
    "bengali": "Bengali",
    "bangla": "Bengali",
    "gu": "Gujarati",
    "guj": "Gujarati",
    "gujarati": "Gujarati",
    "hi": "Hindi",
    "hin": "Hindi",
    "hindi": "Hindi",
    "kn": "Kannada",
    "kan": "Kannada",
    "kannada": "Kannada",
    "ml": "Malayalam",
    "mal": "Malayalam",
    "malayalam": "Malayalam",
    "mr": "Marathi",
    "mar": "Marathi",
    "marathi": "Marathi",
    "ta": "Tamil",
    "tam": "Tamil",
    "tamil": "Tamil",
    "te": "Telugu",
    "tel": "Telugu",
    "telugu": "Telugu",
}
DEFAULT_OUTPUT_DIR = Path("artifacts/vaani_50h_multilingual_train")


def format_language_hours(language_hours: OrderedDict[str, float]) -> str:
    return ",".join(f"{language}={hours:g}" for language, hours in language_hours.items())


def normalize_language_key(value: object) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def resolve_language_config(value: object) -> str:
    key = normalize_language_key(value)
    if not key:
        raise SystemExit("Language name is empty.")
    try:
        return LANGUAGE_ALIASES[key]
    except KeyError as exc:
        known = ", ".join(DEFAULT_LANGUAGE_HOURS.keys())
        raise SystemExit(f"Unsupported Vaani language {value!r}. Known languages: {known}") from exc


def parse_language_hours(raw_value: str | None) -> OrderedDict[str, float]:
    if raw_value is None or not raw_value.strip():
        return OrderedDict(DEFAULT_LANGUAGE_HOURS)

    language_hours: OrderedDict[str, float] = OrderedDict()
    for item in raw_value.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"Invalid language-hours item {chunk!r}. Expected LANGUAGE=hours.")
        raw_language, raw_hours = chunk.split("=", 1)
        language = resolve_language_config(raw_language)
        if language in language_hours:
            raise SystemExit(f"Duplicate Vaani language in --language-hours: {language}")
        try:
            hours = float(raw_hours.strip())
        except ValueError as exc:
            raise SystemExit(f"Invalid hour value in language-hours item {chunk!r}.") from exc
        if hours <= 0:
            raise SystemExit(f"Invalid hour value in language-hours item {chunk!r}. Must be > 0.")
        language_hours[language] = hours

    if not language_hours:
        raise SystemExit("--language-hours is empty.")
    return language_hours


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a multilingual ARTPARK-IISc/Vaani-transcription-part training subset "
            "to local WAV files plus a combined NeMo JSONL manifest."
        )
    )
    parser.add_argument(
        "--language-hours",
        default=format_language_hours(DEFAULT_LANGUAGE_HOURS),
        help=(
            "Comma-separated Vaani language=hours list. "
            f"Default: {format_language_hours(DEFAULT_LANGUAGE_HOURS)}"
        ),
    )
    parser.add_argument("--split", default="train", help="Vaani split. Default: train")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-token", help="Optional Hugging Face token for gated Vaani access.")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face datasets cache dir.")
    parser.add_argument("--seed", type=int, default=42, help="Streaming shuffle seed. Default: 42")
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10000,
        help="Streaming shuffle buffer size per language. Use 1 to preserve source order. Default: 10000",
    )
    parser.add_argument("--max-scan-rows", type=int, help="Optional cap on scanned source rows per language.")
    parser.add_argument("--text-field", help="Override transcript field for all languages. Auto-detected by default.")
    parser.add_argument("--audio-field", help="Override audio field for all languages. Auto-detected by default.")
    parser.add_argument("--language-field", default="language", help="Language metadata field. Default: language")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_FILTERS.min_duration)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_FILTERS.max_duration)
    parser.add_argument("--max-words", type=int, default=DEFAULT_FILTERS.max_words)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_FILTERS.max_chars)
    parser.add_argument("--min-words-per-sec", type=float, default=DEFAULT_FILTERS.min_words_per_sec)
    parser.add_argument("--max-words-per-sec", type=float, default=DEFAULT_FILTERS.max_words_per_sec)
    parser.add_argument(
        "--drop-unintelligible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject transcripts containing <unintelligible>. Default: enabled",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Write the combined manifest even if one or more languages do not reach their requested hours.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{lineno}: expected a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def combine_language_manifests(
    manifest_paths: OrderedDict[str, Path],
    *,
    output_path: Path,
    seed: int,
) -> list[dict[str, Any]]:
    combined_rows: list[dict[str, Any]] = []
    for language, manifest_path in manifest_paths.items():
        rows = load_jsonl(manifest_path)
        if not rows:
            raise SystemExit(f"{manifest_path}: manifest for {language} is empty")
        combined_rows.extend(rows)

    random.Random(seed).shuffle(combined_rows)
    write_jsonl(output_path, combined_rows)
    return combined_rows


def summarize_manifest_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_lang: Counter[str] = Counter()
    rows_by_config: Counter[str] = Counter()
    duration_by_lang: Counter[str] = Counter()
    duration_by_config: Counter[str] = Counter()
    for row in rows:
        lang = str(row.get("lang") or "unknown")
        config = str(row.get("dataset_config") or "unknown")
        duration = float(row.get("duration") or 0.0)
        rows_by_lang[lang] += 1
        rows_by_config[config] += 1
        duration_by_lang[lang] += duration
        duration_by_config[config] += duration

    return {
        "rows": len(rows),
        "rows_by_lang": dict(sorted(rows_by_lang.items())),
        "rows_by_dataset_config": dict(sorted(rows_by_config.items())),
        "duration_hours_by_lang": {
            lang: round(seconds / 3600.0, 6)
            for lang, seconds in sorted(duration_by_lang.items())
        },
        "duration_hours_by_dataset_config": {
            config: round(seconds / 3600.0, 6)
            for config, seconds in sorted(duration_by_config.items())
        },
        "duration_hours_total": round(sum(duration_by_lang.values()) / 3600.0, 6),
    }


def export_multilingual_vaani(args: argparse.Namespace) -> dict[str, Any]:
    token = resolve_hf_token(args.hf_token)
    cache_dir = resolve_cache_dir(args.cache_dir)
    language_hours = parse_language_hours(args.language_hours)
    out_dir = args.out_dir.expanduser().resolve()
    language_root_dir = out_dir / "languages"
    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"
    filters = VaaniExportFilters(
        min_duration=float(args.min_duration),
        max_duration=float(args.max_duration),
        max_words=int(args.max_words),
        max_chars=int(args.max_chars),
        min_words_per_sec=float(args.min_words_per_sec),
        max_words_per_sec=float(args.max_words_per_sec),
        language_field=str(args.language_field or ""),
        drop_unintelligible=bool(args.drop_unintelligible),
    )

    language_manifests: OrderedDict[str, Path] = OrderedDict()
    language_summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()
    shortfalls: list[str] = []

    for offset, (language_config, target_hours) in enumerate(language_hours.items()):
        language_code = VAANI_LANGUAGE_CODES[language_config]
        language_out_dir = language_root_dir / sanitize_component(language_config)
        print(f"Exporting {language_config} ({language_code}) target={target_hours:g}h", file=sys.stderr)

        dataset = load_vaani_stream(
            dataset_config=language_config,
            split=args.split,
            token=token,
            cache_dir=cache_dir,
        )
        iterator = iter(dataset)
        try:
            first_sample = next(iterator)
        except StopIteration as exc:
            raise SystemExit(f"Vaani dataset split is empty for {language_config}.") from exc

        text_field = args.text_field or detect_text_field(first_sample)
        audio_field = args.audio_field or detect_audio_field(first_sample)
        summary = export_samples_to_nemo(
            prepend_first_sample(first_sample, iterator),
            out_dir=language_out_dir,
            dataset_config=language_config,
            split=args.split,
            target_hours=float(target_hours),
            seed=int(args.seed) + offset,
            shuffle_buffer_size=int(args.shuffle_buffer_size),
            filters=filters,
            text_field=text_field,
            audio_field=audio_field,
            language_code=language_code,
            max_scan_rows=args.max_scan_rows,
        )
        language_summaries[language_config] = summary
        language_manifests[language_config] = Path(summary["outputs"]["manifest_jsonl"])
        if not summary["target_reached"]:
            shortfalls.append(f"{language_config}: kept {summary['kept_hours']}h of {summary['target_hours']}h")

    combined_rows = combine_language_manifests(
        language_manifests,
        output_path=manifest_path,
        seed=int(args.seed),
    )
    row_summary = summarize_manifest_rows(combined_rows)
    kept_hours_by_language = OrderedDict(
        (language, language_summaries[language]["kept_hours"])
        for language in language_hours
    )
    target_reached_by_language = OrderedDict(
        (language, bool(language_summaries[language]["target_reached"]))
        for language in language_hours
    )
    combined_summary = {
        "dataset": VAANI_DATASET_ID,
        "split": args.split,
        "target_hours_total": round(sum(language_hours.values()), 6),
        "target_hours_by_language": dict(language_hours),
        "kept_hours_total": row_summary["duration_hours_total"],
        "kept_hours_by_language": dict(kept_hours_by_language),
        "target_reached": all(target_reached_by_language.values()),
        "target_reached_by_language": dict(target_reached_by_language),
        "language_codes": {
            language: VAANI_LANGUAGE_CODES[language]
            for language in language_hours
        },
        "seed": int(args.seed),
        "shuffle_buffer_size": int(args.shuffle_buffer_size),
        "rows": row_summary,
        "shortfalls": shortfalls,
        "language_exports": dict(language_summaries),
        "outputs": {
            "out_dir": str(out_dir),
            "language_root_dir": str(language_root_dir),
            "manifest_jsonl": str(manifest_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return combined_summary


def main() -> int:
    args = parse_args()
    summary = export_multilingual_vaani(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["shortfalls"] and not args.allow_shortfall:
        raise SystemExit(
            "One or more language targets were not reached: "
            + "; ".join(str(item) for item in summary["shortfalls"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
