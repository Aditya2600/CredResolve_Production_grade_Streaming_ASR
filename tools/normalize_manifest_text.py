#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from asr_text_normalizer import normalize_asr_text
except ModuleNotFoundError:  # pragma: no cover - used when imported as tools.*
    from tools.asr_text_normalizer import normalize_asr_text


TEXT_KEY_CANDIDATES = ("text", "reference", "transcript", "normalized_text", "sentence")
LANGUAGE_KEY_CANDIDATES = ("language", "language_id", "lang", "locale", "language_code")
TARGET_FIELDS = ("asr_l1_text", "normalized_l2_text")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("asr_text_normalizer.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a NeMo JSONL manifest's transcript text into ASR-friendly Indic "
            "targets and write kept/rejected manifests for fine-tuning."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL manifest.")
    parser.add_argument("--output", type=Path, required=True, help="Output normalized JSONL manifest.")
    parser.add_argument("--rejects", type=Path, required=True, help="Rejected-row JSONL manifest.")
    parser.add_argument("--summary-json", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--text-key", help="Transcript key. Auto-detected per row by default.")
    parser.add_argument("--language-key", help="Language key. Auto-detected per row by default.")
    parser.add_argument("--language", help="Fixed language override for every row, for example hi or MARATHI.")
    parser.add_argument(
        "--target-field",
        choices=TARGET_FIELDS,
        default="asr_l1_text",
        help="Normalized field to write into manifest text. Default: asr_l1_text",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Language mapping config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=220)
    parser.add_argument("--max-words", type=int, default=40)
    return parser.parse_args()


def detect_key(row: dict[str, Any], explicit: str | None, candidates: tuple[str, ...]) -> str | None:
    if explicit:
        return explicit
    for candidate in candidates:
        if candidate in row:
            return candidate
    return None


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def extract_transcript_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in TEXT_KEY_CANDIDATES:
            candidate = value.get(key)
            if not is_missing(candidate):
                return str(candidate).strip()
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def resolve_language(row: dict[str, Any], *, fixed_language: str | None, language_key: str | None, text: str) -> str | None:
    del text
    if fixed_language:
        return fixed_language
    if language_key is not None and not is_missing(row.get(language_key)):
        return str(row[language_key])
    return None


def reject_row(
    handle: Any,
    row: dict[str, Any],
    *,
    reason: str,
    line_num: int,
    raw_text: str | None = None,
    detail: str | None = None,
) -> None:
    payload = dict(row)
    payload["reject_reason"] = reason
    payload["source_lineno"] = line_num
    if raw_text is not None:
        payload["raw_text"] = raw_text
    if detail:
        payload["reject_detail"] = detail
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_manifest_file(
    input_path: Path,
    output_path: Path,
    rejects_path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    text_key: str | None = None,
    language_key: str | None = None,
    fixed_language: str | None = None,
    target_field: str = "asr_l1_text",
    min_chars: int = 2,
    max_chars: int = 220,
    max_words: int = 40,
) -> dict[str, Any]:
    if target_field not in TARGET_FIELDS:
        raise ValueError(f"target_field must be one of {', '.join(TARGET_FIELDS)}")

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    rejects_path = rejects_path.expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve() if config_path is not None else DEFAULT_CONFIG_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    rejected = 0
    reject_counts: Counter[str] = Counter()

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
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                reject_counts["invalid_json"] += 1
                rejected += 1
                reject_row(
                    frej,
                    {"raw_line": raw_line.rstrip("\n")},
                    reason="invalid_json",
                    line_num=line_num,
                    detail=str(exc),
                )
                continue
            if not isinstance(row, dict):
                reject_counts["non_object_row"] += 1
                rejected += 1
                reject_row(frej, {"row": row}, reason="non_object_row", line_num=line_num)
                continue

            resolved_text_key = detect_key(row, text_key, TEXT_KEY_CANDIDATES)
            if resolved_text_key is None:
                reject_counts["missing_text_key"] += 1
                rejected += 1
                reject_row(frej, row, reason="missing_text_key", line_num=line_num)
                continue

            raw_text = extract_transcript_text(row.get(resolved_text_key))
            lang_key = detect_key(row, language_key, LANGUAGE_KEY_CANDIDATES)
            lang = resolve_language(row, fixed_language=fixed_language, language_key=lang_key, text=raw_text)

            reason = None
            if not raw_text:
                reason = "empty_text"
            else:
                clean_text = normalize_asr_text(raw_text, lang)
            clean_text = clean_text if reason is None else ""
            if reason is None and not clean_text:
                reason = "empty_after_normalization"
            elif reason is None and len(clean_text) < min_chars:
                reason = "too_short"
            elif reason is None and len(clean_text) > max_chars:
                reason = "too_long"
            elif reason is None and len(clean_text.split()) > max_words:
                reason = "too_many_words"

            if reason:
                reject_counts[reason] += 1
                rejected += 1
                reject_row(
                    frej,
                    row,
                    reason=reason,
                    line_num=line_num,
                    raw_text=raw_text,
                )
                continue

            output_row = dict(row)
            output_row["raw_text"] = raw_text
            output_row["text"] = clean_text
            output_row["source_lineno"] = line_num
            fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            kept += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rejects": str(rejects_path),
        "normalizer": "tools.asr_text_normalizer.normalize_asr_text",
        "config": str(config_path),
        "target_field": target_field,
        "total": total,
        "kept": kept,
        "rejected": rejected,
        "reject_counts": dict(reject_counts),
        "filters": {
            "min_chars": min_chars,
            "max_chars": max_chars,
            "max_words": max_words,
        },
    }


def main() -> int:
    args = parse_args()
    summary = normalize_manifest_file(
        args.input,
        args.output,
        args.rejects,
        config_path=args.config,
        text_key=args.text_key,
        language_key=args.language_key,
        fixed_language=args.language,
        target_field=args.target_field,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_words=args.max_words,
    )
    if args.summary_json:
        summary_path = args.summary_json.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
