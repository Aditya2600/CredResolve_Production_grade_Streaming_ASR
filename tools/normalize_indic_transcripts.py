#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "transcript_normalization" / "language_mappings.json"
TEXT_COLUMN_CANDIDATES = (
    "original_text",
    "text",
    "transcript",
    "raw_transcript",
    "native_language_transcript",
    "normalized_transcript",
)
LANGUAGE_COLUMN_CANDIDATES = ("language", "lang", "locale", "language_code")
JSON_TEXT_KEY_CANDIDATES = ("en_text", "text", "transcript", "reference", "utterance")
MULTISPACE_RE = re.compile(r"\s+")
LATIN_RE = re.compile(r"[A-Za-z]")
GENERIC_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{2,4}|(?:[A-Za-z][ .-]){1,3}[A-Za-z])(?![A-Za-z0-9])")
JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")
ODIA_RE = re.compile(r"[\u0B00-\u0B7F]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")
MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")
SCRIPT_LANGUAGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bn", BENGALI_RE),
    ("gu", GUJARATI_RE),
    ("or", ODIA_RE),
    ("ta", TAMIL_RE),
    ("te", TELUGU_RE),
    ("kn", KANNADA_RE),
    ("ml", MALAYALAM_RE),
)
LANGUAGE_CODE_TO_LABEL = {
    "bn": "BENGALI",
    "gu": "GUJARATI",
    "hi": "HINDI",
    "kn": "KANNADA",
    "ml": "MALAYALAM",
    "mr": "MARATHI",
    "or": "ODIA",
    "ta": "TAMIL",
    "te": "TELUGU",
}
HINDI_HINTS = (
    " मैं ",
    " क्या ",
    " हूँ",
    " हैं",
    " बोल रही",
    " बोल रहा",
    " आपके ",
    " आपका ",
    " हमारे रिकॉर्ड",
    " नमस्ते",
)
MARATHI_HINTS = (
    " मी ",
    " आहे ",
    " आहे.",
    " आहेत",
    " बोलत आहे",
    " बोलतेय",
    " यांच्याशी",
    " खात्री",
    " आमच्या नोंदीनुसार",
    " नमस्कार",
)


@dataclass(frozen=True)
class EntityRule:
    entity_id: str
    kind: str
    normalized: str
    asr: str
    patterns: tuple[re.Pattern[str], ...]
    priority: int
    max_pattern_length: int


@dataclass(frozen=True)
class LanguageConfig:
    code: str
    aliases: frozenset[str]
    letter_spellings: dict[str, str]
    entities: tuple[EntityRule, ...]
    generic_acronym_min_length: int
    generic_acronym_max_length: int


@dataclass(frozen=True)
class CandidateMatch:
    start: int
    end: int
    entity_id: str
    kind: str
    original: str
    normalized: str
    asr: str
    priority: int
    score: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize code-mixed Indic transcripts into ASR-friendly native-script targets "
            "plus canonicalized mixed-script text for fine-tuning workflows."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV, JSON, or JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV, JSON, or JSONL file.")
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "json", "jsonl"),
        default="auto",
        help="Override input format. Default: auto",
    )
    parser.add_argument(
        "--output-format",
        choices=("auto", "csv", "json", "jsonl"),
        default="auto",
        help="Override output format. Default: auto",
    )
    parser.add_argument(
        "--text-column",
        help=(
            "Column containing transcript text or a transcript JSON payload. "
            "Auto-detected by default."
        ),
    )
    parser.add_argument(
        "--language-column",
        help="Column containing language labels like HINDI or gu. Auto-detected by default.",
    )
    parser.add_argument(
        "--language",
        help="Optional fixed language override applied to every row, for example hi or MARATHI.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Language mapping config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--columns-only",
        action="store_true",
        help="Write only original_text, asr_l1_text, normalized_l2_text, and entity_map_json.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent to use for JSON array output. Default: 2",
    )
    return parser.parse_args()


def normalize_space(value: str) -> str:
    return MULTISPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except TypeError:
        return False
    if isinstance(result, bool):
        return result
    return False


def normalize_language_alias(value: str) -> str:
    return normalize_space(value).replace("-", " ").replace("_", " ").casefold()


def sanitize_scalar(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def count_occurrences(text: str, hints: tuple[str, ...]) -> int:
    return sum(text.count(hint) for hint in hints)


def literal_pattern_to_regex(value: str, *, word_boundary: bool = True) -> re.Pattern[str]:
    text = normalize_space(value)
    escaped = re.escape(text).replace(r"\ ", r"\s+")
    if word_boundary:
        escaped = rf"(?<!\w){escaped}(?!\w)"
    flags = re.IGNORECASE if LATIN_RE.search(text) else 0
    return re.compile(escaped, flags)


def load_language_configs(config_path: Path) -> tuple[dict[str, LanguageConfig], dict[str, str]]:
    raw_payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_languages = raw_payload.get("languages")
    if not isinstance(raw_languages, dict) or not raw_languages:
        raise SystemExit(f"{config_path}: expected a non-empty `languages` object")

    configs: dict[str, LanguageConfig] = {}
    alias_map: dict[str, str] = {}
    for code, raw_language in raw_languages.items():
        if not isinstance(raw_language, dict):
            raise SystemExit(f"{config_path}: language `{code}` must be an object")

        raw_letters = raw_language.get("letter_spellings")
        if not isinstance(raw_letters, dict) or not raw_letters:
            raise SystemExit(f"{config_path}: language `{code}` is missing `letter_spellings`")
        letter_spellings = {str(key).upper(): normalize_space(str(value)) for key, value in raw_letters.items()}

        entities: list[EntityRule] = []
        for raw_entity in raw_language.get("entities", []):
            if not isinstance(raw_entity, dict):
                raise SystemExit(f"{config_path}: invalid entity entry under `{code}`")
            entity_id = normalize_space(str(raw_entity.get("id", "")))
            normalized = normalize_space(str(raw_entity.get("normalized", "")))
            asr = normalize_space(str(raw_entity.get("asr", "")))
            patterns = raw_entity.get("patterns", [])
            if not entity_id or not normalized or not asr or not isinstance(patterns, list) or not patterns:
                raise SystemExit(f"{config_path}: entity under `{code}` is missing id/normalized/asr/patterns")
            raw_patterns = [normalize_space(str(pattern)) for pattern in patterns if normalize_space(str(pattern))]
            compiled_patterns = tuple(
                literal_pattern_to_regex(
                    pattern,
                    word_boundary=bool(raw_entity.get("word_boundary", True)),
                )
                for pattern in raw_patterns
            )
            entities.append(
                EntityRule(
                    entity_id=entity_id,
                    kind=normalize_space(str(raw_entity.get("kind", "term"))) or "term",
                    normalized=normalized,
                    asr=asr,
                    patterns=compiled_patterns,
                    priority=int(raw_entity.get("priority", 100)),
                    max_pattern_length=max(len(pattern) for pattern in raw_patterns),
                )
            )

        config = LanguageConfig(
            code=code,
            aliases=frozenset(
                normalize_language_alias(str(alias))
                for alias in [code, *raw_language.get("aliases", [])]
                if normalize_language_alias(str(alias))
            ),
            letter_spellings=letter_spellings,
            entities=tuple(sorted(entities, key=lambda entity: (-entity.priority, -entity.max_pattern_length))),
            generic_acronym_min_length=int(raw_language.get("generic_acronym_min_length", 2)),
            generic_acronym_max_length=int(raw_language.get("generic_acronym_max_length", 4)),
        )
        configs[code] = config
        for alias in config.aliases:
            alias_map[alias] = code
    return configs, alias_map


def resolve_language_config(
    language: str,
    *,
    configs: dict[str, LanguageConfig],
    alias_map: dict[str, str],
) -> LanguageConfig:
    key = normalize_language_alias(language)
    code = alias_map.get(key)
    if code is None:
        supported = ", ".join(sorted(configs))
        raise SystemExit(f"unsupported language `{language}`; supported codes: {supported}")
    return configs[code]


def detect_format(path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in JSONL_SUFFIXES:
        return "jsonl"
    raise SystemExit(f"could not infer format from {path}; pass --input-format/--output-format")


def coerce_row(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): sanitize_scalar(item) for key, item in value.items()}
    if isinstance(value, str):
        return {"original_text": value}
    raise SystemExit("JSON rows must be objects or strings")


def load_rows(path: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "csv":
        frame = pd.read_csv(path)
        return [coerce_row(row) for row in frame.to_dict(orient="records")]
    if fmt == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [coerce_row(row) for row in payload]
        return [coerce_row(payload)]
    if fmt == "jsonl":
        rows: list[dict[str, Any]] = []
        for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            rows.append(coerce_row(payload))
        return rows
    raise SystemExit(f"unsupported format `{fmt}`")


def write_rows(path: Path, fmt: str, rows: list[dict[str, Any]], *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    if fmt == "json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
        return
    if fmt == "jsonl":
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        if payload:
            payload += "\n"
        path.write_text(payload, encoding="utf-8")
        return
    raise SystemExit(f"unsupported format `{fmt}`")


def detect_text_column(rows: list[dict[str, Any]], explicit: str | None) -> str:
    if explicit:
        return explicit
    if not rows:
        raise SystemExit("input is empty")
    sample = rows[0]
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in sample:
            return candidate
    raise SystemExit(f"could not detect transcript column; tried: {', '.join(TEXT_COLUMN_CANDIDATES)}")


def detect_language_column(rows: list[dict[str, Any]], explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if not rows:
        return None
    sample = rows[0]
    for candidate in LANGUAGE_COLUMN_CANDIDATES:
        if candidate in sample:
            return candidate
    return None


def extract_transcript_text(value: Any) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, dict):
        if isinstance(value.get("interaction_transcript"), list):
            parts = [extract_transcript_text(item) for item in value["interaction_transcript"]]
            return normalize_space(" ".join(part for part in parts if part))
        for candidate in JSON_TEXT_KEY_CANDIDATES:
            if candidate in value and not is_missing(value[candidate]):
                return normalize_space(str(value[candidate]))
        return normalize_space(json.dumps(value, ensure_ascii=False))
    if isinstance(value, list):
        parts = [extract_transcript_text(item) for item in value]
        return normalize_space(" ".join(part for part in parts if part))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text[0] in "{[":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return normalize_space(text)
            return extract_transcript_text(payload)
        return normalize_space(text)
    return normalize_space(str(value))


def infer_devanagari_language(text: str) -> str | None:
    padded = f" {normalize_space(text)} "
    hindi_score = count_occurrences(padded, HINDI_HINTS)
    marathi_score = count_occurrences(padded, MARATHI_HINTS)
    if marathi_score > hindi_score:
        return "mr"
    if hindi_score > marathi_score:
        return "hi"
    if " मी " in padded or " आहे" in padded:
        return "mr"
    if " मैं " in padded or " क्या " in padded or " हूँ" in padded:
        return "hi"
    return None


def infer_language_code_from_text(text: str) -> str | None:
    cleaned_text = normalize_space(text)
    if not cleaned_text:
        return None

    script_counts = {
        code: len(pattern.findall(cleaned_text))
        for code, pattern in SCRIPT_LANGUAGE_PATTERNS
    }
    devanagari_count = len(DEVANAGARI_RE.findall(cleaned_text))
    top_non_devanagari_code, top_non_devanagari_count = max(
        script_counts.items(),
        key=lambda item: item[1],
    )
    if top_non_devanagari_count > devanagari_count and top_non_devanagari_count > 0:
        return top_non_devanagari_code
    if devanagari_count > 0:
        return infer_devanagari_language(cleaned_text)
    if top_non_devanagari_count > 0:
        return top_non_devanagari_code
    return None


def build_row_identifier(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("row_id", "call_id", "inventory_source_file", "inventory_source_row_number"):
        value = row.get(key)
        if is_missing(value):
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "unknown row"


def collect_explicit_candidates(text: str, config: LanguageConfig) -> list[CandidateMatch]:
    candidates: list[CandidateMatch] = []
    seen: set[tuple[int, int, str]] = set()
    for rule in config.entities:
        for pattern in rule.patterns:
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), rule.entity_id)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    CandidateMatch(
                        start=match.start(),
                        end=match.end(),
                        entity_id=rule.entity_id,
                        kind=rule.kind,
                        original=match.group(0),
                        normalized=rule.normalized,
                        asr=rule.asr,
                        priority=rule.priority,
                        score=rule.max_pattern_length,
                    )
                )
    return candidates


def build_generic_acronym_candidate(match: re.Match[str], config: LanguageConfig) -> CandidateMatch | None:
    letters = [letter.upper() for letter in re.findall(r"[A-Za-z]", match.group(0))]
    if not letters:
        return None
    if len(letters) < config.generic_acronym_min_length or len(letters) > config.generic_acronym_max_length:
        return None
    if any(letter not in config.letter_spellings for letter in letters):
        return None
    normalized = "".join(letters)
    asr = " ".join(config.letter_spellings[letter] for letter in letters)
    return CandidateMatch(
        start=match.start(),
        end=match.end(),
        entity_id=normalized.casefold(),
        kind="acronym",
        original=match.group(0),
        normalized=normalized,
        asr=asr,
        priority=10,
        score=len(normalized),
    )


def collect_generic_candidates(text: str, config: LanguageConfig) -> list[CandidateMatch]:
    candidates: list[CandidateMatch] = []
    seen: set[tuple[int, int]] = set()
    for match in GENERIC_ACRONYM_RE.finditer(text):
        key = (match.start(), match.end())
        if key in seen:
            continue
        seen.add(key)
        candidate = build_generic_acronym_candidate(match, config)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def select_candidates(candidates: list[CandidateMatch]) -> list[CandidateMatch]:
    ordered = sorted(candidates, key=lambda item: (item.start, -item.priority, -item.score, -(item.end - item.start)))
    selected: list[CandidateMatch] = []
    current_end = -1
    for candidate in ordered:
        if candidate.start < current_end:
            continue
        selected.append(candidate)
        current_end = candidate.end
    return selected


def apply_replacements(text: str, matches: list[CandidateMatch], *, target: str) -> str:
    parts: list[str] = []
    cursor = 0
    for match in matches:
        parts.append(text[cursor:match.start])
        parts.append(match.asr if target == "asr" else match.normalized)
        cursor = match.end
    parts.append(text[cursor:])
    return normalize_space("".join(parts))


def serialize_entity_map(matches: list[CandidateMatch]) -> str:
    payload = [
        {
            "id": match.entity_id,
            "kind": match.kind,
            "original": match.original,
            "asr": match.asr,
            "normalized": match.normalized,
            "start": match.start,
            "end": match.end,
        }
        for match in matches
    ]
    return json.dumps(payload, ensure_ascii=False)


def normalize_text_for_language(
    text: str,
    language: str,
    *,
    configs: dict[str, LanguageConfig],
    alias_map: dict[str, str],
) -> dict[str, str]:
    cleaned_text = normalize_space(text)
    if not cleaned_text:
        return {
            "original_text": "",
            "asr_l1_text": "",
            "normalized_l2_text": "",
            "entity_map_json": "[]",
        }

    config = resolve_language_config(language, configs=configs, alias_map=alias_map)
    matches = select_candidates(collect_explicit_candidates(cleaned_text, config) + collect_generic_candidates(cleaned_text, config))
    return {
        "original_text": cleaned_text,
        "asr_l1_text": apply_replacements(cleaned_text, matches, target="asr") if matches else cleaned_text,
        "normalized_l2_text": apply_replacements(cleaned_text, matches, target="normalized") if matches else cleaned_text,
        "entity_map_json": serialize_entity_map(matches),
    }


def resolve_row_language(
    row: dict[str, Any],
    *,
    fixed_language: str | None,
    language_column: str | None,
    transcript_text: str,
) -> tuple[str, str | None]:
    if fixed_language:
        return fixed_language, None
    if language_column is None:
        raise SystemExit("language could not be resolved; pass --language or provide a language column")
    value = row.get(language_column)
    if not is_missing(value):
        return str(value), None

    inferred_code = infer_language_code_from_text(transcript_text)
    if inferred_code is None:
        raise SystemExit(
            f"row is missing language in column `{language_column}` and transcript-based inference failed "
            f"for {build_row_identifier(row)}"
        )
    return inferred_code, LANGUAGE_CODE_TO_LABEL.get(inferred_code, inferred_code.upper())


def build_output_row(
    row: dict[str, Any],
    *,
    transcript_text: str,
    language: str,
    inferred_language_label: str | None,
    language_column: str | None,
    configs: dict[str, LanguageConfig],
    alias_map: dict[str, str],
    columns_only: bool,
) -> dict[str, Any]:
    normalized_fields = normalize_text_for_language(
        transcript_text,
        language,
        configs=configs,
        alias_map=alias_map,
    )
    if columns_only:
        return normalized_fields
    output_row = dict(normalized_fields)
    for key, value in row.items():
        if (
            key == language_column
            and inferred_language_label is not None
            and is_missing(value)
        ):
            output_row[key] = inferred_language_label
            continue
        if key not in output_row:
            output_row[key] = sanitize_scalar(value)
    return output_row


def process_rows(
    rows: list[dict[str, Any]],
    *,
    text_column: str,
    language_column: str | None,
    fixed_language: str | None,
    configs: dict[str, LanguageConfig],
    alias_map: dict[str, str],
    columns_only: bool,
) -> list[dict[str, Any]]:
    processed_rows: list[dict[str, Any]] = []
    for row in rows:
        transcript_text = extract_transcript_text(row.get(text_column))
        resolved_language, inferred_language_label = resolve_row_language(
            row,
            fixed_language=fixed_language,
            language_column=language_column,
            transcript_text=transcript_text,
        )
        processed_rows.append(
            build_output_row(
                row,
                transcript_text=transcript_text,
                language=resolved_language,
                inferred_language_label=inferred_language_label,
                language_column=language_column,
                configs=configs,
                alias_map=alias_map,
                columns_only=columns_only,
            )
        )
    return processed_rows


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    config_path = args.config.expanduser().resolve()

    configs, alias_map = load_language_configs(config_path)
    input_format = detect_format(input_path, args.input_format)
    output_format = detect_format(output_path, args.output_format)
    rows = load_rows(input_path, input_format)
    if not rows:
        raise SystemExit(f"{input_path}: input is empty")

    text_column = detect_text_column(rows, args.text_column)
    language_column = detect_language_column(rows, args.language_column)
    processed_rows = process_rows(
        rows,
        text_column=text_column,
        language_column=language_column,
        fixed_language=args.language,
        configs=configs,
        alias_map=alias_map,
        columns_only=args.columns_only,
    )
    write_rows(output_path, output_format, processed_rows, indent=args.indent)
    print(
        json.dumps(
            {
                "rows": len(processed_rows),
                "input": str(input_path),
                "output": str(output_path),
                "text_column": text_column,
                "language_column": language_column if args.language is None else None,
                "fixed_language": args.language,
                "config": str(config_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
