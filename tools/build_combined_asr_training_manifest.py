#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DOMAIN_TRAIN = Path(
    "artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_train_clean_t4.jsonl"
)
DEFAULT_DOMAIN_DEV = Path(
    "artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_dev_clean_t4.jsonl"
)
DEFAULT_VAANI_TRAIN = Path("artifacts/vaani_50h_multilingual_train/manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/asr_train_domain_plus_vaani50h")

TEXT_KEY_CANDIDATES = ("text", "reference", "normalized_text", "transcript", "sentence")
AUDIO_KEY_CANDIDATES = ("audio_filepath", "audio_path", "audio", "path")
DURATION_KEY_CANDIDATES = ("duration", "audio_duration", "audio_duration_sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a domain-weighted ASR training manifest by mixing existing in-domain "
            "manifests with a filtered multilingual Vaani NeMo manifest."
        )
    )
    parser.add_argument("--domain-train-manifest", type=Path, default=DEFAULT_DOMAIN_TRAIN)
    parser.add_argument("--domain-dev-manifest", type=Path, default=DEFAULT_DOMAIN_DEV)
    parser.add_argument("--vaani-train-manifest", type=Path, default=DEFAULT_VAANI_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strategy",
        choices=("domain_weighted", "pure_additive"),
        default="domain_weighted",
        help="Default: domain_weighted",
    )
    parser.add_argument(
        "--domain-min-row-fraction",
        type=float,
        default=0.25,
        help="Minimum target domain row fraction for domain_weighted. Default: 0.25",
    )
    parser.add_argument(
        "--domain-repeat-cap",
        type=int,
        default=3,
        help="Maximum integer repeat factor for domain train rows. Default: 3",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle/interleave seed. Default: 42")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{resolved}:{lineno}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{resolved}: manifest is empty")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def detect_first_key(row: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in row:
            return key
    return None


def normalize_to_nemo_row(row: dict[str, Any], *, source_name: str, source_index: int, repeat_index: int = 0) -> dict[str, Any]:
    output = dict(row)
    text_key = detect_first_key(output, TEXT_KEY_CANDIDATES)
    audio_key = detect_first_key(output, AUDIO_KEY_CANDIDATES)
    duration_key = detect_first_key(output, DURATION_KEY_CANDIDATES)
    if text_key is None:
        raise SystemExit(f"{source_name} row {source_index}: missing transcript field")
    if audio_key is None:
        raise SystemExit(f"{source_name} row {source_index}: missing audio field")
    if duration_key is None:
        raise SystemExit(f"{source_name} row {source_index}: missing duration field")

    text = str(output.get(text_key, "") or "").strip()
    audio_filepath = str(output.get(audio_key, "") or "").strip()
    if not text:
        raise SystemExit(f"{source_name} row {source_index}: empty transcript")
    if not audio_filepath:
        raise SystemExit(f"{source_name} row {source_index}: empty audio path")

    output["text"] = text
    output["audio_filepath"] = audio_filepath
    output["duration"] = float(output[duration_key])
    output["mix_source"] = source_name
    output["mix_source_index"] = int(source_index)
    output["mix_repeat_index"] = int(repeat_index)
    output["mix_id"] = f"{source_name}:{source_index}:repeat:{repeat_index}"
    if "lang" not in output or not str(output.get("lang") or "").strip():
        output["lang"] = "hi"
    return output


def compute_domain_repeat_factor(
    *,
    domain_rows: int,
    vaani_rows: int,
    strategy: str,
    min_fraction: float,
    repeat_cap: int,
) -> tuple[int, bool]:
    if strategy == "pure_additive" or vaani_rows <= 0:
        return 1, True
    if domain_rows <= 0:
        raise SystemExit("domain train manifest must contain at least one row")
    if repeat_cap < 1:
        raise SystemExit("--domain-repeat-cap must be at least 1")

    target = max(0.0, min(float(min_fraction), 0.95))
    factor = 1
    while factor < repeat_cap:
        fraction = (domain_rows * factor) / ((domain_rows * factor) + vaani_rows)
        if fraction >= target:
            return factor, True
        factor += 1
    final_fraction = (domain_rows * factor) / ((domain_rows * factor) + vaani_rows)
    return factor, final_fraction >= target


def repeat_domain_rows(domain_rows: list[dict[str, Any]], *, repeat_factor: int) -> list[dict[str, Any]]:
    repeated: list[dict[str, Any]] = []
    for repeat_index in range(repeat_factor):
        for source_index, row in enumerate(domain_rows):
            repeated.append(
                normalize_to_nemo_row(
                    row,
                    source_name="domain",
                    source_index=source_index,
                    repeat_index=repeat_index,
                )
            )
    return repeated


def prepare_vaani_rows(vaani_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        normalize_to_nemo_row(row, source_name="vaani", source_index=source_index, repeat_index=0)
        for source_index, row in enumerate(vaani_rows)
    ]


def interleave_sources(
    domain_rows: list[dict[str, Any]],
    vaani_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    domain = list(domain_rows)
    vaani = list(vaani_rows)
    rng.shuffle(domain)
    rng.shuffle(vaani)

    total = len(domain) + len(vaani)
    if total == 0:
        return []
    combined: list[dict[str, Any]] = []
    domain_index = 0
    vaani_index = 0
    domain_target_ratio = len(domain) / total

    for position in range(total):
        if domain_index >= len(domain):
            combined.append(vaani[vaani_index])
            vaani_index += 1
            continue
        if vaani_index >= len(vaani):
            combined.append(domain[domain_index])
            domain_index += 1
            continue

        expected_domain_used = (position + 1) * domain_target_ratio
        if domain_index < expected_domain_used:
            combined.append(domain[domain_index])
            domain_index += 1
        else:
            combined.append(vaani[vaani_index])
            vaani_index += 1
    return combined


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: Counter[str] = Counter(str(row.get("mix_source", "unknown")) for row in rows)
    by_lang: Counter[str] = Counter(str(row.get("lang", "unknown") or "unknown") for row in rows)
    duration_by_source: Counter[str] = Counter()
    duration_by_lang: Counter[str] = Counter()
    for row in rows:
        duration = float(row.get("duration") or 0.0)
        duration_by_source[str(row.get("mix_source", "unknown"))] += duration
        duration_by_lang[str(row.get("lang", "unknown") or "unknown")] += duration
    return {
        "rows": len(rows),
        "rows_by_source": dict(sorted(by_source.items())),
        "rows_by_lang": dict(sorted(by_lang.items())),
        "duration_hours_by_source": {
            source: round(seconds / 3600.0, 6)
            for source, seconds in sorted(duration_by_source.items())
        },
        "duration_hours_by_lang": {
            lang: round(seconds / 3600.0, 6)
            for lang, seconds in sorted(duration_by_lang.items())
        },
        "duration_hours_total": round(sum(duration_by_source.values()) / 3600.0, 6),
    }


def build_combined_manifests(
    *,
    domain_train_manifest: Path,
    domain_dev_manifest: Path,
    vaani_train_manifest: Path,
    output_dir: Path,
    strategy: str = "domain_weighted",
    domain_min_row_fraction: float = 0.25,
    domain_repeat_cap: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    domain_train_path = domain_train_manifest.expanduser().resolve()
    domain_dev_path = domain_dev_manifest.expanduser().resolve()
    vaani_train_path = vaani_train_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    domain_train_rows = load_jsonl(domain_train_path)
    domain_dev_rows = load_jsonl(domain_dev_path)
    vaani_train_rows = load_jsonl(vaani_train_path)

    repeat_factor, domain_fraction_target_met = compute_domain_repeat_factor(
        domain_rows=len(domain_train_rows),
        vaani_rows=len(vaani_train_rows),
        strategy=strategy,
        min_fraction=domain_min_row_fraction,
        repeat_cap=domain_repeat_cap,
    )
    repeated_domain_rows = repeat_domain_rows(domain_train_rows, repeat_factor=repeat_factor)
    prepared_vaani_rows = prepare_vaani_rows(vaani_train_rows)
    combined_train_rows = interleave_sources(repeated_domain_rows, prepared_vaani_rows, seed=seed)
    combined_dev_rows = [
        normalize_to_nemo_row(row, source_name="domain_dev", source_index=index, repeat_index=0)
        for index, row in enumerate(domain_dev_rows)
    ]

    train_manifest_path = output_dir / "combined_train.jsonl"
    dev_manifest_path = output_dir / "combined_dev.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(train_manifest_path, combined_train_rows)
    write_jsonl(dev_manifest_path, combined_dev_rows)

    domain_train_count = sum(1 for row in combined_train_rows if row.get("mix_source") == "domain")
    actual_domain_fraction = domain_train_count / len(combined_train_rows) if combined_train_rows else 0.0
    summary = {
        "strategy": strategy,
        "seed": int(seed),
        "domain_min_row_fraction": float(domain_min_row_fraction),
        "domain_repeat_cap": int(domain_repeat_cap),
        "domain_repeat_factor": int(repeat_factor),
        "domain_fraction_target_met": bool(domain_fraction_target_met),
        "actual_domain_row_fraction": round(actual_domain_fraction, 6),
        "inputs": {
            "domain_train_manifest": str(domain_train_path),
            "domain_dev_manifest": str(domain_dev_path),
            "vaani_train_manifest": str(vaani_train_path),
            "domain_train_rows": len(domain_train_rows),
            "domain_dev_rows": len(domain_dev_rows),
            "vaani_train_rows": len(vaani_train_rows),
        },
        "train": summarize_rows(combined_train_rows),
        "dev": summarize_rows(combined_dev_rows),
        "outputs": {
            "output_dir": str(output_dir),
            "combined_train_manifest": str(train_manifest_path),
            "combined_dev_manifest": str(dev_manifest_path),
            "summary_json": str(summary_path),
        },
        "validation_policy": "domain_only",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = build_combined_manifests(
        domain_train_manifest=args.domain_train_manifest,
        domain_dev_manifest=args.domain_dev_manifest,
        vaani_train_manifest=args.vaani_train_manifest,
        output_dir=args.output_dir,
        strategy=args.strategy,
        domain_min_row_fraction=float(args.domain_min_row_fraction),
        domain_repeat_cap=int(args.domain_repeat_cap),
        seed=int(args.seed),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
