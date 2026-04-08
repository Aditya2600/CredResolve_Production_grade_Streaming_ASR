#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import soundfile as sf


LANGUAGE_ID_MAP = {
    "assamese": "as",
    "bengali": "bn",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "odia": "or",
    "oriya": "or",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a mined training manifest into combined and per-bucket train/dev manifests "
            "that are ready for downstream ASR fine-tuning."
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True, help="Input mined manifest JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write bucket manifests into.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for train/dev splits. Default: 0.1",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed. Default: 42")
    parser.add_argument(
        "--bucket-membership",
        choices=("matched", "primary"),
        default="matched",
        help=(
            "How to assign rows to bucket-specific manifests. "
            "`matched` includes any row whose matched_buckets contain the bucket. "
            "`primary` uses only the row's primary_bucket. Default: matched"
        ),
    )
    parser.add_argument(
        "--include-normal-general",
        action="store_true",
        help="Also write per-bucket train/dev manifests for the normal_general bucket.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if "audio_path" not in row or "reference" not in row:
            raise SystemExit(f"{path}:{lineno}: row must include `audio_path` and `reference`")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: manifest is empty")
    return rows


def build_training_row(row: dict[str, Any], *, bucket_label: str) -> dict[str, Any]:
    audio_path = Path(str(row["audio_path"])).expanduser().resolve()
    info = sf.info(str(audio_path))
    dataset_config = str(row.get("dataset_config", "")).strip().lower()
    lang = str(row.get("lang", "")).strip() or LANGUAGE_ID_MAP.get(dataset_config, dataset_config)
    return {
        "audio_filepath": str(audio_path),
        "duration": float(info.duration),
        "text": str(row["reference"]).strip(),
        "lang": lang,
        "bucket": bucket_label,
        "source_id": str(row.get("id", "")),
        "source": str(row.get("source", "")),
        "primary_bucket": str(row.get("primary_bucket", "")),
        "matched_buckets": list(row.get("matched_buckets", [])),
    }


def split_rows(rows: list[dict[str, Any]], *, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) <= 1 or val_ratio <= 0.0:
        return rows, []
    dev_count = max(1, int(round(len(rows) * val_ratio)))
    dev_count = min(dev_count, len(rows) - 1)
    train_count = len(rows) - dev_count
    return rows[:train_count], rows[train_count:]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def sanitize_bucket_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def main() -> int:
    args = parse_args()

    input_manifest = args.input_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_manifest)
    rng = random.Random(args.seed)

    all_training_rows = [build_training_row(row, bucket_label=str(row.get("primary_bucket", ""))) for row in rows]
    rng.shuffle(all_training_rows)
    all_train, all_dev = split_rows(all_training_rows, val_ratio=max(0.0, min(args.val_ratio, 0.5)))
    write_jsonl(output_dir / "all_train.jsonl", all_train)
    write_jsonl(output_dir / "all_dev.jsonl", all_dev)

    bucket_names: set[str] = set()
    for row in rows:
        if args.bucket_membership == "matched":
            bucket_names.update(str(bucket) for bucket in row.get("matched_buckets", []) if bucket)
        else:
            primary_bucket = str(row.get("primary_bucket", "")).strip()
            if primary_bucket:
                bucket_names.add(primary_bucket)

    if not args.include_normal_general:
        bucket_names.discard("normal_general")

    bucket_summary: dict[str, dict[str, int]] = {}
    for bucket in sorted(bucket_names):
        bucket_rows: list[dict[str, Any]] = []
        for row in rows:
            primary_bucket = str(row.get("primary_bucket", "")).strip()
            matched_buckets = [str(value) for value in row.get("matched_buckets", []) if value]
            if args.bucket_membership == "matched":
                if bucket not in matched_buckets:
                    continue
            elif primary_bucket != bucket:
                continue
            bucket_rows.append(build_training_row(row, bucket_label=bucket))

        rng.shuffle(bucket_rows)
        bucket_train, bucket_dev = split_rows(bucket_rows, val_ratio=max(0.0, min(args.val_ratio, 0.5)))
        stem = sanitize_bucket_name(bucket)
        write_jsonl(output_dir / f"{stem}_train.jsonl", bucket_train)
        write_jsonl(output_dir / f"{stem}_dev.jsonl", bucket_dev)
        bucket_summary[bucket] = {
            "total": len(bucket_rows),
            "train": len(bucket_train),
            "dev": len(bucket_dev),
        }

    summary = {
        "input_manifest": str(input_manifest),
        "output_dir": str(output_dir),
        "val_ratio": max(0.0, min(args.val_ratio, 0.5)),
        "seed": args.seed,
        "bucket_membership": args.bucket_membership,
        "all": {
            "total": len(all_training_rows),
            "train": len(all_train),
            "dev": len(all_dev),
        },
        "buckets": bucket_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
