#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


TEXT_KEY_CANDIDATES = ("text", "reference", "normalized_text", "transcript", "sentence")
AUDIO_KEY_CANDIDATES = ("audio_filepath", "audio_path", "audio", "path")
DURATION_KEY_CANDIDATES = ("duration", "audio_duration", "audio_duration_sec")
ID_KEY_CANDIDATES = ("source_id", "id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a held-out manifest by subtracting one or more used manifests from a source manifest. "
            "This is useful when you trained on a cleaned NeMo manifest but want to evaluate on the exact "
            "remaining mined examples that were never used."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True, help="Full source manifest to subtract from.")
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        required=True,
        help="Manifest whose rows should be excluded from the source manifest. Repeatable.",
    )
    parser.add_argument("--output-manifest", type=Path, required=True, help="Path to write the held-out manifest.")
    parser.add_argument(
        "--output-schema",
        choices=("same", "nemo"),
        default="same",
        help="Whether to preserve the source schema or emit a NeMo-friendly schema. Default: same",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional summary JSON path.",
    )
    parser.add_argument(
        "--unmatched-exclude-jsonl",
        type=Path,
        help="Optional JSONL path for exclude rows that could not be matched back to the source manifest.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()

    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for lineno, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{resolved}:{lineno}: expected JSON object rows")
            rows.append(row)
        if not rows:
            raise SystemExit(f"{resolved}: manifest is empty")
        return rows

    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise SystemExit(f"{resolved}: expected a JSON array of objects")
        if not payload:
            raise SystemExit(f"{resolved}: manifest is empty")
        return payload

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = [dict(row) for row in reader]
        if not rows:
            raise SystemExit(f"{resolved}: manifest is empty")
        return rows

    raise SystemExit(f"Unsupported manifest format for {resolved}. Use .jsonl, .json, .csv, or .tsv")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def resolve_audio_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def detect_first_key(row: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in row:
            return key
    return None


def row_identifier_candidates(row: dict[str, Any]) -> list[tuple[str, Any]]:
    matches: list[tuple[str, Any]] = []

    for key in ID_KEY_CANDIDATES:
        value = str(row.get(key, "") or "").strip()
        if value:
            matches.append(("id", value))
            break

    audio_key = detect_first_key(row, AUDIO_KEY_CANDIDATES)
    if audio_key is not None:
        audio_value = resolve_audio_value(row.get(audio_key))
        if audio_value:
            matches.append(("audio", audio_value))

    dataset_index = row.get("dataset_index")
    if dataset_index not in (None, ""):
        matches.append(
            (
                "dataset_index",
                (
                    str(row.get("dataset", "") or "").strip(),
                    str(row.get("dataset_config", "") or "").strip(),
                    str(row.get("split", "") or "").strip(),
                    str(dataset_index),
                ),
            )
        )

    return matches


def convert_to_nemo_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)

    text_key = detect_first_key(output, TEXT_KEY_CANDIDATES)
    if text_key is None:
        raise SystemExit(f"Cannot convert row to NeMo schema because no text key was found: {sorted(output.keys())}")

    audio_key = detect_first_key(output, AUDIO_KEY_CANDIDATES)
    if audio_key is None:
        raise SystemExit(f"Cannot convert row to NeMo schema because no audio key was found: {sorted(output.keys())}")

    duration_key = detect_first_key(output, DURATION_KEY_CANDIDATES)

    output["text"] = str(output.get(text_key, "") or "").strip()
    audio_value = resolve_audio_value(output.get(audio_key))
    if not audio_value:
        raise SystemExit(f"Cannot convert row to NeMo schema because audio path is empty: {row}")
    output["audio_filepath"] = audio_value

    if duration_key is not None and output.get(duration_key) not in (None, ""):
        output["duration"] = float(output[duration_key])

    if "source_id" not in output and str(output.get("id", "") or "").strip():
        output["source_id"] = str(output["id"]).strip()

    return output


def main() -> int:
    args = parse_args()

    source_manifest = args.source_manifest.expanduser().resolve()
    exclude_manifests = [path.expanduser().resolve() for path in args.exclude_manifest]
    output_manifest = args.output_manifest.expanduser().resolve()

    source_rows = load_manifest(source_manifest)
    id_to_index: dict[str, int] = {}
    audio_to_index: dict[str, int] = {}
    dataset_key_to_index: dict[tuple[str, str, str, str], int] = {}

    duplicate_source_ids: Counter[str] = Counter()
    duplicate_source_audio: Counter[str] = Counter()
    duplicate_source_dataset_keys: Counter[str] = Counter()

    for index, row in enumerate(source_rows):
        for kind, value in row_identifier_candidates(row):
            if kind == "id":
                if value in id_to_index:
                    duplicate_source_ids[str(value)] += 1
                else:
                    id_to_index[str(value)] = index
            elif kind == "audio":
                if value in audio_to_index:
                    duplicate_source_audio[str(value)] += 1
                else:
                    audio_to_index[str(value)] = index
            elif kind == "dataset_index":
                if value in dataset_key_to_index:
                    duplicate_source_dataset_keys[json.dumps(value, ensure_ascii=True)] += 1
                else:
                    dataset_key_to_index[value] = index

    excluded_indices: set[int] = set()
    matched_by: Counter[str] = Counter()
    unmatched_rows: list[dict[str, Any]] = []
    exclude_summary: list[dict[str, Any]] = []

    for exclude_manifest in exclude_manifests:
        exclude_rows = load_manifest(exclude_manifest)
        matched_here = 0
        unmatched_here = 0
        for row in exclude_rows:
            matched_index = None
            matched_kind = None
            for kind, value in row_identifier_candidates(row):
                if kind == "id":
                    matched_index = id_to_index.get(str(value))
                elif kind == "audio":
                    matched_index = audio_to_index.get(str(value))
                elif kind == "dataset_index":
                    matched_index = dataset_key_to_index.get(value)
                if matched_index is not None:
                    matched_kind = kind
                    break

            if matched_index is None:
                unmatched_here += 1
                unmatched_rows.append(
                    {
                        "exclude_manifest": str(exclude_manifest),
                        "row": row,
                    }
                )
                continue

            excluded_indices.add(matched_index)
            matched_here += 1
            matched_by[str(matched_kind)] += 1

        exclude_summary.append(
            {
                "exclude_manifest": str(exclude_manifest),
                "rows": len(exclude_rows),
                "matched_rows": matched_here,
                "unmatched_rows": unmatched_here,
            }
        )

    heldout_rows = [row for index, row in enumerate(source_rows) if index not in excluded_indices]
    if args.output_schema == "nemo":
        heldout_rows = [convert_to_nemo_row(row) for row in heldout_rows]

    write_jsonl(output_manifest, heldout_rows)

    if args.unmatched_exclude_jsonl:
        write_jsonl(args.unmatched_exclude_jsonl, unmatched_rows)

    summary = {
        "source_manifest": str(source_manifest),
        "exclude_manifests": [str(path) for path in exclude_manifests],
        "output_manifest": str(output_manifest),
        "output_schema": args.output_schema,
        "source_rows": len(source_rows),
        "excluded_rows": len(excluded_indices),
        "heldout_rows": len(heldout_rows),
        "matched_by": dict(matched_by),
        "exclude_manifest_summary": exclude_summary,
        "unmatched_exclude_rows": len(unmatched_rows),
        "source_identifier_inventory": {
            "ids": len(id_to_index),
            "audio_paths": len(audio_to_index),
            "dataset_index_keys": len(dataset_key_to_index),
        },
        "duplicate_source_identifiers": {
            "ids": sum(duplicate_source_ids.values()),
            "audio_paths": sum(duplicate_source_audio.values()),
            "dataset_index_keys": sum(duplicate_source_dataset_keys.values()),
        },
    }

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
