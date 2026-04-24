#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


DEFAULT_INPUTS = (
    "Result_7.csv",
    "Result_11.csv",
    "Result_23.csv",
    "Result_28.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe original audio URLs from one or more CSV files with ffprobe and "
            "report channel counts."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help=(
            "CSV files to inspect. Defaults to Result_7.csv Result_11.csv "
            "Result_23.csv Result_28.csv when omitted."
        ),
    )
    parser.add_argument(
        "--url-column",
        default="cr_call_recording_url",
        help="CSV column containing the original audio URL. Default: cr_call_recording_url",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="Path to ffprobe. Default: ffprobe",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="Timeout for each ffprobe call in seconds. Default: 30",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        help="Optional output CSV path for per-row probe results.",
    )
    return parser.parse_args()


def parse_positive_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def ensure_ffprobe_available(ffprobe_bin: str) -> None:
    if shutil.which(ffprobe_bin):
        return
    raise SystemExit(f"ffprobe binary not found on PATH: {ffprobe_bin}")


def resolve_csv_files(csv_files: list[str]) -> list[Path]:
    if csv_files:
        return [Path(item).expanduser().resolve() for item in csv_files]

    resolved: list[Path] = []
    missing: list[str] = []
    for item in DEFAULT_INPUTS:
        path = Path(item).expanduser().resolve()
        if path.exists():
            resolved.append(path)
        else:
            missing.append(item)

    if resolved:
        return resolved

    raise SystemExit(
        "No CSV files were provided and the default files were not found: "
        + ", ".join(missing)
    )


def probe_audio_channels(
    source: str,
    *,
    ffprobe_bin: str,
    timeout_sec: float,
    cache: dict[str, tuple[int | None, str | None]],
) -> tuple[int | None, str | None]:
    if source in cache:
        return cache[source]

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        source,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        result = (None, "timeout")
        cache[source] = result
        return result
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        result = (None, stderr or "ffprobe_failed")
        cache[source] = result
        return result

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        result = (None, "invalid_ffprobe_json")
        cache[source] = result
        return result

    streams = payload.get("streams") or []
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if audio_stream is None and streams:
        audio_stream = streams[0]

    if audio_stream is None:
        result = (None, "no_audio_stream")
        cache[source] = result
        return result

    channels = parse_positive_int(audio_stream.get("channels"))
    if channels is None:
        result = (None, "missing_channels")
    else:
        result = (channels, None)
    cache[source] = result
    return result


def write_details_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_csv",
        "row_number",
        "row_id",
        "language",
        "audio_source",
        "channels",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    ensure_ffprobe_available(args.ffprobe_bin)
    csv_files = resolve_csv_files(args.csv_files)

    probe_cache: dict[str, tuple[int | None, str | None]] = {}
    detail_rows: list[dict[str, object]] = []

    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"\n{csv_path.name}\n  error: file_not_found", file=sys.stderr)
            continue

        counts: Counter[str] = Counter()
        total_rows = 0
        unique_sources: set[str] = set()

        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                print(f"\n{csv_path.name}\n  error: missing_header", file=sys.stderr)
                continue
            if args.url_column not in reader.fieldnames:
                print(
                    f"\n{csv_path.name}\n  error: missing_column_{args.url_column}",
                    file=sys.stderr,
                )
                continue

            for row_number, row in enumerate(reader, start=2):
                total_rows += 1
                audio_source = str(row.get(args.url_column) or "").strip()
                row_id = str(row.get("#") or "").strip()
                language = str(row.get("language") or "").strip()

                if not audio_source:
                    counts["missing_url"] += 1
                    detail_rows.append(
                        {
                            "input_csv": csv_path.name,
                            "row_number": row_number,
                            "row_id": row_id,
                            "language": language,
                            "audio_source": "",
                            "channels": "",
                            "status": "missing_url",
                            "error": "",
                        }
                    )
                    continue

                unique_sources.add(audio_source)
                channels, error = probe_audio_channels(
                    audio_source,
                    ffprobe_bin=args.ffprobe_bin,
                    timeout_sec=float(args.timeout_sec),
                    cache=probe_cache,
                )

                if channels is None:
                    counts["unknown"] += 1
                    detail_rows.append(
                        {
                            "input_csv": csv_path.name,
                            "row_number": row_number,
                            "row_id": row_id,
                            "language": language,
                            "audio_source": audio_source,
                            "channels": "",
                            "status": "unknown",
                            "error": error or "",
                        }
                    )
                    continue

                counts[f"{channels}_channels"] += 1
                detail_rows.append(
                    {
                        "input_csv": csv_path.name,
                        "row_number": row_number,
                        "row_id": row_id,
                        "language": language,
                        "audio_source": audio_source,
                        "channels": channels,
                        "status": "ok",
                        "error": "",
                    }
                )

        print(f"\n{csv_path.name}")
        print(f"  total_rows: {total_rows}")
        print(f"  unique_audio_sources: {len(unique_sources)}")
        for key in sorted(counts):
            print(f"  {key}: {counts[key]}")

    if args.details_csv is not None:
        write_details_csv(args.details_csv.expanduser().resolve(), detail_rows)
        print(f"\nWrote details CSV to: {args.details_csv.expanduser().resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
