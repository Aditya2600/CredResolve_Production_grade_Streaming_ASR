#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


COMMON_DURATION_COLUMNS = (
    "duration",
    "audio_duration",
    "call_duration",
    "duration_seconds",
    "duration_secs",
    "duration_ms",
    "audio_duration_seconds",
    "audio_duration_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show total rows, language counts, and total duration for a CSV file."
    )
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--language-column", default="language", help="Language column name")
    parser.add_argument(
        "--duration-column",
        help="Duration column name. If omitted, common duration columns are auto-detected.",
    )
    parser.add_argument(
        "--audio-url-column",
        default="cr_call_recording_url",
        help="Audio URL column used when no duration column is present",
    )
    parser.add_argument(
        "--duration-unit",
        choices=["auto", "seconds", "milliseconds"],
        default="auto",
        help="How to interpret numeric duration values",
    )
    parser.add_argument(
        "--no-audio-probe",
        action="store_true",
        help="Skip probing audio URLs with ffprobe when no duration column exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel ffprobe workers when probing audio URLs",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="Path to the ffprobe executable",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for each ffprobe call",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def detect_duration_column(fieldnames: list[str], requested: str | None) -> str | None:
    if requested:
        for fieldname in fieldnames:
            if fieldname == requested:
                return fieldname
        raise SystemExit(f"Duration column '{requested}' was not found in the CSV header.")

    normalized = {normalize_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in COMMON_DURATION_COLUMNS:
        fieldname = normalized.get(candidate)
        if fieldname:
            return fieldname
    return None


def parse_clock_duration(value: str) -> float | None:
    parts = value.split(":")
    if not 2 <= len(parts) <= 3:
        return None
    try:
        total = 0.0
        for part in parts:
            total = total * 60 + float(part)
        return total
    except ValueError:
        return None


def parse_duration(value: str | None, unit: str) -> float | None:
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    if ":" in text:
        return parse_clock_duration(text)

    lowered = text.lower()
    cleaned = text.replace(",", "")
    for suffix in ("milliseconds", "millisecond", "ms", "seconds", "second", "secs", "sec", "s"):
        if lowered.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            break

    try:
        numeric = float(cleaned)
    except ValueError:
        return None

    if unit == "milliseconds":
        return numeric / 1000.0
    if unit == "seconds":
        return numeric
    if "ms" in lowered or "millisecond" in lowered:
        return numeric / 1000.0
    if numeric >= 10000:
        return numeric / 1000.0
    return numeric


def format_duration(total_seconds: float) -> str:
    rounded = int(round(total_seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def probe_duration(url: str, ffprobe_bin: str, timeout: float) -> float | None:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        url,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None

    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def probe_audio_durations(
    urls: list[str],
    ffprobe_bin: str,
    workers: int,
    timeout: float,
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(probe_duration, url, ffprobe_bin, timeout): url
            for url in urls
        }
        for future in as_completed(futures):
            url = futures[future]
            results[url] = future.result()
    return results


def load_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("CSV appears to be empty.")

        rows: list[dict[str, str]] = []
        for row in reader:
            if row is None:
                continue
            if all(not (value or "").strip() for value in row.values()):
                continue
            rows.append(row)
        return rows, list(reader.fieldnames)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    rows, fieldnames = load_rows(csv_path)
    total_rows = len(rows)
    language_counts = Counter()
    for row in rows:
        language = (row.get(args.language_column) or "").strip() or "<EMPTY>"
        language_counts[language] += 1

    print(f"File: {csv_path}")
    print(f"Total rows: {total_rows}")
    print("Language breakdown:")
    for language, count in sorted(language_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {language}: {count}")

    duration_column = detect_duration_column(fieldnames, args.duration_column)
    if duration_column:
        total_duration = 0.0
        parsed_rows = 0
        failed_rows = 0
        for row in rows:
            seconds = parse_duration(row.get(duration_column), args.duration_unit)
            if seconds is None:
                if (row.get(duration_column) or "").strip():
                    failed_rows += 1
                continue
            total_duration += seconds
            parsed_rows += 1

        print(f"Duration source: column '{duration_column}'")
        print(
            f"Total duration: {format_duration(total_duration)} "
            f"({total_duration:.2f} seconds)"
        )
        print(f"Rows with parsed duration: {parsed_rows}/{total_rows}")
        if failed_rows:
            print(f"Rows with unparsed duration values: {failed_rows}")
        return 0

    if args.no_audio_probe:
        print("Total duration: unavailable (no duration column found and audio probing disabled)")
        return 0

    if args.audio_url_column not in fieldnames:
        print(
            "Total duration: unavailable "
            f"(no duration column found and audio URL column '{args.audio_url_column}' is missing)"
        )
        return 0

    if shutil.which(args.ffprobe_bin) is None:
        print(
            "Total duration: unavailable "
            f"(no duration column found and ffprobe binary '{args.ffprobe_bin}' is not installed)"
        )
        return 0

    url_counts = Counter()
    empty_url_rows = 0
    for row in rows:
        url = (row.get(args.audio_url_column) or "").strip()
        if url:
            url_counts[url] += 1
        else:
            empty_url_rows += 1

    if not url_counts:
        print(
            "Total duration: unavailable "
            f"(no duration column found and audio URL column '{args.audio_url_column}' is empty)"
        )
        return 0

    print(
        f"Duration source: probed audio URLs from '{args.audio_url_column}' "
        f"using {min(args.workers, len(url_counts))} workers"
    )
    durations = probe_audio_durations(
        list(url_counts),
        ffprobe_bin=args.ffprobe_bin,
        workers=args.workers,
        timeout=args.probe_timeout,
    )

    total_duration = 0.0
    rows_with_duration = 0
    failed_urls = 0
    for url, count in url_counts.items():
        duration = durations.get(url)
        if duration is None:
            failed_urls += 1
            continue
        total_duration += duration * count
        rows_with_duration += count

    print(f"Total duration: {format_duration(total_duration)} ({total_duration:.2f} seconds)")
    print(f"Rows with parsed duration: {rows_with_duration}/{total_rows}")
    if empty_url_rows:
        print(f"Rows missing audio URLs: {empty_url_rows}")
    if failed_urls:
        print(f"URLs that failed to probe: {failed_urls}/{len(url_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
