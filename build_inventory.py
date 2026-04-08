#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


NORMALIZED_COLUMNS = [
    "row_id",
    "call_id",
    "audio_url",
    "local_audio_path",
    "wav_audio_path",
    "lender",
    "portfolio",
    "borrower_name",
    "event_date",
    "amount_1",
    "amount_2",
    "amount_3",
    "transcript_source",
    "raw_transcript",
    "normalized_transcript",
    "download_status",
    "audio_convert_status",
    "segmentation_status",
    "alignment_status",
    "quality_tier",
    "split",
    "slice_tags",
]

SOURCE_TO_NORMALIZED = {
    "call_id": "call_sid",
    "audio_url": "cr_recording_url",
    "lender": "lender",
    "portfolio": "institute_name",
    "borrower_name": "name",
    "event_date": "due_date",
    "amount_1": "principle",
    "amount_2": "emi_amount",
    "amount_3": "total_due",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a normalized ASR data inventory from the source CSV, download MP3 audio, "
            "convert it to 16 kHz mono WAV, and export the inventory as CSV and Parquet."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded MP3 files. Default: data/raw",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("data/wav"),
        help="Directory for converted WAV files. Default: data/wav",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for inventory exports. Default: outputs",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for audio downloads. Default: 60",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download attempts per file. Default: 3",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff between download retries. Default: 2.0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and re-convert audio even if the output files already exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity. Default: INFO",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg"):
        return
    raise SystemExit("ffmpeg was not found on PATH. Please install ffmpeg before running this script.")


def read_source_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Input CSV is empty: {path}")
    return df


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_inventory_frame(source_df: pd.DataFrame, raw_dir: Path, wav_dir: Path) -> pd.DataFrame:
    inventory = pd.DataFrame(index=source_df.index)
    inventory["row_id"] = range(1, len(source_df) + 1)

    for normalized_column, source_column in SOURCE_TO_NORMALIZED.items():
        inventory[normalized_column] = source_df[source_column]

    call_ids = source_df["call_sid"].map(normalize_value)
    inventory["local_audio_path"] = call_ids.map(lambda call_id: str(raw_dir / f"{call_id}.mp3"))
    inventory["wav_audio_path"] = call_ids.map(lambda call_id: str(wav_dir / f"{call_id}.wav"))
    inventory["transcript_source"] = ""
    inventory["raw_transcript"] = ""
    inventory["normalized_transcript"] = ""
    inventory["download_status"] = "pending"
    inventory["audio_convert_status"] = "pending"
    inventory["segmentation_status"] = "not_started"
    inventory["alignment_status"] = "not_started"
    inventory["quality_tier"] = "unreviewed"
    inventory["split"] = "unspecified"
    inventory["slice_tags"] = ""

    ordered_columns = NORMALIZED_COLUMNS + list(source_df.columns)
    return pd.concat([inventory[NORMALIZED_COLUMNS], source_df], axis=1)[ordered_columns]


def download_audio(
    session: requests.Session,
    audio_url: str,
    output_path: Path,
    *,
    timeout: int,
    retries: int,
    retry_backoff_seconds: float,
    overwrite: bool,
) -> str:
    if output_path.exists() and not overwrite:
        logging.debug("Skipping download because file exists: %s", output_path)
        return "skipped_existing"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            with session.get(audio_url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with output_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            logging.debug("Downloaded %s -> %s", audio_url, output_path)
            return "success"
        except requests.RequestException as exc:
            logging.warning(
                "Download attempt %s/%s failed for %s: %s",
                attempt,
                retries,
                audio_url,
                exc,
            )
            if output_path.exists():
                output_path.unlink()
            if attempt < retries:
                time.sleep(retry_backoff_seconds * attempt)
            else:
                return "failed"
    return "failed"


def convert_audio_to_wav(mp3_path: Path, wav_path: Path, *, overwrite: bool) -> str:
    if not mp3_path.exists():
        return "missing_input"
    if wav_path.exists() and not overwrite:
        logging.debug("Skipping conversion because file exists: %s", wav_path)
        return "skipped_existing"

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp3_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.debug("Converted %s -> %s", mp3_path, wav_path)
        return "success"
    except subprocess.CalledProcessError as exc:
        logging.error("ffmpeg conversion failed for %s: %s", mp3_path, exc.stderr.strip())
        if wav_path.exists():
            wav_path.unlink()
        return "failed"


def process_audio(inventory_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "build_inventory/1.0"})

    for idx, row in inventory_df.iterrows():
        call_id = normalize_value(row["call_id"])
        audio_url = normalize_value(row["audio_url"])
        mp3_path = Path(normalize_value(row["local_audio_path"]))
        wav_path = Path(normalize_value(row["wav_audio_path"]))

        if not call_id:
            logging.warning("Row %s is missing call_id; skipping audio work.", row["row_id"])
            inventory_df.at[idx, "download_status"] = "missing_call_id"
            inventory_df.at[idx, "audio_convert_status"] = "skipped_missing_call_id"
            continue

        if not audio_url:
            logging.warning("Row %s (%s) is missing audio_url; skipping download.", row["row_id"], call_id)
            inventory_df.at[idx, "download_status"] = "missing_audio_url"
            inventory_df.at[idx, "audio_convert_status"] = "skipped_missing_audio_url"
            continue

        logging.info("Processing row %s/%s for call_id=%s", row["row_id"], len(inventory_df), call_id)
        download_status = download_audio(
            session,
            audio_url,
            mp3_path,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            overwrite=args.overwrite,
        )
        inventory_df.at[idx, "download_status"] = download_status

        if download_status not in {"success", "skipped_existing"}:
            inventory_df.at[idx, "audio_convert_status"] = "skipped_download_failed"
            continue

        inventory_df.at[idx, "audio_convert_status"] = convert_audio_to_wav(
            mp3_path,
            wav_path,
            overwrite=args.overwrite,
        )

    return inventory_df


def save_outputs(inventory_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "data_inventory.csv"
    parquet_path = output_dir / "data_inventory.parquet"
    inventory_df.to_csv(csv_path, index=False)
    inventory_df.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def log_summary(inventory_df: pd.DataFrame) -> None:
    for column in ("download_status", "audio_convert_status", "segmentation_status", "alignment_status"):
        counts = inventory_df[column].value_counts(dropna=False).to_dict()
        logging.info("%s summary: %s", column, counts)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_ffmpeg_available()

    input_path = args.input.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    wav_dir = args.wav_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    logging.info("Reading input CSV from %s", input_path)
    source_df = read_source_csv(input_path)
    inventory_df = build_inventory_frame(source_df, raw_dir=raw_dir, wav_dir=wav_dir)
    logging.info("Loaded %s rows from source CSV", len(inventory_df))

    inventory_df = process_audio(inventory_df, args)
    csv_path, parquet_path = save_outputs(inventory_df, output_dir)
    log_summary(inventory_df)

    logging.info("Wrote inventory CSV to %s", csv_path)
    logging.info("Wrote inventory Parquet to %s", parquet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
