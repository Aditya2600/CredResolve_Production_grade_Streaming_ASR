#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace public audio URLs with local WAV paths and add annotation columns."
    )
    parser.add_argument("--input-csv", type=Path, default=Path("model_training/batch_transcript-2.csv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("vad_chunks"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("model_training/batch_transcript-2_training.csv"),
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            return round(handle.getnframes() / float(handle.getframerate()), 2)
    except Exception:
        logging.warning("Could not read WAV duration: %s", path)
        return None


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    audio_dir = args.audio_dir.expanduser().resolve()
    rows_written = 0
    missing_audio = 0

    with args.input_csv.open(newline="", encoding="utf-8-sig") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise SystemExit(f"No CSV header found in {args.input_csv}")

        fieldnames = list(reader.fieldnames)
        if "audio_link" not in fieldnames or "filename" not in fieldnames:
            raise SystemExit("CSV must contain filename and audio_link columns")

        if "audio_filepath" not in fieldnames:
            fieldnames.insert(fieldnames.index("audio_link"), "audio_filepath")
        for column in ("duration", "ground_truth", "remarks"):
            if column not in fieldnames:
                fieldnames.append(column)

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                audio_path = audio_dir / (row.get("filename") or "")
                if audio_path.exists():
                    local_path = str(audio_path.resolve())
                    row["audio_filepath"] = local_path
                    row["audio_link"] = local_path
                    row["duration"] = row.get("duration") or wav_duration(audio_path) or ""
                else:
                    missing_audio += 1
                    row["audio_filepath"] = ""
                    row["audio_link"] = ""
                    row["duration"] = row.get("duration") or ""

                row.setdefault("ground_truth", "")
                row.setdefault("remarks", "")
                writer.writerow({column: row.get(column, "") for column in fieldnames})
                rows_written += 1

    logging.info("Wrote %d rows to %s", rows_written, args.output_csv)
    if missing_audio:
        logging.warning("%d rows had no matching local WAV in %s", missing_audio, audio_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
