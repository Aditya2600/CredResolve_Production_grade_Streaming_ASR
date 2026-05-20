#!/usr/bin/env python3
"""Build benchmark and transcript exports from downloaded Vaani WAV manifests.

Input is the manifest written by ``SimpliSmart/download_vaani_sample.py``:
    {"file": "vaani_000.wav", "transcript": "...", "duration_s": 2.25, ...}

Outputs:
  - benchmark JSONL accepted by ``bench_worker_sequential.py``
  - transcript CSV with the Vaani gold transcript and metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Any


LANGUAGE_CODES = {
    "hindi": "hi",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "bengali": "bn",
    "gujarati": "gu",
    "punjabi": "pa",
    "odia": "or",
    "urdu": "ur",
}


def duration_bucket(duration_s: float) -> str:
    if duration_s < 5:
        return "short"
    if duration_s < 15:
        return "medium"
    return "long"


def language_code(value: Any, fallback: str) -> str:
    raw = str(value or fallback or "").strip()
    if not raw:
        return "hi"
    return LANGUAGE_CODES.get(raw.casefold(), raw.casefold())


def read_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono PCM16 WAV")
        return wf.getnframes() / float(wf.getframerate())


def iter_rows(manifest: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{manifest}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Vaani JSONL manifest from download_vaani_sample.py")
    parser.add_argument("--audio-dir", type=Path,
                        help="Directory containing manifest WAVs; defaults to manifest parent")
    parser.add_argument("--bench-out", type=Path, required=True,
                        help="Output JSONL for bench_worker_sequential.py")
    parser.add_argument("--transcripts-out", type=Path, required=True,
                        help="Output CSV containing Vaani gold transcripts")
    parser.add_argument("--utt-prefix", default="vaani",
                        help="Prefix for generated utt_id values")
    parser.add_argument("--language", default="hi",
                        help="Fallback worker language code when manifest language is absent")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    audio_dir = (args.audio_dir or manifest.parent).expanduser().resolve()
    rows = iter_rows(manifest)
    if not rows:
        raise SystemExit(f"{manifest} is empty")

    args.bench_out.parent.mkdir(parents=True, exist_ok=True)
    args.transcripts_out.parent.mkdir(parents=True, exist_ok=True)

    with args.bench_out.open("w", encoding="utf-8") as bench_handle, args.transcripts_out.open(
        "w", encoding="utf-8", newline=""
    ) as transcript_handle:
        transcript_writer = csv.DictWriter(
            transcript_handle,
            fieldnames=[
                "utt_id",
                "wav",
                "language",
                "bucket",
                "duration_s",
                "district",
                "state",
                "transcript",
            ],
        )
        transcript_writer.writeheader()

        for index, row in enumerate(rows, 1):
            wav_path = audio_dir / row["file"]
            duration_s = float(row.get("duration_s") or read_wav_duration(wav_path))
            language = language_code(row.get("language"), args.language)
            utt_id = f"{args.utt_prefix}_{index:03d}"
            bucket = duration_bucket(duration_s)
            transcript = row.get("transcript") or ""

            bench_row = {
                "utt_id": utt_id,
                "language": language,
                "bucket": bucket,
                "wav": str(wav_path.resolve()),
                "transcript": transcript,
                "duration_s": duration_s,
            }
            print(json.dumps(bench_row, ensure_ascii=False), file=bench_handle)

            transcript_writer.writerow({
                "utt_id": utt_id,
                "wav": str(wav_path.resolve()),
                "language": language,
                "bucket": bucket,
                "duration_s": f"{duration_s:.6f}",
                "district": row.get("district") or "",
                "state": row.get("state") or "",
                "transcript": transcript,
            })

    print(f"wrote {len(rows)} benchmark rows: {args.bench_out}")
    print(f"wrote {len(rows)} transcript rows: {args.transcripts_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
