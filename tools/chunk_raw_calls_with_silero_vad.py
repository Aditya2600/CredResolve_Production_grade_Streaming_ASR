#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
HARD_MIN_CHUNK_SEC = 2.0
IDEAL_MIN_CHUNK_SEC = 8.0
IDEAL_MAX_CHUNK_SEC = 15.0
SMALL_GAP_SEC = 0.8


@dataclass(frozen=True)
class Region:
    start: int
    end: int

    def duration(self, sample_rate: int) -> float:
        return max(0, self.end - self.start) / sample_rate


VadDetector = Callable[[np.ndarray, int], list[dict[str, int]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw stereo calls by channel, run Silero VAD, and export ASR chunks."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("model_training/hindi_25h/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("vad_chunks"))
    parser.add_argument("--manifest-out", type=Path, default=Path("chunk_manifest_raw.jsonl"))
    parser.add_argument("--min-chunk-sec", type=float, default=5.0)
    parser.add_argument("--max-chunk-sec", type=float, default=20.0)
    parser.add_argument("--target-sample-rate", type=int, default=8000)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "call"


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def channel_audio(audio: np.ndarray) -> list[tuple[str, np.ndarray]]:
    if audio.shape[1] < 2:
        logging.warning("Input is mono; processing it as ch0 without mean-mixing")
        return [("ch0", audio[:, 0])]
    if audio.shape[1] > 2:
        logging.warning("Input has %d channels; processing ch0/ch1 only", audio.shape[1])
    return [("ch0", audio[:, 0]), ("ch1", audio[:, 1])]


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate == target_rate:
        return audio
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    dst_len = max(1, int(round(audio.shape[0] * (target_rate / source_rate))))
    return np.interp(
        np.linspace(0.0, audio.shape[0] - 1, num=dst_len, dtype=np.float32),
        np.arange(audio.shape[0], dtype=np.float32),
        audio,
    ).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")


def build_silero_detector(max_chunk_sec: float) -> VadDetector:
    logging.info("Loading Silero VAD")
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit("Install torch and silero-vad before running this script.") from exc

    torch.set_num_threads(1)
    model = load_silero_vad()

    def detect(audio: np.ndarray, sample_rate: int) -> list[dict[str, int]]:
        timestamps = get_speech_timestamps(
            torch.from_numpy(np.asarray(audio, dtype=np.float32).copy()),
            model,
            sampling_rate=int(sample_rate),
            threshold=0.5,
            min_speech_duration_ms=250,
            max_speech_duration_s=float(max_chunk_sec),
            min_silence_duration_ms=150,
            speech_pad_ms=80,
            return_seconds=False,
        )
        if hasattr(model, "reset_states"):
            model.reset_states()
        return [dict(item) for item in timestamps]

    return detect


def regions_from_timestamps(timestamps: list[dict[str, int]]) -> list[Region]:
    regions: list[Region] = []
    for item in timestamps:
        start, end = int(item.get("start", 0) or 0), int(item.get("end", 0) or 0)
        if end > start:
            regions.append(Region(start, end))
    return regions


def split_long_regions(regions: list[Region], sample_rate: int, max_sec: float) -> list[Region]:
    max_samples = int(max_sec * sample_rate)
    split: list[Region] = []
    for region in regions:
        start = region.start
        while region.end - start > max_samples:
            split.append(Region(start, start + max_samples))
            start += max_samples
        split.append(Region(start, region.end))
    return split


def chunk_regions(
    regions: list[Region],
    sample_rate: int,
    *,
    min_chunk_sec: float,
    max_chunk_sec: float,
) -> list[Region]:
    regions = split_long_regions(sorted(regions, key=lambda item: item.start), sample_rate, max_chunk_sec)
    chunks: list[Region] = []
    current: Region | None = None
    small_gap = int(SMALL_GAP_SEC * sample_rate)

    for region in regions:
        if current is None:
            current = region
            continue
        gap = region.start - current.end
        merged = Region(current.start, region.end)
        merged_sec = merged.duration(sample_rate)
        current_sec = current.duration(sample_rate)
        should_merge = (
            merged_sec <= max_chunk_sec
            and gap <= small_gap
            and (current_sec < IDEAL_MIN_CHUNK_SEC or merged_sec <= IDEAL_MAX_CHUNK_SEC)
        )
        if should_merge:
            current = merged
        else:
            chunks.append(current)
            current = region

    if current is not None:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.duration(sample_rate) >= min_chunk_sec]


def speech_coverage(chunk: Region, speech_regions: list[Region]) -> float:
    covered = 0
    for region in speech_regions:
        covered += max(0, min(chunk.end, region.end) - max(chunk.start, region.start))
    total = max(1, chunk.end - chunk.start)
    return covered / total


def usable_chunk(audio: np.ndarray, coverage: float, min_chunk_sec: float, sample_rate: int) -> bool:
    duration = len(audio) / sample_rate if sample_rate else 0.0
    if duration < max(HARD_MIN_CHUNK_SEC, min_chunk_sec) or coverage < 0.2:
        return False
    if audio.size == 0 or not np.isfinite(audio).all():
        return False
    abs_audio = np.abs(audio)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))
    peak = float(np.max(abs_audio))
    clipped = float(np.mean(abs_audio > 0.98))
    # ponytail: conservative quality gate; replace with a scored denoiser/SNR model if this drops useful calls.
    return peak >= 1e-3 and rms >= 2e-4 and rms <= 0.5 and clipped <= 0.10


def process_file(
    path: Path,
    *,
    output_dir: Path,
    detector: VadDetector,
    target_sample_rate: int,
    min_chunk_sec: float,
    max_chunk_sec: float,
) -> list[dict[str, object]]:
    try:
        audio, source_rate = read_audio(path)
    except Exception:
        logging.exception("Skipping unreadable/corrupted audio: %s", path)
        return []

    call_id = sanitize_id(path.stem)
    manifest_rows: list[dict[str, object]] = []
    for channel, values in channel_audio(audio):
        mono = resample(values, source_rate, target_sample_rate)
        channel_path = output_dir / f"{call_id}_{channel}.wav"
        write_wav(channel_path, mono, target_sample_rate)

        try:
            speech_regions = regions_from_timestamps(detector(mono, target_sample_rate))
        except Exception:
            logging.exception("VAD failed for %s %s", path, channel)
            continue

        chunks = chunk_regions(
            speech_regions,
            target_sample_rate,
            min_chunk_sec=min_chunk_sec,
            max_chunk_sec=max_chunk_sec,
        )
        written = 0
        for index, chunk in enumerate(chunks, start=1):
            chunk_audio = mono[chunk.start : chunk.end]
            coverage = speech_coverage(chunk, speech_regions)
            if not usable_chunk(chunk_audio, coverage, min_chunk_sec, target_sample_rate):
                continue

            chunk_path = output_dir / f"{call_id}_{channel}_seg_{index:04d}.wav"
            write_wav(chunk_path, chunk_audio, target_sample_rate)
            duration = len(chunk_audio) / target_sample_rate
            manifest_rows.append(
                {
                    "audio_filepath": chunk_path.as_posix(),
                    "duration": round(duration, 2),
                    "call_id": call_id,
                    "channel": channel,
                    "start_time": round(chunk.start / target_sample_rate, 2),
                    "end_time": round(chunk.end / target_sample_rate, 2),
                }
            )
            written += 1
        logging.info("%s %s: %d chunks", path.name, channel, written)

    return manifest_rows


def iter_audio_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in AUDIO_SUFFIXES)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    if args.min_chunk_sec < HARD_MIN_CHUNK_SEC:
        raise SystemExit("--min-chunk-sec must be at least 2.0")
    if args.max_chunk_sec < args.min_chunk_sec:
        raise SystemExit("--max-chunk-sec must be >= --min-chunk-sec")
    if args.target_sample_rate not in {8000, 16000}:
        raise SystemExit("--target-sample-rate must be 8000 or 16000 for Silero VAD")
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = build_silero_detector(args.max_chunk_sec)
    files = iter_audio_files(args.input_dir)
    logging.info("Processing %d audio files from %s", len(files), args.input_dir)

    rows: list[dict[str, object]] = []
    for path in files:
        rows.extend(
            process_file(
                path,
                output_dir=args.output_dir,
                detector=detector,
                target_sample_rate=args.target_sample_rate,
                min_chunk_sec=args.min_chunk_sec,
                max_chunk_sec=args.max_chunk_sec,
            )
        )

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logging.info("Wrote %d chunks to %s and manifest %s", len(rows), args.output_dir, args.manifest_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
