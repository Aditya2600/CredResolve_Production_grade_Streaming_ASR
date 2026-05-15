#!/usr/bin/env python3
"""Fetch deterministic random Vaani WAVs for WebSocket barrier-burst tests.

The output is intentionally shaped for ``tools/ws_burst_barrier.py``:

- mono PCM16 WAV
- 16 kHz
- exactly ``--target-duration`` seconds per clip
- one JSONL manifest beside the audio files

Selection uses reservoir sampling over eligible rows in the scanned Vaani
stream, so the result is random but reproducible for a given dataset snapshot,
seed, and scan limit.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.indicvoices_dataset import (  # noqa: E402
    VAANI_DATASET_ID,
    VAANI_HINDI_CONFIG,
    load_vaani_stream,
)


DEFAULT_COUNT = 50
DEFAULT_SEED = 20260515
DEFAULT_SCAN_LIMIT = 20_000
DEFAULT_TARGET_DURATION_S = 10.0
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_OUT_DIR = Path("artifacts/vaani_hindi_10s_c50_seed20260515")


@dataclass
class Candidate:
    source_index: int
    row: dict[str, Any]
    audio_f32: np.ndarray
    source_sample_rate: int
    source_duration_s: float


def load_dotenv_value(*keys: str) -> str | None:
    for dotenv_path in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        if not dotenv_path.exists():
            continue
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in keys:
                return value.strip().strip("\"'")
    return None


def resolve_hf_token(cli_token: str | None) -> str:
    token = (
        cli_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or load_dotenv_value("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        raise SystemExit(
            "Vaani access requires --hf-token or HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
        )
    return token


def read_audio(audio: Any) -> tuple[np.ndarray, int]:
    if not isinstance(audio, dict):
        raise ValueError(f"Unsupported audio payload: {type(audio)!r}")
    if audio.get("array") is not None:
        return np.asarray(audio["array"], dtype=np.float32), int(audio.get("sampling_rate") or 16_000)
    if audio.get("bytes") is not None:
        samples, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
    if audio.get("path"):
        samples, sample_rate = sf.read(audio["path"], dtype="float32")
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
    raise ValueError("Audio payload has no array, bytes, or path")


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1, dtype=np.float32)
    raise ValueError(f"Unsupported audio shape: {audio.shape!r}")


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    target_len = round(len(audio) * target_rate / source_rate)
    old_positions = np.arange(len(audio), dtype=np.float64)
    new_positions = np.linspace(0, len(audio) - 1, target_len, dtype=np.float64)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(to_pcm16(audio).tobytes())


def find_audio_field(row: dict[str, Any]) -> str:
    for key, value in row.items():
        if isinstance(value, dict) and any(token in value for token in ("array", "bytes", "path")):
            return key
    raise ValueError(f"No audio field found in row keys: {sorted(row)}")


def choose_candidates(
    *,
    token: str,
    count: int,
    seed: int,
    scan_limit: int,
    min_duration_s: float,
    cache_dir: Path | None,
) -> tuple[list[Candidate], int]:
    rng = random.Random(seed)
    stream = load_vaani_stream(token=token, cache_dir=cache_dir)
    chosen: list[Candidate] = []
    eligible_seen = 0

    for source_index, row in enumerate(stream):
        if source_index >= scan_limit:
            break
        try:
            audio_field = find_audio_field(row)
            audio_f32, source_rate = read_audio(row[audio_field])
            audio_f32 = to_mono(audio_f32)
        except Exception as exc:
            print(f"[skip {source_index}] decode failed: {exc}", file=sys.stderr)
            continue

        source_duration_s = len(audio_f32) / float(source_rate)
        if source_duration_s + 1e-9 < min_duration_s:
            continue

        candidate = Candidate(
            source_index=source_index,
            row=row,
            audio_f32=audio_f32,
            source_sample_rate=source_rate,
            source_duration_s=source_duration_s,
        )
        eligible_seen += 1
        if len(chosen) < count:
            chosen.append(candidate)
            continue

        replacement_index = rng.randrange(eligible_seen)
        if replacement_index < count:
            chosen[replacement_index] = candidate

    if len(chosen) < count:
        raise SystemExit(
            f"Only found {len(chosen)} eligible clips >= {min_duration_s:.3f}s "
            f"within the first {scan_limit} rows."
        )

    rng.shuffle(chosen)
    return chosen, eligible_seen


def clip_exact_duration(audio: np.ndarray, sample_rate: int, target_duration_s: float) -> np.ndarray:
    target_samples = round(target_duration_s * sample_rate)
    if len(audio) < target_samples:
        raise ValueError(
            f"Clip has only {len(audio)} samples after resampling; need {target_samples}"
        )
    return audio[:target_samples]


def build_manifest_row(
    *,
    path: Path,
    slot: int,
    candidate: Candidate,
    saved_duration_s: float,
) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = candidate.row
    return {
        "file": path.name,
        "slot": slot,
        "dataset": VAANI_DATASET_ID,
        "config": VAANI_HINDI_CONFIG,
        "split": "train",
        "source_index": candidate.source_index,
        "language": row.get("language"),
        "district": row.get("district"),
        "state": row.get("state"),
        "transcript": row.get("transcript"),
        "source_sample_rate": candidate.source_sample_rate,
        "source_duration_s": round(candidate.source_duration_s, 6),
        "saved_duration_s": round(saved_duration_s, 6),
        "sha256": digest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scan-limit", type=int, default=DEFAULT_SCAN_LIMIT)
    parser.add_argument("--target-duration", type=float, default=DEFAULT_TARGET_DURATION_S)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hf-token", help="Optional Hugging Face token override.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    if args.target_duration <= 0:
        raise SystemExit("--target-duration must be > 0")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be > 0")

    token = resolve_hf_token(args.hf_token)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, eligible_seen = choose_candidates(
        token=token,
        count=args.count,
        seed=args.seed,
        scan_limit=args.scan_limit,
        min_duration_s=args.target_duration,
        cache_dir=args.cache_dir,
    )

    manifest: list[dict[str, Any]] = []
    for slot, candidate in enumerate(candidates):
        resampled = resample_linear(
            candidate.audio_f32,
            candidate.source_sample_rate,
            args.sample_rate,
        )
        clipped = clip_exact_duration(resampled, args.sample_rate, args.target_duration)
        wav_path = out_dir / f"vaani_{slot:03d}.wav"
        write_pcm16_wav(wav_path, clipped, args.sample_rate)
        manifest.append(
            build_manifest_row(
                path=wav_path,
                slot=slot,
                candidate=candidate,
                saved_duration_s=len(clipped) / args.sample_rate,
            )
        )

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest) + "\n",
        encoding="utf-8",
    )

    print(f"dataset={VAANI_DATASET_ID}")
    print(f"config={VAANI_HINDI_CONFIG}")
    print(f"scanned_rows={args.scan_limit}")
    print(f"eligible_seen={eligible_seen}")
    print(f"saved_clips={len(manifest)}")
    print(f"duration_s={args.target_duration:.3f}")
    print(f"sample_rate={args.sample_rate}")
    print(f"out_dir={out_dir}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
