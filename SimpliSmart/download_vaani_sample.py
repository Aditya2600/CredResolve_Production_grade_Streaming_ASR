#!/usr/bin/env python3
"""
Download 50 random Hindi audio samples from ARTPARK-IISc/Vaani-transcription-part
to a local folder so benchmark_asr.py can stream them concurrently.

pip install datasets soundfile huggingface_hub
huggingface-cli login           # the dataset is gated; accept terms on the HF page first
python3 download_vaani_sample.py
"""

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.indicvoices_dataset import VAANI_DATASET_ID, load_vaani_stream  # noqa: E402


DATASET = VAANI_DATASET_ID
DEFAULT_LANGUAGE = "Hindi"
DEFAULT_COUNT = 50
DEFAULT_OUT = Path(__file__).parent / "vaani_hi_sample"
SHUFFLE_BUFFER_SIZE = 200


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


def resolve_hf_token() -> str | bool:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or load_dotenv_value("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
        or True
    )


def language_to_config(language: str) -> str:
    value = " ".join(str(language or "").strip().split())
    if "/" in value:
        return value
    return f"audio/{value.title()}"


def language_matches(row_value: Any, requested_language: str) -> bool:
    value = str(row_value or "").strip().casefold()
    if not value:
        return True
    requested = str(requested_language or "").strip().casefold()
    config_language = language_to_config(requested_language).rsplit("/", 1)[-1].casefold()
    return value in {requested, config_language}


def shuffled_rows(rows, seed: int, buffer_size: int = SHUFFLE_BUFFER_SIZE):
    rng = random.Random(seed)
    iterator = iter(rows)
    buffer = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break

    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        try:
            buffer[index] = next(iterator)
        except StopIteration:
            buffer.pop(index)


def pick_samples(language: str, count: int, seed: int, min_duration_s: float = 0.0) -> list[dict]:
    """Stream the matching Vaani config and collect language-matched samples.

    If min_duration_s > 0, only rows whose decoded audio is at least that long are kept.
    """
    cfg = language_to_config(language)
    token = resolve_hf_token()
    ds = load_vaani_stream(
        dataset_config=cfg,
        split="train",
        token=token,
        cache_dir=None,
    )

    picked: list[dict] = []
    scanned = 0
    for row in shuffled_rows(ds, seed=seed):
        scanned += 1
        if not language_matches(row.get("language"), language):
            continue
        if min_duration_s > 0:
            try:
                audio, sr = read_audio(row["audio"])
            except Exception:
                continue
            if len(audio) / sr < min_duration_s:
                continue
            row = {**row, "_decoded": (audio, sr)}
        picked.append({"config": cfg, "row": row})
        if len(picked) >= count:
            break
        if scanned % 50 == 0:
            print(f"  {cfg}: scanned {scanned}, kept {len(picked)}/{count}")
    print(f"  {cfg}: collected {len(picked)}/{count} (scanned {scanned})")
    return picked[:count]


def read_audio(audio: Any):
    """Read a datasets Audio payload without requiring torchcodec."""
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            return audio["array"], int(audio.get("sampling_rate") or 16000)
        if audio.get("bytes") is not None:
            return sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if audio.get("path"):
            return sf.read(audio["path"], dtype="float32")
    raise ValueError(f"Unsupported audio payload: {type(audio)!r}")


def save_samples(samples: list[dict], out_dir: Path, target_duration_s: float = 0.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, item in enumerate(samples):
        row = item["row"]
        if "_decoded" in row:
            audio, sampling_rate = row["_decoded"]
        else:
            audio, sampling_rate = read_audio(row["audio"])
        if target_duration_s > 0:
            target_samples = int(target_duration_s * sampling_rate)
            audio = audio[:target_samples]
        wav_path = out_dir / f"vaani_{i:03d}.wav"
        sf.write(wav_path, audio, sampling_rate)
        manifest.append({
            "file": wav_path.name,
            "config": item["config"],
            "language": row.get("language"),
            "district": row.get("district"),
            "state": row.get("state"),
            "transcript": row.get("transcript"),
            "duration_s": len(audio) / sampling_rate,
        })
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {len(samples)} files to {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.jsonl'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", default=DEFAULT_LANGUAGE,
                   help="Row-level language field to match (e.g. Hindi, Marathi).")
    p.add_argument("--count", type=int, default=DEFAULT_COUNT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-duration", type=float, default=0.0,
                   help="Reject clips shorter than this many seconds.")
    p.add_argument("--target-duration", type=float, default=0.0,
                   help="Truncate each saved clip to exactly this many seconds.")
    args = p.parse_args()

    print(f"Sampling {args.count} {args.language} clips from {DATASET}"
          f" (min={args.min_duration}s, target={args.target_duration}s)...")
    samples = pick_samples(args.language, args.count, args.seed, args.min_duration)
    if len(samples) < args.count:
        print(f"WARNING: only found {len(samples)} matching clips.")
    save_samples(samples, args.out, args.target_duration)


if __name__ == "__main__":
    main()
