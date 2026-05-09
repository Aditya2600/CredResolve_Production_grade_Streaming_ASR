#!/usr/bin/env python3
"""Materialize 4 deterministic Vaani Hindi fixtures for the audio bench.

Streams ARTPARK-IISc/Vaani-transcription-part (audio/Hindi, train), picks the
first ``N=4`` clips with duration in [MIN_S, MAX_S] under a fixed reservoir
seed, decodes to 16 kHz mono PCM16 WAV, and writes:

  01_<id>.wav, 02_<id>.wav, 03_<id>.wav, 04_<id>.wav
  reference_transcripts.json
  ATTRIBUTION.md  (CC-BY-4.0 attribution, source pointer, snapshot pin)

This is a one-time fixture-prep step (not part of the bench loop). Re-run only
when you want to refresh the snapshot. The seed + filter make the selection
deterministic across machines, given the same Vaani snapshot.

Requires: HUGGINGFACE_HUB_TOKEN with access to the gated Vaani dataset.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.indicvoices_dataset import (  # noqa: E402
    VAANI_DATASET_ID,
    VAANI_HINDI_CONFIG,
    load_vaani_stream,
)
from tools.eval_indicvoices_wer import (  # noqa: E402
    audio_to_float32,
    detect_audio_field,
    detect_text_field,
    resample_linear,
    to_mono,
)
from tools.asr_text_normalizer import normalize_asr_text  # noqa: E402


FIXTURES_DIR = Path(__file__).resolve().parent
TARGET_SR = 16000
N_FIXTURES = 4
MIN_DURATION_S = 2.0
MAX_DURATION_S = 5.0
SEED = 20250507  # Fixed; bump only if you want a new snapshot.
SCAN_LIMIT = 400  # How many candidates to inspect before giving up.

_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_id_fragment(text: str, max_len: int = 24) -> str:
    cleaned = _FILENAME_SAFE.sub("_", text).strip("_")
    return (cleaned[:max_len] or "clip").lower()


def _candidate_id(sample: dict, audio_field: str, idx: int) -> str:
    for key in ("id", "utt_id", "utterance_id", "sample_id", "filename", "file"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return _safe_id_fragment(Path(value).stem)

    audio_value = sample.get(audio_field)
    if isinstance(audio_value, dict):
        path = audio_value.get("path")
        if isinstance(path, str) and path:
            return _safe_id_fragment(Path(path).stem)

    return f"vaani_{idx:04d}"


def _existing_wavs(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.iterdir() if p.is_file() and p.suffix == ".wav")


def _write_pcm16_wav(path: Path, samples_int16: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int16.tobytes())


def _to_pcm16(audio_f32: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio_f32, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", default=None, help="HF token (default: $HUGGINGFACE_HUB_TOKEN)")
    parser.add_argument("--n", type=int, default=N_FIXTURES)
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION_S)
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION_S)
    parser.add_argument("--scan-limit", type=int, default=SCAN_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="HF cache dir (defaults to env HF_HOME or ~/.cache/huggingface)")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Don't delete existing 0X_*.wav before fetching.")
    args = parser.parse_args()

    token = args.token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("error: HUGGINGFACE_HUB_TOKEN not set", file=sys.stderr)
        return 2

    cache_dir = args.cache_dir or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.keep_existing:
        for wav in _existing_wavs(FIXTURES_DIR):
            wav.unlink()

    print(f"Streaming {VAANI_DATASET_ID} ({VAANI_HINDI_CONFIG}, train) ...")
    stream = load_vaani_stream(token=token, cache_dir=cache_dir)

    # Reservoir-style: scan up to scan_limit candidates that pass the duration
    # filter, then pick N indices via seeded random.sample. Deterministic given
    # the same Vaani snapshot ordering.
    candidates: list[tuple[int, dict, np.ndarray, int, float, str, str]] = []
    audio_field: str | None = None
    text_field: str | None = None

    for raw_idx, sample in enumerate(stream):
        if raw_idx >= args.scan_limit:
            break
        if audio_field is None:
            audio_field = detect_audio_field(sample)
        if text_field is None:
            text_field = detect_text_field(sample)
        text_value_raw = sample.get(text_field) or ""
        if not isinstance(text_value_raw, str):
            continue
        text_value = normalize_asr_text(text_value_raw)
        if not text_value:
            continue
        try:
            audio_f32, src_sr = audio_to_float32(sample[audio_field])
        except Exception as exc:
            print(f"  skip {raw_idx}: decode failed ({exc})", file=sys.stderr)
            continue
        audio_f32 = to_mono(audio_f32)
        duration_s = audio_f32.shape[0] / float(src_sr)
        if not (args.min_duration <= duration_s <= args.max_duration):
            continue
        clip_id = _candidate_id(sample, audio_field, raw_idx)
        resampled = resample_linear(audio_f32, src_sr, TARGET_SR)
        candidates.append(
            (raw_idx, sample, resampled, TARGET_SR, duration_s, clip_id, text_value)
        )

    if len(candidates) < args.n:
        print(
            f"error: only {len(candidates)} candidates passed the duration "
            f"filter [{args.min_duration}, {args.max_duration}]s in the first "
            f"{args.scan_limit} samples; bump --scan-limit",
            file=sys.stderr,
        )
        return 3

    rng = random.Random(args.seed)
    chosen_indices = sorted(rng.sample(range(len(candidates)), args.n))
    chosen = [candidates[i] for i in chosen_indices]

    references: dict[str, str] = {}
    attribution_rows: list[dict] = []
    for slot, (raw_idx, sample, audio_f32, sr, duration_s, clip_id, text) in enumerate(chosen, start=1):
        wav_name = f"{slot:02d}_{clip_id}.wav"
        wav_path = FIXTURES_DIR / wav_name
        _write_pcm16_wav(wav_path, _to_pcm16(audio_f32), sr)
        references[wav_name] = text
        attribution_rows.append(
            {
                "wav": wav_name,
                "vaani_index": raw_idx,
                "vaani_id": clip_id,
                "duration_s": round(duration_s, 3),
                "sample_rate": sr,
                "transcript": text,
            }
        )
        print(f"  -> {wav_name}  ({duration_s:.2f}s)  {text[:60]!r}")

    references_path = FIXTURES_DIR / "reference_transcripts.json"
    with open(references_path, "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {references_path}")

    snapshot_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    attribution_md = _render_attribution(
        snapshot_iso=snapshot_iso,
        seed=args.seed,
        scan_limit=args.scan_limit,
        n=args.n,
        min_dur=args.min_duration,
        max_dur=args.max_duration,
        rows=attribution_rows,
    )
    attribution_path = FIXTURES_DIR / "ATTRIBUTION.md"
    with open(attribution_path, "w", encoding="utf-8") as f:
        f.write(attribution_md)
    print(f"Wrote {attribution_path}")
    return 0


def _render_attribution(
    *,
    snapshot_iso: str,
    seed: int,
    scan_limit: int,
    n: int,
    min_dur: float,
    max_dur: float,
    rows: list[dict],
) -> str:
    table_header = "| WAV | Vaani index | Duration (s) | Transcript |\n|---|---|---|---|\n"
    table_rows = "".join(
        f"| `{r['wav']}` | {r['vaani_index']} | {r['duration_s']} | {r['transcript']} |\n"
        for r in rows
    )
    return (
        "# Audio bench fixtures — attribution\n\n"
        "These four WAV files are excerpts from the **Vaani** speech corpus by\n"
        "ARTPARK @ IISc Bangalore, redistributed under the original\n"
        "[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.\n\n"
        "## Source\n\n"
        f"- Dataset: `{VAANI_DATASET_ID}` (HuggingFace Hub, gated)\n"
        f"- Config: `{VAANI_HINDI_CONFIG}`\n"
        f"- Split: `train`\n"
        f"- Snapshot fetched: {snapshot_iso} (UTC)\n"
        f"- Selection: deterministic, seed={seed}, scan_limit={scan_limit}, n={n}, "
        f"duration ∈ [{min_dur}, {max_dur}] s\n"
        "- Audio decoded to 16 kHz mono PCM16 WAV (no other processing).\n\n"
        "## Citation\n\n"
        "ARTPARK-IISc, *Vaani — A bilingual, code-mixed Indian speech dataset*. "
        "https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part\n\n"
        "## Selected clips\n\n"
        + table_header
        + table_rows
        + "\nRefer to `_fetch_vaani_fixtures.py` to regenerate this set.\n"
    )


if __name__ == "__main__":
    sys.exit(main())
