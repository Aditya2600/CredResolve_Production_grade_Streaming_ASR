#!/usr/bin/env python3
"""Materialize a public Hindi ASR benchmark into a NeMo JSON-lines manifest.

Writes each clip as a 16 kHz mono PCM16 WAV and emits standard NeMo rows
``{"audio_filepath": <abs wav>, "duration": <sec>, "text": <raw gold>}`` so that
both Nemotron 3.5 ASR (via NeMo's cache-aware streaming infer script) and the
production IndicConformer (via tools/eval_nemo_manifest_wer.py) can consume the
*same* manifest. Gold text is kept RAW; scoring normalizes both sides identically
in tools/benchmarks/score_nemotron_manifest.py.

Sources:
  --source vaani      : ARTPARK-IISc/Vaani-transcription-part (config audio/Hindi) -> the repo's
                        in-domain Hindi anchor (GATED: needs --hf-token / $HF_TOKEN). Reuses the
                        explicit-parquet streaming loader in tools/indicvoices_dataset.py. Huge ->
                        pass --limit (e.g. 500). Audio arrives as encoded bytes (decoded here).
  --source contextual : sarvamai/contextual_asr_benchmark (config hi-IN)           -> apples-to-apples.
  --source fleurs     : google/fleurs (config hi_in, split test)                   -> clean read-speech
                        anchor to reproduce Nemotron's published ~6.81% WER.

The 16 kHz linear resample mirrors the production path (worker/app/model.py
`_prepare_model_audio` -> np.interp) so Nemotron sees the same signal IndicConformer does.

Run in the isolated Nemotron env OR the production env (needs numpy/soundfile/datasets;
vaani also needs huggingface_hub + pyarrow). After building, gate with tools/validate_nemo_manifest_audio.py.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = None  # bound lazily in main(); keeps `--help` working without numpy/soundfile/datasets installed


def configure_hf_cache_env() -> None:
    """Keep HF caches under the repo .cache, matching the other benchmark tools."""
    cache_root = (Path.cwd() / ".cache").resolve()
    hf_home = cache_root / "huggingface"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))


def _load_dotenv(*paths) -> None:
    """Minimal .env loader (no python-dotenv dependency): KEY=VALUE lines; real env vars win."""
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if sep and key and val and key not in os.environ:
                os.environ[key] = val


_load_dotenv(REPO_ROOT / ".env", Path.cwd() / ".env")
configure_hf_cache_env()


# --- audio helpers: identical math to worker/app/model.py + contextual_asr_hindi_benchmark.py
def to_mono(audio):
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)


def resample_linear(audio, src_sr: int, dst_sr: int):
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    src_positions = np.arange(audio.shape[0], dtype=np.float32)
    dst_length = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def extract_audio(audio):
    """Return (float32 samples, sample_rate) from a HF audio cell.

    Handles decoded dicts ({"array","sampling_rate"}), encoded dicts ({"bytes"/"path"}),
    raw bytes, and path strings — covers both `load_dataset` (decoded) and the Vaani
    explicit-parquet path (encoded bytes).
    """
    import soundfile as sf

    if isinstance(audio, dict):
        if audio.get("array") is not None:
            return np.asarray(audio["array"], dtype=np.float32), int(audio.get("sampling_rate") or 0)
        data = audio.get("bytes")
        if data:
            arr, sr = sf.read(io.BytesIO(data), dtype="float32")
            return arr, int(sr)
        path = audio.get("path")
        if path and os.path.exists(path):
            arr, sr = sf.read(path, dtype="float32")
            return arr, int(sr)
        return None
    if isinstance(audio, (bytes, bytearray)):
        arr, sr = sf.read(io.BytesIO(audio), dtype="float32")
        return arr, int(sr)
    if isinstance(audio, str) and os.path.exists(audio):
        arr, sr = sf.read(audio, dtype="float32")
        return arr, int(sr)
    return None


# --- load_dataset-style sources (decoded rows with audio={"array","sampling_rate"})
SOURCES: dict[str, dict[str, object]] = {
    "contextual": {
        "dataset": "sarvamai/contextual_asr_benchmark",
        "config": "hi-IN",
        "split": None,  # use the first available split
        "ref_keys": ("transcript", "text", "sentence"),
    },
    "fleurs": {
        "dataset": "google/fleurs",
        "config": "hi_in",
        "split": "test",
        "ref_keys": ("raw_transcription", "transcription", "text"),
    },
}

VAANI_REF_KEYS = ("transcript", "text", "transcription", "sentence", "normalized_text")
AUDIO_CELL_KEYS = ("audio", "audio_filepath", "wav", "path")


def first_present(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def load_rows(args):
    if args.source == "vaani":
        from tools.indicvoices_dataset import load_vaani_stream

        if not args.hf_token:
            raise SystemExit("vaani is gated — pass --hf-token or set $HF_TOKEN.")
        print(f"Loading ARTPARK-IISc/Vaani-transcription-part (config={args.vaani_config}, split={args.vaani_split}) ...", flush=True)
        ds = load_vaani_stream(
            dataset_config=args.vaani_config,
            split=args.vaani_split,
            token=args.hf_token,
            cache_dir=None,
        )
        return ds, VAANI_REF_KEYS

    from datasets import load_dataset

    spec = SOURCES[args.source]
    print(f"Loading {spec['dataset']} (config={spec['config']}, split={spec['split'] or 'first'}) ...", flush=True)
    kwargs = {"token": args.hf_token} if args.hf_token else {}
    if spec["split"]:
        ds = load_dataset(spec["dataset"], spec["config"], split=spec["split"], **kwargs)
    else:
        dsd = load_dataset(spec["dataset"], spec["config"], **kwargs)
        split = list(dsd.keys())[0]
        print(f"  splits={list(dsd.keys())} -> using '{split}'", flush=True)
        ds = dsd[split]
    return ds, spec["ref_keys"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=["vaani", "contextual", "fleurs"])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/benchmarks/nemotron_vs_indic"),
        help="Base output dir; WAVs go under <out-dir>/wav/<source>/, manifest at <out-dir>/<source>.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap number of clips (0 = all). For vaani, ALWAYS set this (e.g. 500).")
    parser.add_argument("--target-sr", type=int, default=16000, help="Output sample rate (Nemotron + IndicConformer use 16 kHz).")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
        help="HF token for gated datasets. Default: HUGGINGFACE_HUB_TOKEN or HF_TOKEN (auto-loaded from repo .env).",
    )
    parser.add_argument("--vaani-config", default="audio/Hindi", help="Vaani config (default audio/Hindi).")
    parser.add_argument("--vaani-split", default="train", help="Vaani split (Vaani ships only 'train').")
    args = parser.parse_args()

    global np
    import numpy as np  # deferred so --help works without these installed
    import soundfile as sf

    if args.source == "vaani" and args.limit == 0:
        print("WARNING: --source vaani with --limit 0 streams the entire Hindi split (very large). "
              "Pass e.g. --limit 500 for a benchmark subset.", file=sys.stderr, flush=True)

    ds, ref_keys = load_rows(args)
    n_total = len(ds) if hasattr(ds, "__len__") else None
    print(f"Dataset rows: {n_total if n_total is not None else 'streaming'}", flush=True)

    out_dir: Path = args.out_dir.resolve()
    wav_dir = out_dir / "wav" / args.source
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{args.source}.jsonl"

    written = 0
    skipped_empty = 0
    skipped_audio = 0
    total_audio_s = 0.0
    decode_warned = False
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(ds):
            if args.limit and written >= args.limit:
                break
            if idx == 0:
                print(f"  first-row keys: {sorted(row.keys())}", file=sys.stderr, flush=True)

            ref = first_present(row, ref_keys).strip()
            if not ref:
                skipped_empty += 1
                continue

            audio_cell = next((row[k] for k in AUDIO_CELL_KEYS if k in row), None)
            try:
                got = extract_audio(audio_cell)
            except Exception as exc:  # e.g. mp3 without ffmpeg backend
                if not decode_warned:
                    print(f"  audio decode error (will skip such rows): {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    decode_warned = True
                got = None
            if got is None:
                skipped_audio += 1
                continue
            arr, sr = got
            wav = to_mono(arr)
            wav = resample_linear(wav, sr or args.target_sr, args.target_sr)
            if wav.size == 0:
                skipped_audio += 1
                continue

            wav_path = wav_dir / f"{args.source}_{idx:05d}.wav"
            sf.write(str(wav_path), wav, args.target_sr, subtype="PCM_16")
            duration = round(wav.shape[0] / float(args.target_sr), 6)
            total_audio_s += duration

            fout.write(json.dumps(
                {"audio_filepath": str(wav_path), "duration": duration, "text": ref},
                ensure_ascii=False,
            ) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"  wrote {written} clips ...", flush=True)

    print(
        f"\nDone. source={args.source} clips={written} skipped_empty={skipped_empty} "
        f"skipped_audio={skipped_audio} audio={total_audio_s/60.0:.1f} min\n"
        f"manifest: {manifest_path}\nwavs: {wav_dir}",
        flush=True,
    )
    print(
        "\nNext: gate it ->\n"
        f"  python tools/validate_nemo_manifest_audio.py \\\n"
        f"    --input {manifest_path} \\\n"
        f"    --output {out_dir / (args.source + '.valid.jsonl')} \\\n"
        f"    --rejects {out_dir / (args.source + '.rejects.jsonl')} \\\n"
        f"    --summary-json {out_dir / (args.source + '.validate.json')} \\\n"
        f"    --expected-sample-rate {args.target_sr} --require-mono \\\n"
        f"    --min-duration-sec 0.3 --max-duration-sec 40",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
