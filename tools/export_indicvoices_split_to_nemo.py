#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.indicvoices_dataset import load_indicvoices_stream

try:
    from eval_indicvoices_wer import (
        audio_to_float32,
        detect_audio_field,
        detect_text_field,
        resolve_cache_dir,
        resolve_hf_token,
        resample_linear,
        to_mono,
    )
except ImportError:  # pragma: no cover - allows package-style imports
    from tools.eval_indicvoices_wer import (
        audio_to_float32,
        detect_audio_field,
        detect_text_field,
        resolve_cache_dir,
        resolve_hf_token,
        resample_linear,
        to_mono,
    )


LANGUAGE_MAP = {
    "assamese": "as",
    "bengali": "bn",
    "bodo": "brx",
    "dogri": "doi",
    "english": "en",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "kashmiri": "ks",
    "konkani": "gom",
    "maithili": "mai",
    "malayalam": "ml",
    "manipuri": "mni",
    "marathi": "mr",
    "nepali": "ne",
    "odia": "or",
    "punjabi": "pa",
    "sanskrit": "sa",
    "santali": "sat",
    "sindhi": "sd",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an ai4bharat/IndicVoices split into local WAV files plus a NeMo-friendly JSONL manifest."
        )
    )
    parser.add_argument("--dataset-config", default="hindi", help="IndicVoices config, e.g. hindi")
    parser.add_argument("--split", default="valid", help="IndicVoices split, e.g. valid")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write audio/ and manifest.jsonl into")
    parser.add_argument("--hf-token", help="Optional Hugging Face token for gated IndicVoices access")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face cache dir")
    parser.add_argument("--limit", type=int, help="Optional max number of rows to export")
    parser.add_argument("--language", help="Optional language ID override, e.g. hi")
    return parser.parse_args()


def resolve_language(dataset_config: str, explicit_language: str | None) -> str:
    if explicit_language:
        return explicit_language
    code = LANGUAGE_MAP.get(dataset_config.strip().lower())
    if not code:
        raise SystemExit(
            f"Do not know the language id for dataset config {dataset_config!r}. Pass --language explicitly."
        )
    return code


def main() -> int:
    args = parse_args()
    token = resolve_hf_token(args.hf_token)
    language = resolve_language(args.dataset_config, args.language)

    cache_dir = resolve_cache_dir(args.cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(REPO_ROOT / ".cache" / "huggingface" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".cache" / "huggingface" / "datasets"))

    dataset = load_indicvoices_stream(
        dataset_config=args.dataset_config,
        split=args.split,
        token=token,
        cache_dir=cache_dir,
    )

    iterator = iter(dataset)
    try:
        first_sample = next(iterator)
    except StopIteration as exc:
        raise SystemExit("Dataset split is empty.") from exc

    text_field = detect_text_field(first_sample)
    audio_field = detect_audio_field(first_sample)

    out_dir = args.out_dir.expanduser().resolve()
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    def iter_samples():
        yield 0, first_sample
        for index, sample in enumerate(iterator, start=1):
            if args.limit is not None and index >= max(0, args.limit):
                break
            yield index, sample

    count = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, sample in iter_samples():
            if args.limit is not None and count >= max(0, args.limit):
                break
            audio, sr = audio_to_float32(sample[audio_field])
            audio = to_mono(audio)
            audio = resample_linear(audio, sr, 16000)
            wav_path = audio_dir / f"{args.dataset_config}_{args.split}_{index}.wav"
            sf.write(wav_path, audio, 16000, subtype="PCM_16")

            row = {
                "id": f"{args.dataset_config}-{args.split}-{index}",
                "dataset": "ai4bharat/IndicVoices",
                "dataset_config": args.dataset_config,
                "split": args.split,
                "dataset_index": index,
                "audio_filepath": str(wav_path),
                "duration": round(len(audio) / 16000.0, 3),
                "text": str(sample[text_field]).strip(),
                "lang": language,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    summary = {
        "dataset": "ai4bharat/IndicVoices",
        "dataset_config": args.dataset_config,
        "split": args.split,
        "language": language,
        "exported": count,
        "manifest": str(manifest_path),
        "audio_dir": str(audio_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
