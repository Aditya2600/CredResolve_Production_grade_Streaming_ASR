#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from asr_text_normalizer import normalize_asr_text
except ModuleNotFoundError:  # pragma: no cover - used when imported as tools.*
    from tools.asr_text_normalizer import normalize_asr_text

try:
    import soundfile as sf
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    sf = None

try:
    from scipy.signal import resample_poly
except ModuleNotFoundError:  # pragma: no cover - linear fallback is tested indirectly
    resample_poly = None


SPLITS = ("train", "dev", "test")
TEXT_KEY_CANDIDATES = ("raw_text", "text", "reference", "transcript", "normalized_text", "sentence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite Vaani split manifests to telephone-bandwidth 8 kHz WAVs. "
            "Keep NeMo training --sample-rate at the model rate, normally 16000."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/vaani_50h_multilingual_split"),
        help="Directory containing train/dev/test JSONL manifests.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/vaani_8khz_split"),
        help="Output directory for rewritten manifests and 8 kHz audio.",
    )
    parser.add_argument(
        "--splits",
        default=",".join(SPLITS),
        help="Comma-separated split names to convert. Default: train,dev,test",
    )
    parser.add_argument("--target-sample-rate", type=int, default=8000)
    parser.add_argument(
        "--model-sample-rate",
        type=int,
        default=16000,
        help="Model input rate used before downsampling. Stored as metadata; use this for PEFT --sample-rate.",
    )
    parser.add_argument("--limit", type=int, help="Optional per-split row limit for smoke tests.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing output WAVs even if they already have the target sample rate.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{lineno}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: manifest is empty")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def split_arg(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in value.split(",") if item.strip())
    if not splits:
        raise SystemExit("--splits must include at least one split name")
    return splits


def row_language(row: dict[str, Any]) -> str:
    for key in ("lang", "language", "language_id", "locale", "language_code"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def extract_text_for_normalization(row: dict[str, Any]) -> str:
    for key in TEXT_KEY_CANDIDATES:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_training_text(row: dict[str, Any], *, manifest: Path, line_num: int) -> str:
    raw_text = extract_text_for_normalization(row)
    clean_text = normalize_asr_text(raw_text, row_language(row))
    if not clean_text:
        row_id = row.get("id") or row.get("audio_filepath") or line_num
        raise SystemExit(f"{manifest}:{line_num}: empty text after normalization for row {row_id!r}")
    return clean_text


def row_duration(row: dict[str, Any]) -> float:
    try:
        return max(0.0, float(row.get("duration") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seconds_by_language: defaultdict[str, float] = defaultdict(float)
    rows_by_language: Counter[str] = Counter()
    total_seconds = 0.0
    for row in rows:
        language = row_language(row)
        duration = row_duration(row)
        rows_by_language[language] += 1
        seconds_by_language[language] += duration
        total_seconds += duration
    return {
        "rows": len(rows),
        "duration_hours_total": round(total_seconds / 3600.0, 6),
        "rows_by_language": dict(sorted(rows_by_language.items())),
        "duration_hours_by_language": {
            key: round(value / 3600.0, 6) for key, value in sorted(seconds_by_language.items())
        },
    }


def sanitize_component(value: Any, *, fallback: str = "row") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        text = fallback
    return text[:96]


def resolve_audio_path(value: Any, manifest_path: Path) -> Path:
    if value is None or not str(value).strip():
        raise SystemExit(f"{manifest_path}: row is missing audio_filepath")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def output_audio_path(row: dict[str, Any], src_audio: Path, *, split: str, index: int, out_dir: Path) -> Path:
    row_id = sanitize_component(row.get("id") or src_audio.stem, fallback=f"{split}_{index:06d}")
    digest = hashlib.sha1(str(src_audio).encode("utf-8")).hexdigest()[:10]
    return out_dir / "audio" / split / f"{index:06d}_{row_id}_{digest}.wav"


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if resample_poly is not None:
        divisor = math.gcd(int(src_sr), int(dst_sr))
        return resample_poly(audio, int(dst_sr) // divisor, int(src_sr) // divisor).astype(np.float32)

    src_positions = np.arange(audio.shape[0], dtype=np.float32)
    dst_length = max(1, int(round(audio.shape[0] * (float(dst_sr) / float(src_sr)))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    if sf is None:
        raise SystemExit("soundfile is required to prepare 8 kHz Vaani manifests")
    if not path.exists():
        raise SystemExit(f"audio file does not exist: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono(np.asarray(audio)), int(sample_rate)


def reusable_wav(path: Path, sample_rate: int) -> bool:
    if sf is None or not path.exists():
        return False
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return False
    return int(info.samplerate) == int(sample_rate)


def write_telephony_wav(
    *,
    src_audio: Path,
    dst_audio: Path,
    target_sample_rate: int,
    model_sample_rate: int,
) -> dict[str, Any]:
    audio, source_sample_rate = read_audio(src_audio)
    model_rate_audio = resample_audio(audio, source_sample_rate, model_sample_rate)
    telephony_audio = resample_audio(model_rate_audio, model_sample_rate, target_sample_rate)
    dst_audio.parent.mkdir(parents=True, exist_ok=True)
    assert sf is not None
    sf.write(str(dst_audio), telephony_audio, target_sample_rate, subtype="PCM_16")
    return {
        "source_sample_rate": int(source_sample_rate),
        "source_samples": int(audio.shape[0]),
        "telephony_samples": int(telephony_audio.shape[0]),
    }


def convert_split(
    *,
    input_manifest: Path,
    output_manifest: Path,
    split: str,
    out_dir: Path,
    target_sample_rate: int,
    model_sample_rate: int,
    overwrite: bool,
    limit: int | None,
    progress_every: int,
) -> dict[str, Any]:
    rows = read_jsonl(input_manifest)
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    converted_rows: list[dict[str, Any]] = []
    reused = 0
    written = 0
    source_sample_rates: Counter[str] = Counter()

    for index, row in enumerate(rows):
        src_audio = resolve_audio_path(row.get("audio_filepath"), input_manifest)
        dst_audio = output_audio_path(row, src_audio, split=split, index=index, out_dir=out_dir)
        if not overwrite and reusable_wav(dst_audio, target_sample_rate):
            src_info = sf.info(str(src_audio)) if sf is not None else None
            info = sf.info(str(dst_audio)) if sf is not None else None
            conversion = {
                "source_sample_rate": int(src_info.samplerate) if src_info is not None else row.get("source_sample_rate"),
                "source_samples": int(src_info.frames) if src_info is not None else None,
                "telephony_samples": int(info.frames) if info is not None else None,
            }
            reused += 1
        else:
            conversion = write_telephony_wav(
                src_audio=src_audio,
                dst_audio=dst_audio,
                target_sample_rate=target_sample_rate,
                model_sample_rate=model_sample_rate,
            )
            written += 1

        source_sr = conversion.get("source_sample_rate")
        if source_sr is not None:
            source_sample_rates[str(source_sr)] += 1

        out_row = dict(row)
        if "raw_text" not in out_row:
            out_row["raw_text"] = str(row.get("text") or "")
        out_row["text"] = normalize_training_text(row, manifest=input_manifest, line_num=index + 1)
        out_row["source_audio_filepath"] = str(src_audio)
        out_row["audio_filepath"] = str(dst_audio)
        out_row["source_sample_rate"] = source_sr
        out_row["telephony_sample_rate"] = int(target_sample_rate)
        out_row["model_sample_rate"] = int(model_sample_rate)
        out_row["telephony_simulation"] = "downsampled_to_8khz"
        converted_rows.append(out_row)

        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"{split}: converted {index + 1}/{len(rows)} rows", file=sys.stderr, flush=True)

    write_jsonl(output_manifest, converted_rows)
    return {
        "input": str(input_manifest),
        "output": str(output_manifest),
        "audio_dir": str(out_dir / "audio" / split),
        "written_wavs": int(written),
        "reused_wavs": int(reused),
        "source_sample_rates": dict(sorted(source_sample_rates.items())),
        "summary": summarize(converted_rows),
    }


def prepare_splits(
    input_dir: Path,
    out_dir: Path,
    *,
    splits: tuple[str, ...] = SPLITS,
    target_sample_rate: int = 8000,
    model_sample_rate: int = 16000,
    overwrite: bool = False,
    limit: int | None = None,
    progress_every: int = 1000,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, Any] = {}
    for split in splits:
        input_manifest = input_dir / f"{split}.jsonl"
        if not input_manifest.exists():
            raise SystemExit(f"split manifest does not exist: {input_manifest}")
        split_summaries[split] = convert_split(
            input_manifest=input_manifest,
            output_manifest=out_dir / f"{split}.jsonl",
            split=split,
            out_dir=out_dir,
            target_sample_rate=target_sample_rate,
            model_sample_rate=model_sample_rate,
            overwrite=overwrite,
            limit=limit,
            progress_every=progress_every,
        )

    summary = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "target_sample_rate": int(target_sample_rate),
        "model_sample_rate": int(model_sample_rate),
        "training_note": (
            "Use the 8 kHz manifests with PEFT --sample-rate equal to model_sample_rate "
            "so NeMo resamples narrowband audio to the pretrained model frontend."
        ),
        "resampler": "scipy.signal.resample_poly" if resample_poly is not None else "linear_interp",
        "splits": split_summaries,
        "outputs": {split: str(out_dir / f"{split}.jsonl") for split in splits},
    }
    summary["outputs"]["summary_json"] = str(out_dir / "split_summary.json")
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = prepare_splits(
        args.input_dir,
        args.out_dir,
        splits=split_arg(args.splits),
        target_sample_rate=int(args.target_sample_rate),
        model_sample_rate=int(args.model_sample_rate),
        overwrite=bool(args.overwrite),
        limit=args.limit,
        progress_every=int(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
