#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import soundfile as sf
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.indicvoices_dataset import VAANI_DATASET_ID, VAANI_HINDI_CONFIG, load_vaani_stream

try:
    from eval_indicvoices_wer import (
        audio_to_float32,
        detect_audio_field,
        detect_text_field,
        resolve_cache_dir,
        resample_linear,
        to_mono,
    )
except ImportError:  # pragma: no cover - allows package-style imports
    from tools.eval_indicvoices_wer import (
        audio_to_float32,
        detect_audio_field,
        detect_text_field,
        resolve_cache_dir,
        resample_linear,
        to_mono,
    )


HINDI_LANGUAGE_VALUES = frozenset({"hi", "hin", "hindi"})
DEFAULT_TARGET_HOURS = 20.0


@dataclass(frozen=True)
class VaaniExportFilters:
    min_duration: float = 0.3
    max_duration: float = 9999.0
    max_words: int = 9999
    max_chars: int = 9999
    min_words_per_sec: float = 0.0
    max_words_per_sec: float = 9999.0
    language_field: str = "language"
    drop_unintelligible: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a filtered 20-hour Hindi subset of ARTPARK-IISc/Vaani-transcription-part "
            "to local WAV files plus a NeMo JSONL manifest."
        )
    )
    parser.add_argument("--dataset-config", default=VAANI_HINDI_CONFIG, help="Vaani config. Default: audio/Hindi")
    parser.add_argument("--split", default="train", help="Vaani split. Default: train")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/vaani_hindi_20h_train"))
    parser.add_argument("--hf-token", help="Optional Hugging Face token for gated Vaani access.")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face datasets cache dir.")
    parser.add_argument("--target-hours", type=float, default=DEFAULT_TARGET_HOURS, help="Default: 20.0")
    parser.add_argument("--seed", type=int, default=42, help="Streaming shuffle seed. Default: 42")
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10000,
        help="Streaming shuffle buffer size. Use 1 to preserve source order. Default: 10000",
    )
    parser.add_argument("--max-scan-rows", type=int, help="Optional cap on scanned source rows.")
    parser.add_argument("--text-field", help="Override transcript field. Auto-detected by default.")
    parser.add_argument("--audio-field", help="Override audio field. Auto-detected by default.")
    parser.add_argument("--language-field", default="language", help="Language metadata field. Default: language")
    parser.add_argument("--language", default="hi", help="Language code to write in the NeMo manifest. Default: hi")
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=9999.0)
    parser.add_argument("--max-words", type=int, default=9999)
    parser.add_argument("--max-chars", type=int, default=9999)
    parser.add_argument("--min-words-per-sec", type=float, default=0.0)
    parser.add_argument("--max-words-per-sec", type=float, default=9999.0)
    parser.add_argument(
        "--drop-unintelligible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject transcripts containing <unintelligible>. Default: enabled",
    )
    return parser.parse_args()


def load_dotenv_value(*keys: str) -> str | None:
    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.exists():
        return None
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
            "Vaani-transcription-part is gated. Provide --hf-token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
        )
    return token


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).strip()


def is_hindi_language(value: Any) -> bool:
    text = normalize_text(value).casefold()
    if not text:
        return True
    return text in HINDI_LANGUAGE_VALUES


def sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "value"


def iter_buffered_shuffle(
    rows: Iterable[tuple[int, dict[str, Any]]],
    *,
    seed: int,
    buffer_size: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if buffer_size <= 1:
        yield from rows
        return

    rng = random.Random(seed)
    iterator = iter(rows)
    buffer: list[tuple[int, dict[str, Any]]] = []
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


def audio_identity(audio_value: Any, *, dataset_index: int) -> str:
    if isinstance(audio_value, dict):
        path = normalize_text(audio_value.get("path"))
        if path:
            return f"path:{path}"
        raw_bytes = audio_value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            digest = hashlib.sha1(bytes(raw_bytes)).hexdigest()
            return f"bytes:{digest}"
    if isinstance(audio_value, (str, Path)):
        return f"path:{Path(str(audio_value)).expanduser()}"
    return f"index:{dataset_index}"


def reject_row(
    *,
    dataset_index: int,
    reason: str,
    text: str = "",
    language_value: Any = "",
    detail: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_index": dataset_index,
        "reason": reason,
        "text": text,
        "language": normalize_text(language_value),
    }
    if detail:
        row["detail"] = detail
    return row


def process_sample(
    sample: dict[str, Any],
    *,
    dataset_index: int,
    text_field: str,
    audio_field: str,
    filters: VaaniExportFilters,
    seen_dedupe_keys: set[tuple[str, str]],
    dataset_config: str,
    split: str,
    language_code: str,
) -> dict[str, Any]:
    text = normalize_text(sample.get(text_field))
    language_value = sample.get(filters.language_field, "") if filters.language_field else ""

    if not text:
        return {"status": "rejected", "reject": reject_row(dataset_index=dataset_index, reason="empty_text")}
    if filters.drop_unintelligible and "<unintelligible>" in text.casefold():
        return {
            "status": "rejected",
            "reject": reject_row(dataset_index=dataset_index, reason="unintelligible_text", text=text, language_value=language_value),
        }
    if not is_hindi_language(language_value):
        return {
            "status": "rejected",
            "reject": reject_row(dataset_index=dataset_index, reason="non_hindi_language", text=text, language_value=language_value),
        }

    try:
        audio, sample_rate = audio_to_float32(sample[audio_field])
        audio = to_mono(audio)
        audio = resample_linear(audio, sample_rate, 16000)
    except Exception as exc:
        return {
            "status": "rejected",
            "reject": reject_row(
                dataset_index=dataset_index,
                reason="audio_decode_failed",
                text=text,
                language_value=language_value,
                detail=str(exc),
            ),
        }

    duration = float(len(audio) / 16000.0) if len(audio) else 0.0
    if duration < filters.min_duration:
        return {
            "status": "rejected",
            "reject": reject_row(dataset_index=dataset_index, reason="too_short", text=text, language_value=language_value),
        }
    if duration > filters.max_duration:
        return {
            "status": "rejected",
            "reject": reject_row(dataset_index=dataset_index, reason="too_long", text=text, language_value=language_value),
        }

    words = text.split()
    word_count = len(words)
    if word_count > filters.max_words:
        return {
            "status": "rejected",
            "reject": reject_row(dataset_index=dataset_index, reason="too_many_words", text=text, language_value=language_value),
        }
    if len(text) > filters.max_chars:
        return {
            "status": "rejected",
            "reject": reject_row(dataset_index=dataset_index, reason="too_many_chars", text=text, language_value=language_value),
        }

    words_per_sec = word_count / duration if duration > 0 else 0.0
    if words_per_sec < filters.min_words_per_sec:
        return {
            "status": "rejected",
            "reject": reject_row(dataset_index=dataset_index, reason="mismatch_low_words_per_sec", text=text, language_value=language_value),
        }
    if words_per_sec > filters.max_words_per_sec:
        return {
            "status": "rejected",
            "reject": reject_row(dataset_index=dataset_index, reason="mismatch_high_words_per_sec", text=text, language_value=language_value),
        }

    dedupe_key = (audio_identity(sample.get(audio_field), dataset_index=dataset_index), text.casefold())
    if dedupe_key in seen_dedupe_keys:
        return {
            "status": "rejected",
            "reject": reject_row(dataset_index=dataset_index, reason="duplicate_audio_text", text=text, language_value=language_value),
        }
    seen_dedupe_keys.add(dedupe_key)

    row = {
        "id": f"vaani-{sanitize_component(dataset_config)}-{split}-{dataset_index}",
        "dataset": VAANI_DATASET_ID,
        "dataset_config": dataset_config,
        "split": split,
        "dataset_index": dataset_index,
        "duration": round(duration, 3),
        "text": text,
        "lang": language_code,
        "source": "vaani",
        "words": word_count,
        "words_per_sec": round(words_per_sec, 4),
    }
    for key in ("language", "gender", "state", "district", "referenceImage"):
        if key in sample:
            row[f"vaani_{key}"] = sample.get(key)

    return {
        "status": "accepted",
        "row": row,
        "audio": audio,
        "duration": duration,
    }


def write_jsonl_line(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_samples_to_nemo(
    indexed_samples: Iterable[tuple[int, dict[str, Any]]],
    *,
    out_dir: Path,
    dataset_config: str,
    split: str,
    target_hours: float,
    seed: int,
    shuffle_buffer_size: int,
    filters: VaaniExportFilters,
    text_field: str,
    audio_field: str,
    language_code: str,
    max_scan_rows: int | None = None,
) -> dict[str, Any]:
    if target_hours <= 0:
        raise SystemExit("--target-hours must be positive")

    out_dir = out_dir.expanduser().resolve()
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.jsonl"
    rejects_path = out_dir / "rejects.jsonl"
    summary_path = out_dir / "summary.json"
    audio_dir.mkdir(parents=True, exist_ok=True)

    target_seconds = float(target_hours) * 3600.0
    seen_dedupe_keys: set[tuple[str, str]] = set()
    reject_counts: Counter[str] = Counter()
    kept_seconds = 0.0
    scanned_rows = 0
    kept_rows = 0
    rejected_rows = 0
    sanitized_config = sanitize_component(dataset_config)

    shuffled_samples = iter_buffered_shuffle(
        indexed_samples,
        seed=seed,
        buffer_size=shuffle_buffer_size,
    )

    with (
        manifest_path.open("w", encoding="utf-8") as manifest_handle,
        rejects_path.open("w", encoding="utf-8") as rejects_handle,
        tqdm(total=target_seconds, desc="Exporting Audio (seconds)", unit="s") as pbar,
    ):
        for dataset_index, sample in shuffled_samples:
            if max_scan_rows is not None and scanned_rows >= max(0, int(max_scan_rows)):
                break
            scanned_rows += 1

            decision = process_sample(
                sample,
                dataset_index=dataset_index,
                text_field=text_field,
                audio_field=audio_field,
                filters=filters,
                seen_dedupe_keys=seen_dedupe_keys,
                dataset_config=dataset_config,
                split=split,
                language_code=language_code,
            )
            if decision["status"] != "accepted":
                rejected_rows += 1
                reject = decision["reject"]
                reject_counts[str(reject["reason"])] += 1
                write_jsonl_line(rejects_handle, reject)
                continue

            wav_path = audio_dir / f"{sanitized_config}_{split}_{dataset_index}.wav"
            sf.write(wav_path, decision["audio"], 16000, subtype="PCM_16")
            manifest_row = dict(decision["row"])
            manifest_row["audio_filepath"] = str(wav_path)
            write_jsonl_line(manifest_handle, manifest_row)

            kept_rows += 1
            added_duration = float(decision["duration"])
            kept_seconds += added_duration
            pbar.update(added_duration)
            if kept_seconds >= target_seconds:
                break

    summary = {
        "dataset": VAANI_DATASET_ID,
        "dataset_config": dataset_config,
        "split": split,
        "target_hours": float(target_hours),
        "target_seconds": round(target_seconds, 3),
        "kept_hours": round(kept_seconds / 3600.0, 6),
        "kept_seconds": round(kept_seconds, 3),
        "target_reached": kept_seconds >= target_seconds,
        "rows_scanned": scanned_rows,
        "rows_kept": kept_rows,
        "rows_rejected": rejected_rows,
        "reject_counts": dict(sorted(reject_counts.items())),
        "seed": int(seed),
        "shuffle_buffer_size": int(shuffle_buffer_size),
        "detected_fields": {
            "text_field": text_field,
            "audio_field": audio_field,
            "language_field": filters.language_field,
        },
        "filters": {
            "min_duration": filters.min_duration,
            "max_duration": filters.max_duration,
            "max_words": filters.max_words,
            "max_chars": filters.max_chars,
            "min_words_per_sec": filters.min_words_per_sec,
            "max_words_per_sec": filters.max_words_per_sec,
            "drop_unintelligible": filters.drop_unintelligible,
        },
        "outputs": {
            "out_dir": str(out_dir),
            "audio_dir": str(audio_dir),
            "manifest_jsonl": str(manifest_path),
            "rejects_jsonl": str(rejects_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def prepend_first_sample(first_sample: dict[str, Any], iterator: Iterator[dict[str, Any]]) -> Iterator[tuple[int, dict[str, Any]]]:
    yield 0, first_sample
    for index, sample in enumerate(iterator, start=1):
        yield index, sample


def main() -> int:
    args = parse_args()
    token = resolve_hf_token(args.hf_token)
    cache_dir = resolve_cache_dir(args.cache_dir)
    filters = VaaniExportFilters(
        min_duration=float(args.min_duration),
        max_duration=float(args.max_duration),
        max_words=int(args.max_words),
        max_chars=int(args.max_chars),
        min_words_per_sec=float(args.min_words_per_sec),
        max_words_per_sec=float(args.max_words_per_sec),
        language_field=str(args.language_field or ""),
        drop_unintelligible=bool(args.drop_unintelligible),
    )

    dataset = load_vaani_stream(
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

    text_field = args.text_field or detect_text_field(first_sample)
    audio_field = args.audio_field or detect_audio_field(first_sample)
    summary = export_samples_to_nemo(
        prepend_first_sample(first_sample, iterator),
        out_dir=args.out_dir,
        dataset_config=args.dataset_config,
        split=args.split,
        target_hours=float(args.target_hours),
        seed=int(args.seed),
        shuffle_buffer_size=int(args.shuffle_buffer_size),
        filters=filters,
        text_field=text_field,
        audio_field=audio_field,
        language_code=str(args.language),
        max_scan_rows=args.max_scan_rows,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["target_reached"]:
        raise SystemExit(
            f"Only exported {summary['kept_hours']}h, below requested {summary['target_hours']}h. "
            "Increase --max-scan-rows or loosen filters."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
