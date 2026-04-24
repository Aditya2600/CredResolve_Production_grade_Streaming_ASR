#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import soundfile as sf
except Exception:  # pragma: no cover - depends on local environment
    sf = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - depends on local environment
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker.app.lid import DetectionResult, build_language_detector

try:
    from tools.normalize_indic_transcripts import (
        DEFAULT_CONFIG_PATH,
        load_language_configs,
        normalize_language_alias,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from normalize_indic_transcripts import (  # type: ignore
        DEFAULT_CONFIG_PATH,
        load_language_configs,
        normalize_language_alias,
    )


AUDIO_COLUMN_CANDIDATES = (
    "wav_audio_path",
    "local_audio_path",
    "audio_path",
    "channel_0_path",
    "channel_1_path",
)
DEFAULT_INVENTORY_PATH = Path("outputs/results_all/data_inventory_normalized.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/results_all/lid_inventory_scan")
DEFAULT_LID_PRIMARY_PROVIDER = "vakgyata"
DEFAULT_LID_PRIMARY_SOURCE = "onecxi/vakgyata-small"
DEFAULT_LID_PRIMARY_MODEL_DIR = "models/lid_primary"
DEFAULT_LID_FALLBACK_PROVIDER = "speechbrain"
DEFAULT_LID_FALLBACK_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
DEFAULT_LID_FALLBACK_MODEL_DIR = "models/lid_fallback"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
LANGUAGE_CODE_RE = re.compile(r"^[a-z]{2,3}$")

log = logging.getLogger("tools.scan_inventory_lid_mismatches")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the worker LID detector chain over an inventory and report rows whose "
            "labeled language differs from the detected language."
        )
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help="Inventory CSV/Parquet to scan. Default: outputs/results_all/data_inventory_normalized.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the scan outputs. Default: outputs/results_all/lid_inventory_scan",
    )
    parser.add_argument(
        "--audio-column",
        default="auto",
        help="Audio path column to scan. Default: auto-detect, preferring wav_audio_path",
    )
    parser.add_argument(
        "--language-column",
        default="language",
        help="Inventory column containing the labeled language. Default: language",
    )
    parser.add_argument(
        "--supported-languages",
        help=(
            "Comma-separated ISO codes or language labels accepted from LID. "
            "Default: derive from the inventory labels."
        ),
    )
    parser.add_argument("--row-id-column", default="row_id", help="Row identifier column. Default: row_id")
    parser.add_argument("--call-id-column", default="call_id", help="Call identifier column. Default: call_id")
    parser.add_argument(
        "--language-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Language normalization config path. Default: configs/transcript_normalization/language_mappings.json",
    )
    parser.add_argument(
        "--primary-provider",
        default=DEFAULT_LID_PRIMARY_PROVIDER,
        help="Primary LID provider. Default: vakgyata",
    )
    parser.add_argument(
        "--primary-source",
        default=DEFAULT_LID_PRIMARY_SOURCE,
        help="Primary LID model source. Default: onecxi/vakgyata-small",
    )
    parser.add_argument(
        "--primary-model-dir",
        default=DEFAULT_LID_PRIMARY_MODEL_DIR,
        help="Primary LID cache directory. Default: models/lid_primary",
    )
    parser.add_argument(
        "--fallback-provider",
        default=DEFAULT_LID_FALLBACK_PROVIDER,
        help="Fallback LID provider. Default: speechbrain",
    )
    parser.add_argument(
        "--fallback-source",
        default=DEFAULT_LID_FALLBACK_SOURCE,
        help="Fallback LID model source. Default: speechbrain/lang-id-voxlingua107-ecapa",
    )
    parser.add_argument(
        "--fallback-model-dir",
        default=DEFAULT_LID_FALLBACK_MODEL_DIR,
        help="Fallback LID cache directory. Default: models/lid_fallback",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Vakgyata confidence threshold before falling back. Default: 0.70",
    )
    parser.add_argument("--limit", type=int, help="Optional number of rows to scan")
    parser.add_argument(
        "--max-audio-sec",
        type=float,
        help="Optional cap on audio seconds read from the start of each recording",
    )
    parser.add_argument(
        "--min-audio-sec",
        type=float,
        default=0.25,
        help="Skip rows whose effective audio is shorter than this many seconds. Default: 0.25",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Progress logging cadence in rows when tqdm is disabled. Default: 50",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar and use periodic logging instead.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level. Default: INFO",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first row error instead of continuing",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def should_use_tqdm(args: argparse.Namespace) -> bool:
    return bool(tqdm is not None and not args.no_progress and sys.stderr.isatty())


def load_inventory(path: Path) -> pd.DataFrame:
    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved, dtype=str).fillna("")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved).fillna("")
    raise SystemExit(f"unsupported inventory format for {resolved}; use CSV or Parquet")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=True, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_wav_bytes(path: Path, *, max_audio_sec: float | None) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate())
        sample_width = int(wav_file.getsampwidth())
        channels = int(wav_file.getnchannels())
        frame_count = int(wav_file.getnframes())
        if max_audio_sec is not None:
            frame_count = min(frame_count, max(1, int(round(max_audio_sec * sample_rate))))
        pcm = wav_file.readframes(frame_count)

    if sample_width == 1:
        audio = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise SystemExit(f"{path}: unsupported WAV sample width {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, sample_rate


def load_audio(path: Path, *, max_audio_sec: float | None) -> tuple[np.ndarray, int]:
    if sf is not None:
        with sf.SoundFile(str(path)) as handle:
            sample_rate = int(handle.samplerate)
            frame_count = -1
            if max_audio_sec is not None:
                frame_count = max(1, int(round(max_audio_sec * sample_rate)))
            audio = handle.read(frames=frame_count, dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), sample_rate

    if path.suffix.lower() != ".wav":
        raise SystemExit(f"{path}: soundfile is unavailable, so only .wav inputs are supported")
    return read_wav_bytes(path, max_audio_sec=max_audio_sec)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    src_positions = np.arange(audio.shape[0], dtype=np.float32)
    dst_length = max(1, int(round(audio.shape[0] * (dst_sr / float(src_sr)))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    return np.clip(np.rint(clipped * 32767.0), -32768, 32767).astype("<i2").tobytes()


def normalize_inventory_language(value: Any, alias_map: dict[str, str]) -> str | None:
    normalized = normalize_language_alias(str(value or ""))
    if not normalized:
        return None
    resolved = alias_map.get(normalized)
    if resolved:
        return resolved

    compact = normalized.replace(" ", "")
    if LANGUAGE_CODE_RE.fullmatch(compact):
        return compact
    return None


def resolve_audio_column(frame: pd.DataFrame, requested: str) -> str:
    if requested and requested != "auto":
        if requested not in frame.columns:
            raise SystemExit(f"audio column `{requested}` not found in inventory")
        return requested

    for candidate in AUDIO_COLUMN_CANDIDATES:
        if candidate in frame.columns and frame[candidate].astype(str).str.strip().any():
            return candidate
    raise SystemExit(
        "could not detect an audio path column; pass --audio-column explicitly. "
        f"Available columns: {list(frame.columns)}"
    )


def resolve_audio_path(value: Any, inventory_path: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise FileNotFoundError("missing audio path")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate

    repo_relative = (REPO_ROOT / candidate).resolve()
    if repo_relative.exists():
        return repo_relative

    inventory_relative = (inventory_path.parent / candidate).resolve()
    if inventory_relative.exists():
        return inventory_relative

    return repo_relative


def parse_supported_languages(
    value: str | None,
    *,
    frame: pd.DataFrame,
    language_column: str,
    alias_map: dict[str, str],
) -> list[str]:
    candidate_value = value or os.environ.get("ASR_SUPPORTED_LANGS", "").strip() or None
    if candidate_value:
        codes: list[str] = []
        for part in candidate_value.split(","):
            code = normalize_inventory_language(part, alias_map)
            if code and code not in codes:
                codes.append(code)
        if codes:
            return codes
        raise SystemExit(f"no supported languages could be resolved from `{candidate_value}`")

    if language_column not in frame.columns:
        raise SystemExit(f"language column `{language_column}` not found in inventory")

    codes = []
    for raw in frame[language_column].tolist():
        code = normalize_inventory_language(raw, alias_map)
        if code and code not in codes:
            codes.append(code)
    if not codes:
        raise SystemExit(
            "no supported languages could be derived from the inventory labels; "
            "pass --supported-languages explicitly"
        )
    return codes


def detection_to_row(detection: DetectionResult | None) -> dict[str, Any]:
    if detection is None:
        return {
            "primary_predicted_language": "",
            "primary_raw_label": "",
            "primary_normalized_label": "",
            "primary_provider": "",
            "primary_confidence": None,
            "predicted_language": "",
            "predicted_raw_label": "",
            "predicted_normalized_label": "",
            "lid_provider": "",
            "lid_confidence": None,
            "lid_fallback_from": "",
            "lid_fallback_reason": "",
        }
    return {
        "primary_predicted_language": detection.primary_language or "",
        "primary_raw_label": detection.primary_raw_label,
        "primary_normalized_label": detection.primary_normalized_label,
        "primary_provider": detection.primary_provider or "",
        "primary_confidence": detection.primary_confidence,
        "predicted_language": detection.language or "",
        "predicted_raw_label": detection.raw_label,
        "predicted_normalized_label": detection.normalized_label,
        "lid_provider": detection.provider,
        "lid_confidence": detection.confidence,
        "lid_fallback_from": detection.fallback_from or "",
        "lid_fallback_reason": detection.fallback_reason or "",
    }


def build_confusion_matrix(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in results:
        if row["status"] != "ok":
            continue
        expected = str(row.get("label_language_code") or "")
        predicted = str(row.get("predicted_language") or "")
        if not expected:
            continue
        matrix.setdefault(expected, {})
        key = predicted or "<missing>"
        matrix[expected][key] = matrix[expected].get(key, 0) + 1
    return {
        expected: dict(sorted(predictions.items(), key=lambda item: item[0]))
        for expected, predictions in sorted(matrix.items(), key=lambda item: item[0])
    }


def summarize_results(
    results: list[dict[str, Any]],
    *,
    inventory_path: Path,
    audio_column: str,
    supported_languages: list[str],
    scan_started_at_utc: str,
    scan_finished_at_utc: str,
    scan_elapsed_sec: float,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "") for row in results)
    ok_rows = [row for row in results if row["status"] == "ok"]
    mismatch_rows = [row for row in ok_rows if not row["language_match"]]
    lid_latencies_ms = [
        float(row["lid_latency_ms"])
        for row in ok_rows
        if row.get("lid_latency_ms") is not None
    ]
    row_elapsed_ms = [
        float(row["row_elapsed_ms"])
        for row in results
        if row.get("row_elapsed_ms") is not None
    ]
    summary = {
        "inventory": str(inventory_path),
        "audio_column": audio_column,
        "scan_started_at_utc": scan_started_at_utc,
        "scan_finished_at_utc": scan_finished_at_utc,
        "scan_elapsed_sec": round(float(scan_elapsed_sec), 3),
        "rows_total": len(results),
        "rows_ok": len(ok_rows),
        "rows_mismatch": len(mismatch_rows),
        "rows_error": len(results) - len(ok_rows),
        "mismatch_rate": round((len(mismatch_rows) / len(ok_rows)), 6) if ok_rows else 0.0,
        "supported_languages": supported_languages,
        "status_counts": dict(sorted(status_counts.items(), key=lambda item: item[0])),
        "labeled_language_counts": dict(sorted(Counter(row["label_language_code"] for row in ok_rows).items())),
        "predicted_language_counts": dict(sorted(Counter(row["predicted_language"] for row in ok_rows).items())),
        "confusion_matrix": build_confusion_matrix(results),
        "timing": {
            "avg_lid_latency_ms": round(sum(lid_latencies_ms) / len(lid_latencies_ms), 3)
            if lid_latencies_ms
            else None,
            "avg_row_elapsed_ms": round(sum(row_elapsed_ms) / len(row_elapsed_ms), 3)
            if row_elapsed_ms
            else None,
        },
    }
    return summary


def run_scan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scan_started_perf = time.perf_counter()
    scan_started_at_utc = utc_now_iso()
    inventory_path = args.inventory.expanduser().resolve()
    full_frame = load_inventory(inventory_path)
    if full_frame.empty:
        raise SystemExit("inventory is empty")

    frame = full_frame
    if args.limit is not None:
        frame = full_frame.head(max(0, int(args.limit))).copy()
    if frame.empty:
        raise SystemExit("inventory is empty after applying --limit")

    _, alias_map = load_language_configs(args.language_config.expanduser().resolve())
    audio_column = resolve_audio_column(frame, args.audio_column)
    supported_languages = parse_supported_languages(
        args.supported_languages,
        frame=full_frame,
        language_column=args.language_column,
        alias_map=alias_map,
    )

    detector = build_language_detector(
        primary_provider=args.primary_provider,
        primary_source=args.primary_source,
        primary_savedir=args.primary_model_dir,
        fallback_provider=args.fallback_provider,
        fallback_source=args.fallback_source,
        fallback_savedir=args.fallback_model_dir,
        confidence_threshold=float(args.confidence_threshold),
    )
    if detector is None:
        raise SystemExit("no LID detector is configured; set a primary and/or fallback provider")
    if not detector.load_model():
        raise SystemExit(
            "failed to load the LID detector chain: "
            f"{detector.last_error or 'unknown error'}. "
            "Install worker LID dependencies in a Python 3.11 environment, or run "
            "`tools/run_inventory_lid_scan.sh` to execute the scan inside the worker image."
        )

    results: list[dict[str, Any]] = []
    total_rows = len(frame)
    rows = frame.to_dict(orient="records")
    use_tqdm = should_use_tqdm(args)
    ok_count = 0
    mismatch_count = 0
    error_count = 0

    progress = None
    row_iterable: Any = rows
    if use_tqdm:
        progress = tqdm(
            rows,
            total=total_rows,
            desc="LID scan",
            unit="row",
            dynamic_ncols=True,
            leave=True,
        )
        progress.set_postfix(ok=0, mismatch=0, error=0, refresh=False)
        row_iterable = progress

    for index, row in enumerate(row_iterable, start=1):
        row_started_perf = time.perf_counter()
        row_started_at_utc = utc_now_iso()
        base_result = {
            "index": index - 1,
            "row_id": str(row.get(args.row_id_column) or ""),
            "call_id": str(row.get(args.call_id_column) or ""),
            "label_language": str(row.get(args.language_column) or ""),
            "label_language_code": "",
            "audio_column": audio_column,
            "audio_path": "",
            "audio_sample_rate": None,
            "audio_duration_sec": None,
            "row_started_at_utc": row_started_at_utc,
            "row_finished_at_utc": "",
            "row_elapsed_ms": None,
            "lid_latency_ms": None,
            "status": "ok",
            "error": "",
            "language_match": False,
        }
        try:
            label_language_code = normalize_inventory_language(row.get(args.language_column), alias_map)
            if not label_language_code:
                raise ValueError(f"unsupported labeled language `{row.get(args.language_column)}`")

            audio_path = resolve_audio_path(row.get(audio_column), inventory_path)
            if not audio_path.exists():
                raise FileNotFoundError(str(audio_path))

            audio, sample_rate = load_audio(audio_path, max_audio_sec=args.max_audio_sec)
            mono = to_mono(np.asarray(audio, dtype=np.float32))
            duration_sec = round(float(mono.shape[0]) / float(sample_rate), 6) if sample_rate else 0.0
            if duration_sec < float(args.min_audio_sec):
                raise ValueError(
                    f"audio shorter than min duration after clipping ({duration_sec:.3f}s < {args.min_audio_sec:.3f}s)"
                )

            resampled = resample_linear(mono, sample_rate, 16000)
            lid_started_perf = time.perf_counter()
            detection = detector.identify_language(
                audio_bytes=float32_to_pcm16(resampled),
                sample_rate=16000,
                supported_languages=set(supported_languages),
            )
            lid_latency_ms = round((time.perf_counter() - lid_started_perf) * 1000.0, 3)
            result = {
                **base_result,
                **detection_to_row(detection),
                "label_language_code": label_language_code,
                "audio_path": str(audio_path),
                "audio_sample_rate": int(sample_rate),
                "audio_duration_sec": duration_sec,
                "lid_latency_ms": lid_latency_ms,
            }
            result["language_match"] = result["predicted_language"] == label_language_code
        except Exception as exc:
            result = {
                **base_result,
                **detection_to_row(None),
                "label_language_code": normalize_inventory_language(row.get(args.language_column), alias_map) or "",
                "status": "error",
                "error": str(exc),
            }
            audio_value = str(row.get(audio_column) or "").strip()
            if audio_value:
                try:
                    result["audio_path"] = str(resolve_audio_path(audio_value, inventory_path))
                except Exception:
                    result["audio_path"] = audio_value
            if args.fail_fast:
                raise

        result["row_finished_at_utc"] = utc_now_iso()
        result["row_elapsed_ms"] = round((time.perf_counter() - row_started_perf) * 1000.0, 3)

        results.append(result)
        if result["status"] == "ok":
            ok_count += 1
            if not result["language_match"]:
                mismatch_count += 1
        else:
            error_count += 1

        if progress is not None:
            progress.set_postfix(
                ok=ok_count,
                mismatch=mismatch_count,
                error=error_count,
                refresh=False,
            )
        elif args.log_every > 0 and (index == total_rows or index % args.log_every == 0):
            log.info(
                "Scanned %s/%s rows ok=%s mismatches=%s errors=%s",
                index,
                total_rows,
                ok_count,
                mismatch_count,
                error_count,
            )

    if progress is not None:
        progress.close()

    scan_finished_at_utc = utc_now_iso()
    scan_elapsed_sec = time.perf_counter() - scan_started_perf
    summary = summarize_results(
        results,
        inventory_path=inventory_path,
        audio_column=audio_column,
        supported_languages=supported_languages,
        scan_started_at_utc=scan_started_at_utc,
        scan_finished_at_utc=scan_finished_at_utc,
        scan_elapsed_sec=scan_elapsed_sec,
    )
    return results, summary


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    results, summary = run_scan(args)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results_path = output_dir / "inventory_lid_results.csv"
    mismatches_path = output_dir / "inventory_lid_mismatches.csv"
    summary_path = output_dir / "inventory_lid_summary.json"
    results_jsonl_path = output_dir / "inventory_lid_results.jsonl"
    mismatches_jsonl_path = output_dir / "inventory_lid_mismatches.jsonl"

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(all_results_path, index=False)
    results_frame.loc[
        (results_frame["status"] == "ok") & (~results_frame["language_match"].astype(bool))
    ].to_csv(mismatches_path, index=False)
    write_json(summary_path, summary)
    write_jsonl(results_jsonl_path, results)
    write_jsonl(
        mismatches_jsonl_path,
        [row for row in results if row["status"] == "ok" and not row["language_match"]],
    )

    log.info("Wrote %s", all_results_path)
    log.info("Wrote %s", mismatches_path)
    log.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
