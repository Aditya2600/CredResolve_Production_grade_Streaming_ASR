#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np

try:  # Optional, but useful for non-WAV formats.
    import soundfile as sf
except Exception:  # pragma: no cover - depends on local environment
    sf = None


AUDIO_FIELD_CANDIDATES = (
    "audio_path",
    "audio",
    "path",
    "wav_path",
    "file",
    "filename",
)

LANGUAGE_FIELD_CANDIDATES = (
    "language",
    "language_code",
    "lang",
    "expected_language",
    "label",
    "target_language",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate worker-side LID by sending labeled audio clips to /v1/transcribe "
            "with x-language=auto and scoring the returned resolved language."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL/JSON/CSV/TSV manifest")
    parser.add_argument(
        "--worker-url",
        default="http://localhost:8001/v1/transcribe",
        help="Worker /v1/transcribe endpoint",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        choices=(8000, 16000),
        help="Sample rate sent to the worker after local resampling",
    )
    parser.add_argument("--decoder", default="rnnt", help="x-decoder header")
    parser.add_argument("--mode", default="final", help="x-mode header")
    parser.add_argument("--timeout-sec", type=float, default=15.0, help="HTTP timeout in seconds")
    parser.add_argument("--limit", type=int, help="Optional number of manifest rows to score")
    parser.add_argument("--audio-field", help="Override manifest audio path field")
    parser.add_argument("--language-field", help="Override manifest expected language field")
    parser.add_argument(
        "--session-prefix",
        default="lid-eval",
        help="Unique session prefix used to avoid LID cache contamination",
    )
    parser.add_argument(
        "--use-manifest-session-id",
        action="store_true",
        help="Respect a manifest session_id field instead of generating unique sessions per row",
    )
    parser.add_argument("--out-summary-json", type=Path, help="Optional summary JSON path")
    parser.add_argument("--out-results-jsonl", type=Path, help="Optional per-row results JSONL path")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
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
                raise SystemExit(f"{path}:{lineno}: expected each JSONL row to be an object")
            rows.append(row)
        return rows

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise SystemExit(f"{path}: expected a JSON array of objects")
        return list(data)

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]

    raise SystemExit(f"Unsupported manifest format for {path}. Use .jsonl, .json, .csv, or .tsv")


def detect_audio_field(sample: dict[str, Any]) -> str:
    for candidate in AUDIO_FIELD_CANDIDATES:
        value = sample.get(candidate)
        if isinstance(value, str) and value.strip():
            return candidate
        if isinstance(value, dict) and value.get("path"):
            return candidate
    raise SystemExit(
        "Unable to detect audio field. Pass --audio-field. "
        f"Available keys: {sorted(sample.keys())}"
    )


def detect_language_field(sample: dict[str, Any]) -> str:
    for candidate in LANGUAGE_FIELD_CANDIDATES:
        value = sample.get(candidate)
        if isinstance(value, str) and value.strip():
            return candidate
    raise SystemExit(
        "Unable to detect language field. Pass --language-field. "
        f"Available keys: {sorted(sample.keys())}"
    )


def resolve_path(value: str, manifest_path: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    return candidate


def resolve_audio_path(audio_value: Any, manifest_path: Path) -> Path:
    if isinstance(audio_value, dict) and audio_value.get("path"):
        audio_value = audio_value["path"]
    if not isinstance(audio_value, str) or not audio_value.strip():
        raise SystemExit(f"Unsupported audio value: {audio_value!r}")
    return resolve_path(audio_value, manifest_path)


def read_wav_bytes(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
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
    return audio, int(sample_rate)


def load_audio(audio_value: Any, manifest_path: Path) -> tuple[np.ndarray, int]:
    audio_path = resolve_audio_path(audio_value, manifest_path)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    if sf is not None:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    if audio_path.suffix.lower() != ".wav":
        raise SystemExit(
            f"{audio_path}: soundfile is unavailable, so only .wav inputs are supported"
        )
    return read_wav_bytes(audio_path)


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


def request_transcription(
    *,
    worker_url: str,
    pcm16le: bytes,
    sample_rate: int,
    decoder: str,
    mode: str,
    session_id: str,
    utterance_id: str,
    timeout_sec: float,
) -> tuple[int, dict[str, Any] | None, str | None]:
    request = urllib.request.Request(
        worker_url,
        data=pcm16le,
        method="POST",
        headers={
            "Content-Type": "application/octet-stream",
            "x-sample-rate": str(sample_rate),
            "x-decoder": decoder,
            "x-language": "auto",
            "x-mode": mode,
            "x-session-id": session_id,
            "x-utterance-id": utterance_id,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
            payload = json.loads(body) if body else {}
            if not isinstance(payload, dict):
                raise SystemExit("Worker returned a non-object JSON payload")
            return int(response.status), payload, None
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        return int(exc.code), None, error_body or str(exc)
    except urllib.error.URLError as exc:
        return 0, None, str(exc.reason)


def normalize_language(value: Any) -> str:
    return str(value or "").strip().lower()


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(values[lower])
    weight = rank - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def build_confusion_matrix(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in results:
        if row["status"] != "ok":
            continue
        expected = row["expected_language"]
        predicted = row["predicted_language"] or "<missing>"
        matrix.setdefault(expected, {})
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
    return {
        expected: dict(sorted(predictions.items(), key=lambda item: (item[0])))
        for expected, predictions in sorted(matrix.items(), key=lambda item: item[0])
    }


def build_per_language_metrics(results: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], float, float, float]:
    ok_rows = [row for row in results if row["status"] == "ok"]
    labels = sorted(
        {
            row["expected_language"]
            for row in ok_rows
            if row["expected_language"]
        }
        | {
            row["predicted_language"]
            for row in ok_rows
            if row.get("predicted_language")
        }
    )
    metrics: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label in labels:
        tp = sum(1 for row in ok_rows if row["expected_language"] == label and row["predicted_language"] == label)
        fp = sum(1 for row in ok_rows if row["expected_language"] != label and row["predicted_language"] == label)
        fn = sum(1 for row in ok_rows if row["expected_language"] == label and row["predicted_language"] != label)
        support = sum(1 for row in ok_rows if row["expected_language"] == label)
        predicted = sum(1 for row in ok_rows if row["predicted_language"] == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[label] = {
            "support": support,
            "predicted": predicted,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    if not labels:
        return {}, 0.0, 0.0, 0.0
    return (
        metrics,
        round(sum(precisions) / len(precisions), 6),
        round(sum(recalls) / len(recalls), 6),
        round(sum(f1s) / len(f1s), 6),
    )


def build_top_confusions(confusion_matrix: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expected, predictions in confusion_matrix.items():
        for predicted, count in predictions.items():
            if expected == predicted:
                continue
            rows.append(
                {
                    "expected_language": expected,
                    "predicted_language": predicted,
                    "count": count,
                }
            )
    rows.sort(key=lambda item: (-int(item["count"]), item["expected_language"], item["predicted_language"]))
    return rows


def count_by_key(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in results:
        value = str(row.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def summarize_results(
    results: list[dict[str, Any]],
    *,
    worker_url: str,
    manifest_path: Path,
    target_sample_rate: int,
) -> dict[str, Any]:
    total_rows = len(results)
    ok_rows = [row for row in results if row["status"] == "ok"]
    error_rows = [row for row in results if row["status"] != "ok"]
    correct_rows = [row for row in ok_rows if row["correct"]]
    latencies = sorted(float(row["latency_ms"]) for row in results if row.get("latency_ms") is not None)
    confusion_matrix = build_confusion_matrix(results)
    per_language, macro_precision, macro_recall, macro_f1 = build_per_language_metrics(results)
    fallback_default_count = sum(
        1
        for row in ok_rows
        if row.get("language_source") in {"auto_default", "lid_fallback_default"}
    )

    summary = {
        "manifest": str(manifest_path),
        "worker_url": worker_url,
        "target_sample_rate": target_sample_rate,
        "rows_total": total_rows,
        "rows_ok": len(ok_rows),
        "rows_error": len(error_rows),
        "request_success_rate": round((len(ok_rows) / total_rows), 6) if total_rows else 0.0,
        "accuracy": round((len(correct_rows) / len(ok_rows)), 6) if ok_rows else 0.0,
        "overall_correct_rate": round((len(correct_rows) / total_rows), 6) if total_rows else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "fallback_default_rate": round((fallback_default_count / len(ok_rows)), 6) if ok_rows else 0.0,
        "language_source_counts": count_by_key(ok_rows, "language_source"),
        "http_status_counts": count_by_key(results, "http_status"),
        "error_counts": count_by_key(error_rows, "error"),
        "confusion_matrix": confusion_matrix,
        "top_confusions": build_top_confusions(confusion_matrix),
        "per_language": per_language,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p50": round(percentile(latencies, 0.50), 3) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 3) if latencies else None,
            "p99": round(percentile(latencies, 0.99), 3) if latencies else None,
        },
    }
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_results_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [json.dumps(row, ensure_ascii=True, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(serialized) + ("\n" if serialized else ""), encoding="utf-8")


def evaluate_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = args.manifest.expanduser().resolve()
    rows = load_manifest(manifest_path)
    if args.limit is not None:
        rows = rows[: max(args.limit, 0)]
    if not rows:
        raise SystemExit("Manifest is empty after applying --limit")

    audio_field = args.audio_field or detect_audio_field(rows[0])
    language_field = args.language_field or detect_language_field(rows[0])
    results: list[dict[str, Any]] = []

    for offset, row in enumerate(rows, start=1):
        expected_language = normalize_language(row.get(language_field))
        if not expected_language:
            raise SystemExit(f"Manifest row {offset} is missing a language value in `{language_field}`")

        audio, source_rate = load_audio(row.get(audio_field), manifest_path)
        mono = to_mono(np.asarray(audio, dtype=np.float32))
        resampled = resample_linear(mono, source_rate, args.target_sample_rate)
        pcm16le = float32_to_pcm16(resampled)

        session_id = (
            str(row.get("session_id") or "").strip()
            if args.use_manifest_session_id
            else f"{args.session_prefix}-{offset:06d}"
        )
        utterance_id = str(row.get("utterance_id") or f"utt-{offset:06d}").strip()
        audio_path = resolve_audio_path(row.get(audio_field), manifest_path)

        started = time.perf_counter()
        http_status, payload, error_text = request_transcription(
            worker_url=args.worker_url,
            pcm16le=pcm16le,
            sample_rate=args.target_sample_rate,
            decoder=args.decoder,
            mode=args.mode,
            session_id=session_id or f"{args.session_prefix}-{offset:06d}",
            utterance_id=utterance_id,
            timeout_sec=args.timeout_sec,
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

        if payload is not None:
            predicted_language = normalize_language(payload.get("language"))
            language_source = str(payload.get("language_source") or "").strip()
            text = str(payload.get("text") or "")
            status = "ok"
            error = ""
            correct = predicted_language == expected_language
        else:
            predicted_language = ""
            language_source = ""
            text = ""
            status = "error"
            error = error_text or "request_failed"
            correct = False

        result = {
            "index": offset - 1,
            "expected_language": expected_language,
            "predicted_language": predicted_language,
            "language_source": language_source,
            "correct": correct,
            "status": status,
            "http_status": http_status,
            "latency_ms": latency_ms,
            "error": error,
            "response_text": text,
            "audio_path": str(audio_path),
            "input_sample_rate": source_rate,
            "request_sample_rate": args.target_sample_rate,
            "session_id": session_id or f"{args.session_prefix}-{offset:06d}",
            "utterance_id": utterance_id,
        }
        results.append(result)

    summary = summarize_results(
        results,
        worker_url=args.worker_url,
        manifest_path=manifest_path,
        target_sample_rate=args.target_sample_rate,
    )
    return results, summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"rows_total={summary['rows_total']}")
    print(f"rows_ok={summary['rows_ok']}")
    print(f"rows_error={summary['rows_error']}")
    print(f"request_success_rate={summary['request_success_rate']:.4f}")
    print(f"accuracy={summary['accuracy']:.4f}")
    print(f"overall_correct_rate={summary['overall_correct_rate']:.4f}")
    print(f"macro_precision={summary['macro_precision']:.4f}")
    print(f"macro_recall={summary['macro_recall']:.4f}")
    print(f"macro_f1={summary['macro_f1']:.4f}")
    print(f"fallback_default_rate={summary['fallback_default_rate']:.4f}")

    latency = summary["latency_ms"]
    if latency["mean"] is not None:
        print(
            "latency_ms_mean={mean:.3f} latency_ms_p50={p50:.3f} latency_ms_p95={p95:.3f} latency_ms_p99={p99:.3f}".format(
                mean=latency["mean"],
                p50=latency["p50"],
                p95=latency["p95"],
                p99=latency["p99"],
            )
        )

    if summary["language_source_counts"]:
        print("language_source_counts=" + json.dumps(summary["language_source_counts"], ensure_ascii=True, sort_keys=True))
    if summary["top_confusions"]:
        preview = summary["top_confusions"][:5]
        print("top_confusions=" + json.dumps(preview, ensure_ascii=True))


def main() -> None:
    args = parse_args()
    results, summary = evaluate_manifest(args)
    print_summary(summary)

    if args.out_summary_json:
        write_json(args.out_summary_json, summary)
    if args.out_results_jsonl:
        write_results_jsonl(args.out_results_jsonl, results)


if __name__ == "__main__":
    main()
