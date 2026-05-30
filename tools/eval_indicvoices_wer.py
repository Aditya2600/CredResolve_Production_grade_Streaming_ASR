#!/usr/bin/env python3
import argparse
import asyncio
import base64
import io
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
import soundfile as sf

try:
    import websockets
except ImportError:  # Optional for non-WebSocket helper imports.
    websockets = None

from tools.indicvoices_dataset import load_indicvoices_stream

try:
    from compute_wer import edit_distance, normalize
except ImportError:  # pragma: no cover - allows package-style imports from tests
    from tools.compute_wer import edit_distance, normalize


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

TEXT_CANDIDATES = (
    "normalized_text",
    "transcript",
    "transcription",
    "sentence",
    "text",
    "label",
)

DATASET_NOISE_BUCKET = "dataset-noise"
INFRA_FAILURE_BUCKET = "infra-failure"
ASR_BUCKET = "asr"
REQUEST_ERROR_BUCKET = "request-error"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate gateway ASR WER on ai4bharat/IndicVoices via the Sarvam-like websocket path."
    )
    parser.add_argument("--ws", default="ws://localhost/ws/stt", help="Gateway websocket endpoint")
    parser.add_argument("--api-key", default="dev", help="Api-Subscription-Key for the websocket handshake")
    parser.add_argument("--dataset-config", default="hindi", help="IndicVoices config, e.g. hindi")
    parser.add_argument("--split", default="valid", help="Dataset split, e.g. valid")
    parser.add_argument("--language", help="ASR language code sent as language-code; defaults from dataset config")
    parser.add_argument("--model", default="credresolve:v1", help="Public model query param")
    parser.add_argument("--mode", default="transcribe", help="Public mode query param")
    parser.add_argument("--limit", type=int, default=10, help="Number of dataset samples to score")
    parser.add_argument("--hf-token", help="Hugging Face token for gated IndicVoices access")
    parser.add_argument("--cache-dir", type=Path, help="Optional datasets cache dir")
    parser.add_argument("--text-field", help="Override transcript column name")
    parser.add_argument("--audio-field", help="Override audio column name")
    parser.add_argument("--out-tsv", type=Path, help="Optional output TSV with reference<TAB>hypothesis")
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        help="Optional JSON path for aggregate evaluation metrics",
    )
    parser.add_argument(
        "--out-errors-jsonl",
        type=Path,
        help="Optional JSONL path for per-sample results sorted by worst error first",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=20,
        help="Number of worst samples to print to stdout",
    )
    return parser.parse_args()


def resolve_hf_token(cli_token: str | None) -> str:
    token = (
        cli_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or load_dotenv_value("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        raise SystemExit(
            "IndicVoices is gated. Provide --hf-token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
        )
    return token


def resolve_language_code(dataset_config: str, cli_language: str | None) -> str:
    if cli_language:
        return cli_language
    code = LANGUAGE_MAP.get(dataset_config.strip().lower())
    if not code:
        raise SystemExit(
            f"Don't know how to map dataset config '{dataset_config}' to the gateway language code. "
            "Pass --language explicitly."
        )
    return code


def resolve_cache_dir(cli_cache_dir: Path | None) -> Path:
    cache_dir = cli_cache_dir or (Path.cwd() / ".cache" / "huggingface")
    cache_dir = cache_dir.expanduser().resolve()
    hub_cache = cache_dir / "hub"
    datasets_cache = cache_dir / "datasets"
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir.parent)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    return datasets_cache


def detect_text_field(sample: dict[str, Any]) -> str:
    for candidate in TEXT_CANDIDATES:
        value = sample.get(candidate)
        if isinstance(value, str) and value.strip():
            return candidate

    for key, value in sample.items():
        if isinstance(value, str) and value.strip():
            lowered = key.lower()
            if any(token in lowered for token in ("text", "trans", "sentence")):
                return key

    for key, value in sample.items():
        if isinstance(value, str) and value.strip():
            return key

    raise SystemExit(f"Unable to detect transcript field from sample keys: {sorted(sample.keys())}")


def detect_audio_field(sample: dict[str, Any]) -> str:
    for key, value in sample.items():
        if isinstance(value, dict):
            if "array" in value or "bytes" in value or "path" in value:
                return key
        if isinstance(value, (str, Path)) and str(value).lower().endswith((".wav", ".flac", ".mp3")):
            return key

    raise SystemExit(f"Unable to detect audio field from sample keys: {sorted(sample.keys())}")


def audio_to_float32(audio_value: Any) -> tuple[np.ndarray, int]:
    if isinstance(audio_value, dict):
        if "array" in audio_value and audio_value["array"] is not None:
            arr = np.asarray(audio_value["array"], dtype=np.float32)
            sr = int(audio_value.get("sampling_rate") or 16000)
            return arr, sr
        if audio_value.get("bytes") is not None:
            arr, sr = sf.read(io.BytesIO(audio_value["bytes"]), dtype="float32")
            return np.asarray(arr, dtype=np.float32), int(sr)
        if audio_value.get("path"):
            arr, sr = sf.read(audio_value["path"], dtype="float32")
            return np.asarray(arr, dtype=np.float32), int(sr)

    if isinstance(audio_value, (str, Path)):
        arr, sr = sf.read(str(audio_value), dtype="float32")
        return np.asarray(arr, dtype=np.float32), int(sr)

    raise SystemExit(f"Unsupported audio payload type: {type(audio_value)!r}")


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1, dtype=np.float32)


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    src_positions = np.arange(audio.shape[0], dtype=np.float32)
    dst_length = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def write_temp_wav(audio: np.ndarray, sample_rate: int) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="indicvoices_", suffix=".wav", delete=False)
    handle.close()
    path = Path(handle.name)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path


def sample_wer(substitutions: int, deletions: int, insertions: int, ref_word_count: int) -> float:
    if ref_word_count <= 0:
        return 0.0
    return (substitutions + deletions + insertions) / ref_word_count


def total_edits(result: dict[str, Any]) -> int:
    return int(result["substitutions"]) + int(result["deletions"]) + int(result["insertions"])


def classify_result_bucket(reference: str, hypothesis: str, status: str) -> str:
    if status != "ok":
        return REQUEST_ERROR_BUCKET
    if "worker-fallback" in (hypothesis or "").lower():
        return INFRA_FAILURE_BUCKET
    if "<unintelligible>" in (reference or "").lower():
        return DATASET_NOISE_BUCKET
    return ASR_BUCKET


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(result: dict[str, Any]) -> tuple[int, int, float, int, int]:
        if result["status"] != "ok":
            return (2, 0, 0.0, 0, int(result["index"]))
        if not result.get("counted_in_clean_wer", True):
            return (1, 0, -float(result["sample_wer"]), -total_edits(result), int(result["index"]))
        return (
            0,
            0,
            -float(result["sample_wer"]),
            -total_edits(result),
            int(result["index"]),
        )

    return sorted(results, key=sort_key)


def ensure_parent_dir(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def print_top_errors(results: list[dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        return

    worst = [
        result
        for result in results
        if result["status"] == "ok"
        and result.get("counted_in_clean_wer", True)
        and total_edits(result) > 0
    ][:limit]
    if not worst:
        print("top_errors=0 detail=no_nonzero_clean_asr_sample_errors")
        return

    print(f"top_errors={len(worst)}")
    for rank, result in enumerate(worst, start=1):
        print(
            f"rank={rank} sample={result['index']} sample_wer={result['sample_wer']:.4f} "
            f"edits={total_edits(result)} s={result['substitutions']} d={result['deletions']} "
            f"i={result['insertions']} ref_words={len(result['ref_words'])} "
            f"hyp_words={len(result['hyp_words'])}"
        )
        print(f"  ref={json.dumps(result['reference'], ensure_ascii=False)}")
        print(f"  hyp={json.dumps(result['hypothesis'], ensure_ascii=False)}")


def update_ws_query(base_ws_url: str, params: dict[str, str]) -> str:
    parsed = urlparse(base_ws_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    return urlunparse(parsed._replace(query=urlencode(query)))


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * ratio))))
    return float(sorted_values[index])


async def transcribe_wav(
    *,
    ws_uri: str,
    wav_path: Path,
    api_key: str,
    call_id: str,
    language: str,
    model: str,
    mode: str,
 ) -> dict[str, Any]:
    final_texts: list[str] = []
    processing_latencies: list[float] = []
    audio_durations: list[float] = []
    ws_url = update_ws_query(
        ws_uri,
        {
            "language-code": language,
            "model": model,
            "mode": mode,
            "sample_rate": "16000",
            "high_vad_sensitivity": "false",
            "vad_signals": "false",
            "flush_signal": "true",
            "input_audio_codec": "wav",
        },
    )
    auth_headers = {"Api-Subscription-Key": api_key}
    started_at = time.time()

    if websockets is None:
        raise RuntimeError("The websockets package is required for WebSocket ASR evaluation. Install it with: pip install websockets")

    async with websockets.connect(ws_url, max_size=20_000_000, additional_headers=auth_headers) as ws:
        wav_bytes = wav_path.read_bytes()
        await ws.send(
            json.dumps(
                {
                    "audio": {
                        "data": base64.b64encode(wav_bytes).decode("ascii"),
                        "sample_rate": "16000",
                        "encoding": "wav",
                    }
                }
            )
        )
        await ws.send(json.dumps({"type": "flush"}))

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                break
            event = json.loads(raw)
            if event.get("type") == "data":
                data = event.get("data", {}) or {}
                transcript = str(data.get("transcript", "")).strip()
                if transcript:
                    final_texts.append(transcript)
                metrics = data.get("metrics", {}) or {}
                processing_latency = metrics.get("processing_latency")
                audio_duration = metrics.get("audio_duration")
                if isinstance(processing_latency, (int, float)):
                    processing_latencies.append(float(processing_latency))
                if isinstance(audio_duration, (int, float)):
                    audio_durations.append(float(audio_duration))
            elif event.get("type") == "error":
                raise RuntimeError(f"gateway error for {call_id}: {event}")

    return {
        "text": " ".join(part for part in final_texts if part).strip(),
        "client_latency_sec": max(0.0, time.time() - started_at),
        "server_processing_latency_sec": sum(processing_latencies) if processing_latencies else None,
        "server_audio_duration_sec": sum(audio_durations) if audio_durations else None,
        "data_events": len(final_texts),
    }


async def run() -> None:
    args = parse_args()
    token = resolve_hf_token(args.hf_token)
    language = resolve_language_code(args.dataset_config, args.language)
    cache_dir = resolve_cache_dir(args.cache_dir)
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

    text_field = args.text_field or detect_text_field(first_sample)
    audio_field = args.audio_field or detect_audio_field(first_sample)

    samples = [(0, first_sample)]
    for index, sample in enumerate(iterator, start=1):
        if index >= args.limit:
            break
        samples.append((index, sample))

    total_s = total_d = total_i = total_words = 0
    clean_total_s = clean_total_d = clean_total_i = clean_total_words = 0
    processed = 0
    failures = 0
    dataset_noise_samples = 0
    infra_failure_samples = 0
    clean_asr_processed = 0
    tsv_lines: list[str] = []
    sample_results: list[dict[str, Any]] = []
    client_latencies: list[float] = []
    server_processing_latencies: list[float] = []

    print(
        f"dataset=ai4bharat/IndicVoices config={args.dataset_config} split={args.split} "
        f"limit={args.limit} language={language} cache_dir={cache_dir} "
        f"text_field={text_field} audio_field={audio_field}"
    )

    for index, sample in samples:
        reference = str(sample[text_field]).strip()
        if not reference:
            print(f"skip index={index} reason=empty_reference")
            continue

        ref_words = normalize(reference)
        temp_wav: Path | None = None
        try:
            audio, sr = audio_to_float32(sample[audio_field])
            audio = to_mono(audio)
            audio = resample_linear(audio, sr, 16000)
            temp_wav = write_temp_wav(audio, 16000)

            transcript_result = await transcribe_wav(
                ws_uri=args.ws,
                wav_path=temp_wav,
                api_key=args.api_key,
                call_id=f"indicvoices-{args.dataset_config}-{index}",
                language=language,
                model=args.model,
                mode=args.mode,
            )
            hypothesis = str(transcript_result.get("text", "")).strip()

            hyp_words = normalize(hypothesis)
            s, d, ins = edit_distance(ref_words, hyp_words)
            evaluation_bucket = classify_result_bucket(reference, hypothesis, "ok")
            counted_in_clean_wer = evaluation_bucket == ASR_BUCKET
            result = {
                "index": index,
                "reference": reference,
                "hypothesis": hypothesis,
                "ref_words": ref_words,
                "hyp_words": hyp_words,
                "substitutions": s,
                "deletions": d,
                "insertions": ins,
                "sample_wer": sample_wer(s, d, ins, len(ref_words)),
                "client_latency_sec": transcript_result.get("client_latency_sec"),
                "server_processing_latency_sec": transcript_result.get("server_processing_latency_sec"),
                "server_audio_duration_sec": transcript_result.get("server_audio_duration_sec"),
                "data_events": transcript_result.get("data_events"),
                "evaluation_bucket": evaluation_bucket,
                "counted_in_clean_wer": counted_in_clean_wer,
                "status": "ok",
            }
            total_s += s
            total_d += d
            total_i += ins
            total_words += len(ref_words)
            if evaluation_bucket == DATASET_NOISE_BUCKET:
                dataset_noise_samples += 1
            elif evaluation_bucket == INFRA_FAILURE_BUCKET:
                infra_failure_samples += 1
            else:
                clean_total_s += s
                clean_total_d += d
                clean_total_i += ins
                clean_total_words += len(ref_words)
                clean_asr_processed += 1
            processed += 1
            if isinstance(result["client_latency_sec"], (int, float)):
                client_latencies.append(float(result["client_latency_sec"]))
            if isinstance(result["server_processing_latency_sec"], (int, float)):
                server_processing_latencies.append(float(result["server_processing_latency_sec"]))
            tsv_lines.append(f"{reference}	{hypothesis}")
            sample_results.append(result)
            print(
                f"sample={index} ref_words={len(ref_words)} hyp_words={len(hyp_words)} "
                f"s={s} d={d} i={ins} sample_wer={result['sample_wer']:.4f} "
                f"hyp={json.dumps(hypothesis, ensure_ascii=False)}"
            )
        except Exception as exc:
            failures += 1
            sample_results.append(
                {
                    "index": index,
                    "reference": reference,
                    "hypothesis": "",
                    "ref_words": ref_words,
                    "hyp_words": [],
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": 0,
                    "sample_wer": None,
                    "evaluation_bucket": REQUEST_ERROR_BUCKET,
                    "counted_in_clean_wer": False,
                    "status": "error",
                    "error_detail": str(exc),
                }
            )
            print(f"sample={index} status=error detail={exc}")
        finally:
            if temp_wav is not None:
                temp_wav.unlink(missing_ok=True)

    wer = ((total_s + total_d + total_i) / total_words) if total_words else 0.0
    clean_wer = (
        (clean_total_s + clean_total_d + clean_total_i) / clean_total_words
        if clean_total_words
        else 0.0
    )
    ranked_results = rank_results(sample_results)
    summary = {
        "dataset": "ai4bharat/IndicVoices",
        "dataset_config": args.dataset_config,
        "split": args.split,
        "limit": args.limit,
        "language": language,
        "processed": processed,
        "failures": failures,
        "reference_words": total_words,
        "substitutions": total_s,
        "deletions": total_d,
        "insertions": total_i,
        "wer": wer,
        "wer_percent": wer * 100,
        "dataset_noise_samples": dataset_noise_samples,
        "infra_failure_samples": infra_failure_samples,
        "clean_asr_processed": clean_asr_processed,
        "clean_reference_words": clean_total_words,
        "clean_substitutions": clean_total_s,
        "clean_deletions": clean_total_d,
        "clean_insertions": clean_total_i,
        "clean_wer": clean_wer,
        "clean_wer_percent": clean_wer * 100,
        "avg_client_latency_sec": (sum(client_latencies) / len(client_latencies)) if client_latencies else None,
        "p95_client_latency_sec": percentile(client_latencies, 0.95),
        "avg_server_processing_latency_sec": (
            (sum(server_processing_latencies) / len(server_processing_latencies))
            if server_processing_latencies
            else None
        ),
        "p95_server_processing_latency_sec": percentile(server_processing_latencies, 0.95),
    }

    if args.out_tsv:
        ensure_parent_dir(args.out_tsv)
        args.out_tsv.write_text("\n".join(tsv_lines) + ("\n" if tsv_lines else ""), encoding="utf-8")

    if args.out_summary_json:
        ensure_parent_dir(args.out_summary_json)
        args.out_summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.out_errors_jsonl:
        ensure_parent_dir(args.out_errors_jsonl)
        args.out_errors_jsonl.write_text(
            "\n".join(json.dumps(result, ensure_ascii=False) for result in ranked_results)
            + ("\n" if ranked_results else ""),
            encoding="utf-8",
        )

    print(f"processed={processed}")
    print(f"failures={failures}")
    print(f"reference_words={total_words}")
    print(f"substitutions={total_s}")
    print(f"deletions={total_d}")
    print(f"insertions={total_i}")
    print(f"wer={wer:.4f}")
    print(f"wer_percent={wer * 100:.2f}")
    print(f"dataset_noise_samples={dataset_noise_samples}")
    print(f"infra_failure_samples={infra_failure_samples}")
    print(f"clean_asr_processed={clean_asr_processed}")
    print(f"clean_reference_words={clean_total_words}")
    print(f"clean_substitutions={clean_total_s}")
    print(f"clean_deletions={clean_total_d}")
    print(f"clean_insertions={clean_total_i}")
    print(f"clean_wer={clean_wer:.4f}")
    print(f"clean_wer_percent={clean_wer * 100:.2f}")
    print_top_errors(ranked_results, args.top_errors)


if __name__ == "__main__":
    asyncio.run(run())
