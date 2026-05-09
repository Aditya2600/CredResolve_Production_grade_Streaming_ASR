#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from compute_wer import edit_distance, normalize
except ImportError:  # pragma: no cover
    from tools.compute_wer import edit_distance, normalize

from tools.asr_text_normalizer import normalize_asr_text
from worker.app.audio_processing import AudioPreprocessor
from worker.app.context_biasing import (
    ContextBiasingConfig,
    ContextBiasingDecision,
    NeMoContextBiasingRuntime,
    PhraseLexicon,
    should_return_active_biasing_transcript,
)
from worker.app.model import ONNXIndicASRWorker


TEXT_KEY_CANDIDATES = ("text", "reference", "normalized_text", "transcript", "sentence")
AUDIO_KEY_CANDIDATES = ("audio_filepath", "audio_path", "audio", "path")
LANGUAGE_KEY_CANDIDATES = ("lang", "language", "language_id")
DEFAULT_BASE_MODEL = "ai4bharat/indic-conformer-600m-multilingual"


def load_dotenv_defaults() -> dict[str, str]:
    dotenv_path = REPO_ROOT / ".env"
    values: dict[str, str] = {}
    if not dotenv_path.exists():
        return values
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


DOTENV = load_dotenv_defaults()


def env_default(name: str, default: str = "") -> str:
    return os.environ.get(name) or DOTENV.get(name) or default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct worker-model transcription on a manifest and compute WER."
    )
    parser.add_argument("--model-name", default=env_default("ASR_MODEL_NAME", DEFAULT_BASE_MODEL))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decoder", choices=("rnnt", "ctc"), default=env_default("ASR_DECODER", "rnnt"))
    parser.add_argument("--language", default=env_default("ASR_DEFAULT_LANGUAGE", "hi"))
    parser.add_argument("--supported-langs", default=env_default("ASR_SUPPORTED_LANGS", "hi,bn,ta,te,mr,gu,kn,ml,pa,or,as,ur"))
    parser.add_argument("--hf-token", default=env_default("HUGGINGFACE_HUB_TOKEN", env_default("HF_TOKEN", "")))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=env_default("ASR_DEVICE", "cuda"))
    parser.add_argument("--inference-timeout-ms", type=int, default=int(env_default("ASR_INFERENCE_TIMEOUT_MS", "4000") or "4000"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--text-field")
    parser.add_argument("--audio-field")
    parser.add_argument("--language-field")
    parser.add_argument("--reference-normalization", choices=("raw", "vaani"), default="vaani")
    parser.add_argument("--out-tsv", type=Path)
    parser.add_argument("--out-jsonl", type=Path)
    parser.add_argument("--out-summary-json", type=Path)
    parser.add_argument("--top-errors", type=int, default=20)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--processed-audio-dir", type=Path)
    parser.add_argument("--context-biasing-mode", choices=("disabled", "shadow", "active"), default="disabled")
    parser.add_argument("--context-biasing-source", default=env_default("ASR_CONTEXT_BIASING_NEMO_SOURCE", ""))
    parser.add_argument("--context-biasing-model-class", default=env_default("ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS", ""))
    parser.add_argument("--context-biasing-phrases-dir", type=Path, default=Path(env_default("ASR_CONTEXT_BIASING_PHRASES_DIR", "context_biasing/phrases")))
    parser.add_argument("--context-biasing-device", default=env_default("ASR_CONTEXT_BIASING_DEVICE", ""))
    parser.add_argument("--context-biasing-timeout-ms", type=int, default=int(env_default("ASR_CONTEXT_BIASING_TIMEOUT_MS", "8000") or "8000"))
    parser.add_argument("--context-biasing-beam-threshold", type=float, default=float(env_default("ASR_CONTEXT_BIASING_BEAM_THRESHOLD", "8.0") or "8.0"))
    parser.add_argument("--context-biasing-context-score", type=float, default=float(env_default("ASR_CONTEXT_BIASING_CONTEXT_SCORE", "3.0") or "3.0"))
    parser.add_argument("--context-biasing-ctc-ali-token-weight", type=float, default=float(env_default("ASR_CONTEXT_BIASING_CTC_ALI_TOKEN_WEIGHT", "0.6") or "0.6"))
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for lineno, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{resolved}:{lineno}: expected JSON object rows")
            rows.append(row)
        return rows
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise SystemExit(f"{resolved}: expected a JSON array of objects")
        return payload
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    raise SystemExit(f"Unsupported manifest format for {resolved}. Use .jsonl, .json, .csv, or .tsv")


def detect_key(rows: list[dict[str, Any]], candidates: tuple[str, ...], explicit: str | None) -> str:
    if explicit:
        return explicit
    if not rows:
        raise SystemExit("Manifest is empty")
    for key in candidates:
        if key in rows[0]:
            return key
    raise SystemExit(f"Could not auto-detect a key from: {', '.join(candidates)}")


def resolve_path(value: Any, manifest_path: Path) -> str:
    candidate = Path(str(value or "").strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return str(candidate)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value or "")).strip()


def clean_reference(text: str, mode: str) -> str:
    text = normalize_space(text)
    if mode == "raw":
        return text
    return normalize_asr_text(text)


def audio_to_float32(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


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
    dst_length = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def pcm16_to_float(pcm16le: bytes) -> np.ndarray:
    if not pcm16le:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm16le, dtype=np.int16).astype(np.float32) / 32768.0


def read_pcm16_mono(path: str | Path, *, sample_rate: int = 16000) -> tuple[bytes, int]:
    audio, source_sample_rate = audio_to_float32(path)
    audio = resample_linear(to_mono(audio), source_sample_rate, sample_rate)
    return float_to_pcm16(audio), sample_rate


def preprocess_audio_paths(
    audio_paths: list[str],
    *,
    denoise: bool,
    processed_audio_dir: Path | None,
) -> tuple[list[bytes], list[str], dict[str, Any]]:
    denoiser = AudioPreprocessor() if denoise else None
    output_dir = processed_audio_dir.expanduser().resolve() if processed_audio_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    pcm_rows: list[bytes] = []
    effective_paths: list[str] = []
    for index, audio_path in enumerate(audio_paths):
        pcm16le, sample_rate = read_pcm16_mono(audio_path, sample_rate=16000)
        if denoiser is not None:
            pcm16le = denoiser.process(pcm16le, sample_rate, vad_enabled=False, denoise_enabled=True)
        pcm_rows.append(pcm16le)
        if output_dir:
            output_path = output_dir / f"{index:06d}.wav"
            sf.write(output_path, pcm16_to_float(pcm16le), sample_rate, subtype="PCM_16")
            effective_paths.append(str(output_path))
        else:
            effective_paths.append(audio_path)

    metadata: dict[str, Any] = {"denoise": bool(denoise)}
    if output_dir:
        metadata["processed_audio_dir"] = str(output_dir)
    return pcm_rows, effective_paths, metadata


def parse_supported_languages(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in (value or "").split(",") if item.strip())


def build_worker(args: argparse.Namespace) -> ONNXIndicASRWorker:
    previous_device = os.environ.get("ASR_DEVICE")
    if args.device != "auto":
        os.environ["ASR_DEVICE"] = args.device
    worker = ONNXIndicASRWorker(
        model_name=args.model_name,
        default_decoder=args.decoder,
        hf_token=args.hf_token,
        inference_timeout_ms=args.inference_timeout_ms,
        default_language=args.language,
        supported_language_allowlist=parse_supported_languages(args.supported_langs),
        enable_lid=False,
        max_jobs=1,
    )
    worker.load()
    if previous_device is None:
        os.environ.pop("ASR_DEVICE", None)
    else:
        os.environ["ASR_DEVICE"] = previous_device
    return worker


def resolve_context_biasing_device(args: argparse.Namespace) -> str:
    explicit_device = str(args.context_biasing_device or "").strip().lower()
    if explicit_device:
        return explicit_device
    if args.device in {"cpu", "cuda"}:
        return str(args.device)
    return env_default("ASR_DEVICE", "cuda") or "cuda"


def build_context_biasing_runtime(args: argparse.Namespace) -> NeMoContextBiasingRuntime | None:
    if args.context_biasing_mode == "disabled":
        return None
    runtime = NeMoContextBiasingRuntime(
        ContextBiasingConfig(
            mode=args.context_biasing_mode,
            method="ctc_ws",
            nemo_source=args.context_biasing_source,
            nemo_model_class=args.context_biasing_model_class,
            phrases_dir=str(args.context_biasing_phrases_dir.expanduser().resolve()),
            timeout_ms=int(args.context_biasing_timeout_ms),
            device=resolve_context_biasing_device(args),
            shadow_sample_rate=1.0,
            beam_threshold=float(args.context_biasing_beam_threshold),
            context_score=float(args.context_biasing_context_score),
            ctc_ali_token_weight=float(args.context_biasing_ctc_ali_token_weight),
            max_dynamic_phrases=0,
        )
    )
    if importlib.util.find_spec("nemo") is None:
        runtime.ready = False
        runtime.init_error = (
            "NeMo ASR runtime is not installed in this Python environment; "
            "context biasing marked not_ready."
        )
        print(runtime.init_error, file=sys.stderr)
        return runtime
    runtime.load()
    return runtime


def empty_context_decision(mode: str, language: str, reason: str) -> ContextBiasingDecision:
    return ContextBiasingDecision(
        mode=mode,
        eligible=False,
        reason=reason,
        language=language,
        phrase_file=None,
    )


def apply_context_biasing(
    runtime: NeMoContextBiasingRuntime | None,
    baseline_result,
    pcm16le: bytes,
    *,
    sample_rate: int,
    requested_language: str,
    utterance_id: str,
    mode: str,
) -> tuple[Any, dict[str, Any]]:
    if runtime is None:
        return baseline_result, {}

    decision = runtime.decide(
        requested_language=requested_language,
        session_id="direct-serving-model-eval",
        utterance_id=utterance_id,
        requested_mode=mode,
        biasing_context=None,
    )
    biased_text: str | None = None
    fallback_reason: str | None = None
    returned_source = "baseline"
    selection_reason = decision.reason
    baseline_hits = 0
    biased_hits = 0

    if decision.eligible and decision.phrase_file:
        try:
            biased = runtime.transcribe_pcm16(
                pcm16le=pcm16le,
                sample_rate=sample_rate,
                language=decision.language,
                phrase_file=decision.phrase_file,
                session_id="direct-serving-model-eval",
                utterance_id=utterance_id,
                mode=mode,
            )
            biased_text = biased.text
        except Exception as exc:
            fallback_reason = str(exc)

    returned = baseline_result
    if mode == "shadow":
        selection_reason = "shadow_mode"
    elif mode == "active" and biased_text:
        lexicon = None
        if decision.phrase_file:
            try:
                lexicon = PhraseLexicon.from_file(decision.phrase_file, language=decision.language)
            except Exception:
                lexicon = None
        use_biased, selection_reason, baseline_hits, biased_hits = should_return_active_biasing_transcript(
            baseline_text=baseline_result.text,
            biased_text=biased_text,
            lexicon=lexicon,
        )
        if use_biased:
            returned = replace(baseline_result, text=biased_text)
            returned_source = "biased"

    if decision.cleanup_phrase_file and decision.phrase_file:
        Path(decision.phrase_file).unlink(missing_ok=True)

    return returned, {
        "context_biasing": {
            "mode": decision.mode,
            "reason": decision.reason,
            "eligible": decision.eligible,
            "phrase_file": decision.phrase_file,
            "phrase_source": decision.phrase_source,
            "returned_source": returned_source,
            "selection_reason": selection_reason,
            "fallback_reason": fallback_reason,
            "baseline_phrase_hits": baseline_hits,
            "biased_phrase_hits": biased_hits,
        },
        "baseline_hypothesis": baseline_result.text,
        "biased_hypothesis": biased_text,
    }


def total_edits(row: dict[str, Any]) -> int:
    return int(row["substitutions"]) + int(row["deletions"]) + int(row["insertions"])


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    rows = load_manifest(manifest_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        raise SystemExit("Manifest is empty after applying --limit")

    text_field = detect_key(rows, TEXT_KEY_CANDIDATES, args.text_field)
    audio_field = detect_key(rows, AUDIO_KEY_CANDIDATES, args.audio_field)
    language_field = args.language_field
    if language_field is None:
        for candidate in LANGUAGE_KEY_CANDIDATES:
            if candidate in rows[0]:
                language_field = candidate
                break

    original_audio_paths = [resolve_path(row.get(audio_field), manifest_path) for row in rows]
    pcm_rows, effective_audio_paths, preprocessing_metadata = preprocess_audio_paths(
        original_audio_paths,
        denoise=bool(args.denoise),
        processed_audio_dir=args.processed_audio_dir,
    )

    worker = build_worker(args)
    context_runtime = build_context_biasing_runtime(args)

    results: list[dict[str, Any]] = []
    total_s = total_d = total_i = total_words = 0
    for index, (row, pcm16le) in enumerate(zip(rows, pcm_rows)):
        row_language = args.language or (str(row.get(language_field, "") or "").strip() if language_field else "")
        baseline = worker.transcribe_pcm16(
            pcm16le=pcm16le,
            sample_rate=16000,
            decoder=args.decoder,
            language=row_language,
            session_id="direct-serving-model-eval",
            utterance_id=f"row-{index}",
            mode="final",
        )
        selected, metadata = apply_context_biasing(
            context_runtime,
            baseline,
            pcm16le,
            sample_rate=16000,
            requested_language=row_language,
            utterance_id=f"row-{index}",
            mode=args.context_biasing_mode,
        )

        raw_reference = str(row.get(text_field, "") or "").strip()
        reference = clean_reference(raw_reference, args.reference_normalization)
        hypothesis = selected.text
        ref_words = normalize(reference)
        hyp_words = normalize(hypothesis)
        s, d, ins = edit_distance(ref_words, hyp_words)
        total_s += s
        total_d += d
        total_i += ins
        total_words += len(ref_words)

        result = {
            "index": index,
            "audio_filepath": effective_audio_paths[index],
            "reference": reference,
            "raw_reference": raw_reference,
            "hypothesis": hypothesis,
            "reference_words": len(ref_words),
            "substitutions": s,
            "deletions": d,
            "insertions": ins,
            "sample_wer": ((s + d + ins) / len(ref_words)) if ref_words else 0.0,
            "language_id": selected.language,
            "language_source": selected.language_source,
        }
        if original_audio_paths[index] != effective_audio_paths[index]:
            result["original_audio_filepath"] = original_audio_paths[index]
        if metadata:
            result.update(metadata)
        if "source_id" in row:
            result["source_id"] = row["source_id"]
        elif "id" in row:
            result["source_id"] = row["id"]
        if "dataset_index" in row:
            result["dataset_index"] = row["dataset_index"]
        results.append(result)

    wer = ((total_s + total_d + total_i) / total_words) if total_words else 0.0
    summary = {
        "model_name": args.model_name,
        "snapshot_path": worker.snapshot_path,
        "manifest": str(manifest_path),
        "decoder": args.decoder,
        "language_id": args.language,
        "reference_normalization": args.reference_normalization,
        "audio_preprocessing": preprocessing_metadata,
        "context_biasing_mode": args.context_biasing_mode,
        "context_biasing_source": args.context_biasing_source,
        "context_biasing_model_class": args.context_biasing_model_class,
        "context_biasing_ready": bool(context_runtime.ready) if context_runtime is not None else False,
        "context_biasing_init_error": (
            context_runtime.init_error if context_runtime is not None and context_runtime.init_error else None
        ),
        "utterances": len(results),
        "reference_words": total_words,
        "substitutions": total_s,
        "deletions": total_d,
        "insertions": total_i,
        "wer": wer,
        "wer_percent": wer * 100,
    }

    ranked_results = sorted(results, key=lambda row: (-float(row["sample_wer"]), -total_edits(row), int(row["index"])))
    if args.out_tsv:
        path = args.out_tsv.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(f"{row['reference']}\t{row['hypothesis']}" for row in results) + "\n", encoding="utf-8")
    if args.out_jsonl:
        path = args.out_jsonl.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in ranked_results) + "\n", encoding="utf-8")
    if args.out_summary_json:
        path = args.out_summary_json.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"utterances={summary['utterances']}")
    print(f"reference_words={summary['reference_words']}")
    print(f"substitutions={summary['substitutions']}")
    print(f"deletions={summary['deletions']}")
    print(f"insertions={summary['insertions']}")
    print(f"wer={summary['wer']:.4f}")
    print(f"wer_percent={summary['wer_percent']:.2f}")
    for rank, row in enumerate(ranked_results[: max(0, args.top_errors)], start=1):
        print(
            f"rank={rank} sample={row['index']} sample_wer={row['sample_wer']:.4f} "
            f"s={row['substitutions']} d={row['deletions']} i={row['insertions']} "
            f"ref={json.dumps(row['reference'], ensure_ascii=False)} hyp={json.dumps(row['hypothesis'], ensure_ascii=False)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
