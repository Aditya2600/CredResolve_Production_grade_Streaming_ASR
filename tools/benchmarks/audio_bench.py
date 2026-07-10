"""
Benchmark harness for AudioPreprocessor + worker STT.

Reads WAV files from a directory and a reference transcript JSON
({filename: expected_text}), runs each clip through
AudioPreprocessor.process_with_stats() and then the worker's STT entry point,
computes WER per file and aggregate, plus per-stage latency p50/p95.

Outputs:
    <out_dir>/audio_bench_results.csv
    <out_dir>/audio_bench_summary.txt

Acceptance:
    python tools/benchmarks/audio_bench.py tests/fixtures/audio_bench/

The STT model is built via worker.app.main.build_worker_model(), so the
target backend (ONNX local vs Triton) is the same one production uses and
follows env config (ASR_BACKEND, ASR_MODEL_NAME, TRITON_URL, ...).

Pass --no-stt to bench preprocessing only (useful when no model is available
locally). Without --no-stt, STT is required: model startup failures stop the
benchmark instead of silently producing a no-STT run.

For producing a stable baseline (p50/p95 over many iterations), pass
``--iterations N`` (each fixture is processed N times) and ``--baseline-out
<dir>`` to additionally emit ``baseline.csv`` (raw per-iteration rows) and
``baseline.json`` (structured per-fixture aggregates) into that directory.
``regenerate_baseline.py`` under tests/fixtures/audio_bench/ is a thin
wrapper that pins those flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker.app._audio_ops import (
    int_bytes_to_int16,
    resample_int16,
    stereo_to_mono_int16,
)
from worker.app.audio_processing import AudioPreprocessor


TARGET_SAMPLE_RATE = 16000


def _log(*args, **kwargs) -> None:
    """Print benchmark progress immediately, even when stdout is piped to tee."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _denoiser_backend_name(preprocessor: AudioPreprocessor) -> str:
    denoiser = getattr(preprocessor, "rnnoise", None)
    if denoiser is None:
        return "none"
    class_name = type(denoiser).__name__
    if "DeepFilterNet" in class_name:
        return "deepfilternet"
    if "RNNoise" in class_name:
        return "rnnoise"
    return class_name


def read_wav_pcm16_mono(wav_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[bytes, int, float]:
    """Read a WAV file, downmix to mono and resample to target_sr.

    Returns (pcm_bytes, sample_rate, duration_seconds).
    """
    with wave.open(str(wav_path), "rb") as wf:
        sampwidth = wf.getsampwidth()
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth != 2:
        raw = int_bytes_to_int16(raw, sampwidth)
        sampwidth = 2

    if n_channels > 1:
        raw = stereo_to_mono_int16(raw)

    if sr != target_sr:
        raw = resample_int16(raw, sr, target_sr)
        sr = target_sr

    duration_s = (len(raw) / 2) / float(sr)
    return raw, sr, duration_s


_PUNCT = set(".,!?;:\"'()[]{}<>—–-")


def _normalize_text(text: str) -> list[str]:
    if not text:
        return []
    cleaned = "".join(" " if ch in _PUNCT else ch for ch in text)
    return cleaned.lower().split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein on whitespace-tokenized words.

    Returns 0.0 when both strings are empty, 1.0 when only hypothesis is empty
    against a non-empty reference.
    """
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[m] / float(n)


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _aggregate_wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus-level WER: total edits / total ref words."""
    total_ref = 0
    total_edits = 0
    for r, h in zip(refs, hyps):
        ref_tokens = _normalize_text(r)
        hyp_tokens = _normalize_text(h)
        total_ref += len(ref_tokens)
        if not ref_tokens and not hyp_tokens:
            continue
        if not ref_tokens:
            total_edits += len(hyp_tokens)
            continue
        n, m = len(ref_tokens), len(hyp_tokens)
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            prev = curr
        total_edits += prev[m]
    return (total_edits / total_ref) if total_ref else 0.0


def _build_stt_model():
    """Build and initialize the worker STT model used by the FastAPI app.

    main creates `model` at module scope, but FastAPI normally calls
    `model.load()` from its startup event. The benchmark runs as a one-off
    process, so it must perform that startup step itself.
    """
    from worker.app import main

    model = main.model
    if not getattr(model, "ready", False):
        model.load()
    return model


def _load_reference(reference_path: Optional[Path], wav_dir: Path) -> dict[str, str]:
    if reference_path is None:
        candidate = wav_dir / "reference_transcripts.json"
        if candidate.exists():
            reference_path = candidate
    if reference_path is None or not reference_path.exists():
        return {}
    with open(reference_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Reference JSON {reference_path} must be an object {{filename: text}}")
    return {str(k): str(v) for k, v in data.items()}


def _list_wavs(wav_dir: Path) -> list[Path]:
    return sorted(p for p in wav_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav_dir", type=Path, help="Directory of WAV files to bench")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Path to reference JSON {filename: text}. Defaults to <wav_dir>/reference_transcripts.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for CSV + summary output (defaults to <wav_dir>)",
    )
    parser.add_argument("--vad", dest="vad", action="store_true", default=True)
    parser.add_argument("--no-vad", dest="vad", action="store_false")
    parser.add_argument("--denoise", dest="denoise", action="store_true", default=True)
    parser.add_argument("--no-denoise", dest="denoise", action="store_false")
    parser.add_argument("--no-stt", action="store_true", help="Skip STT (preprocessing-only bench)")
    parser.add_argument(
        "--allow-stt-fallback",
        action="store_true",
        help=(
            "If STT startup fails, continue as a preprocessing-only run. "
            "By default, STT startup failure exits non-zero."
        ),
    )
    parser.add_argument("--decoder", default=None, help="Decoder for STT (default: ASR_DECODER from env)")
    parser.add_argument("--language", default=None, help="Language for STT (default: ASR_DEFAULT_LANGUAGE from env)")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Process each fixture N times (for stable p50/p95). Default: 1.",
    )
    parser.add_argument(
        "--baseline-out",
        type=Path,
        default=None,
        help=(
            "If set, also write baseline.csv (raw per-iteration rows) and "
            "baseline.json (per-fixture aggregates) into this directory."
        ),
    )
    args = parser.parse_args()
    if args.iterations < 1:
        print("error: --iterations must be >= 1", file=sys.stderr)
        return 2

    wav_dir: Path = args.wav_dir
    if not wav_dir.is_dir():
        print(f"error: {wav_dir} is not a directory", file=sys.stderr)
        return 2

    out_dir = args.out_dir or wav_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "audio_bench_results.csv"
    summary_path = out_dir / "audio_bench_summary.txt"

    references = _load_reference(args.reference, wav_dir)
    wav_paths = _list_wavs(wav_dir)
    if not wav_paths:
        print(f"error: no .wav files in {wav_dir}", file=sys.stderr)
        return 2

    _log(f"Loading AudioPreprocessor (vad={args.vad}, denoise={args.denoise}) ...")
    preprocessor = AudioPreprocessor()
    denoiser_backend = _denoiser_backend_name(preprocessor)
    _log(f"Loaded denoiser backend: {denoiser_backend}")

    model = None
    decoder = args.decoder
    language = args.language
    if not args.no_stt:
        try:
            _log("Building STT model via worker.app.main.build_worker_model() ...")
            model = _build_stt_model()
            from worker.app import config as worker_config
            if decoder is None:
                decoder = worker_config.ASR_DECODER
            if language is None:
                language = worker_config.ASR_DEFAULT_LANGUAGE
        except Exception as exc:
            if not args.allow_stt_fallback:
                print(
                    f"error: failed to build STT model ({exc}); "
                    "set ASR_MODEL_NAME/ASR_BACKEND/TRITON_URL correctly, "
                    "or pass --no-stt for a preprocessing-only run",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            print(
                f"warning: failed to build STT model ({exc}); continuing with --no-stt",
                file=sys.stderr,
                flush=True,
            )
            model = None

    rows: list[dict] = []
    vad_times_ms: list[float] = []
    denoise_times_ms: list[float] = []
    total_pre_times_ms: list[float] = []
    stt_times_ms: list[float] = []
    refs_for_corpus: list[str] = []
    hyps_for_corpus: list[str] = []
    per_fixture: dict[str, dict] = {}

    for wav_path in wav_paths:
        try:
            pcm, sr, duration_s = read_wav_pcm16_mono(wav_path, TARGET_SAMPLE_RATE)
        except Exception as exc:
            print(f"warning: skipping {wav_path.name}: {exc}", file=sys.stderr)
            continue

        fx = per_fixture.setdefault(
            wav_path.name,
            {
                "duration_s": duration_s,
                "vad_ms": [],
                "denoise_ms": [],
                "total_pre_ms": [],
                "stt_ms": [],
                "vad_segments": [],
                "vad_output_samples": [],
                "speech_ratio": [],
                "wers": [],
                "predicted_texts": [],
                "expected_text": references.get(wav_path.name, ""),
            },
        )

        for it in range(args.iterations):
            processed_pcm, stats = preprocessor.process_with_stats(
                pcm, sr, vad_enabled=args.vad, denoise_enabled=args.denoise
            )

            vad_ms = (stats["vad_seconds"] * 1000.0) if stats["vad_seconds"] is not None else None
            denoise_ms = (stats["denoise_seconds"] * 1000.0) if stats["denoise_seconds"] is not None else None
            total_ms = stats["total_seconds"] * 1000.0
            if vad_ms is not None:
                vad_times_ms.append(vad_ms)
                fx["vad_ms"].append(vad_ms)
            if denoise_ms is not None:
                denoise_times_ms.append(denoise_ms)
                fx["denoise_ms"].append(denoise_ms)
            total_pre_times_ms.append(total_ms)
            fx["total_pre_ms"].append(total_ms)
            if stats["vad_segments"] is not None:
                fx["vad_segments"].append(int(stats["vad_segments"]))
            if stats["vad_output_samples"] is not None:
                fx["vad_output_samples"].append(int(stats["vad_output_samples"]))
            if stats["speech_ratio"] is not None:
                fx["speech_ratio"].append(float(stats["speech_ratio"]))

            predicted_text = ""
            stt_ms: Optional[float] = None
            stt_error: Optional[str] = None
            if model is not None:
                stt_input = processed_pcm if processed_pcm else pcm
                stt_t0 = time.perf_counter()
                try:
                    result = model.transcribe_pcm16(
                        pcm16le=stt_input,
                        sample_rate=sr,
                        decoder=decoder,
                        language=language,
                        session_id=None,
                        utterance_id=f"{wav_path.stem}-{it}",
                        mode="final",
                        timestamp_type="none",
                    )
                    predicted_text = (getattr(result, "text", "") or "").strip()
                except Exception as exc:
                    stt_error = f"{type(exc).__name__}: {exc}"
                stt_ms = (time.perf_counter() - stt_t0) * 1000.0
                stt_times_ms.append(stt_ms)
                fx["stt_ms"].append(stt_ms)
                fx["predicted_texts"].append(predicted_text)

            expected_text = fx["expected_text"]
            per_file_wer: Optional[float] = None
            if model is not None and expected_text:
                per_file_wer = wer(expected_text, predicted_text)
                fx["wers"].append(per_file_wer)
                refs_for_corpus.append(expected_text)
                hyps_for_corpus.append(predicted_text)

            rows.append(
                {
                    "filename": wav_path.name,
                    "iteration": it,
                    "duration_s": f"{duration_s:.3f}",
                    "input_samples": stats["input_samples"],
                    "vad_segments": stats["vad_segments"],
                    "vad_output_samples": "" if stats["vad_output_samples"] is None else stats["vad_output_samples"],
                    "speech_ratio": "" if stats["speech_ratio"] is None else f"{stats['speech_ratio']:.4f}",
                    "vad_ms": "" if vad_ms is None else f"{vad_ms:.3f}",
                    "denoise_ms": "" if denoise_ms is None else f"{denoise_ms:.3f}",
                    "total_pre_ms": f"{total_ms:.3f}",
                    "stt_ms": "" if stt_ms is None else f"{stt_ms:.3f}",
                    "expected_text": expected_text,
                    "predicted_text": predicted_text,
                    "wer": "" if per_file_wer is None else f"{per_file_wer:.4f}",
                    "stt_error": stt_error or "",
                }
            )
            if it == 0 or args.iterations == 1:
                _log(
                    f"  {wav_path.name}[{it}]: pre_total={total_ms:.1f}ms "
                    f"vad={vad_ms if vad_ms is None else f'{vad_ms:.1f}'}ms "
                    f"denoise={denoise_ms if denoise_ms is None else f'{denoise_ms:.1f}'}ms "
                    f"segs={stats['vad_segments']} "
                    f"speech_ratio={stats['speech_ratio']} "
                    f"stt={stt_ms if stt_ms is None else f'{stt_ms:.1f}'}ms "
                    f"wer={per_file_wer if per_file_wer is None else f'{per_file_wer:.3f}'}"
                )
        if args.iterations > 1:
            _log(
                f"  {wav_path.name}: completed {args.iterations} iterations "
                f"(total_pre p50={_percentile(fx['total_pre_ms'], 50):.2f}ms, "
                f"p95={_percentile(fx['total_pre_ms'], 95):.2f}ms)"
            )

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _fmt_pct(values: list[float], pct: float) -> str:
        v = _percentile(values, pct)
        return "n/a" if v is None else f"{v:.2f}ms"

    def _fmt_mean(values: list[float]) -> str:
        return "n/a" if not values else f"{statistics.mean(values):.2f}ms"

    summary_lines: list[str] = []
    summary_lines.append("Audio bench summary")
    summary_lines.append("=" * 40)
    summary_lines.append(f"wav_dir              : {wav_dir}")
    summary_lines.append(f"files                : {len(rows)}")
    summary_lines.append(f"vad_enabled          : {args.vad}")
    summary_lines.append(f"denoise_enabled      : {args.denoise}")
    summary_lines.append(f"denoiser_backend     : {denoiser_backend if args.denoise else 'disabled'}")
    summary_lines.append(f"vad_select_mode      : {preprocessor.vad_select_mode}")
    summary_lines.append(f"stt_enabled          : {model is not None}")
    summary_lines.append("")
    summary_lines.append("Latency (ms):")
    summary_lines.append(f"  vad     mean={_fmt_mean(vad_times_ms)}  p50={_fmt_pct(vad_times_ms, 50)}  p95={_fmt_pct(vad_times_ms, 95)}  n={len(vad_times_ms)}")
    summary_lines.append(f"  denoise mean={_fmt_mean(denoise_times_ms)}  p50={_fmt_pct(denoise_times_ms, 50)}  p95={_fmt_pct(denoise_times_ms, 95)}  n={len(denoise_times_ms)}")
    summary_lines.append(f"  total   mean={_fmt_mean(total_pre_times_ms)}  p50={_fmt_pct(total_pre_times_ms, 50)}  p95={_fmt_pct(total_pre_times_ms, 95)}  n={len(total_pre_times_ms)}")
    summary_lines.append(f"  stt     mean={_fmt_mean(stt_times_ms)}  p50={_fmt_pct(stt_times_ms, 50)}  p95={_fmt_pct(stt_times_ms, 95)}  n={len(stt_times_ms)}")
    summary_lines.append("")
    if refs_for_corpus:
        per_file_wers = [
            float(r["wer"]) for r in rows if r.get("wer") not in ("", None)
        ]
        mean_wer = statistics.mean(per_file_wers) if per_file_wers else 0.0
        agg_wer = _aggregate_wer(refs_for_corpus, hyps_for_corpus)
        summary_lines.append("WER:")
        summary_lines.append(f"  per-file mean : {mean_wer:.4f} (n={len(per_file_wers)})")
        summary_lines.append(f"  aggregate     : {agg_wer:.4f}")
    else:
        summary_lines.append("WER: not computed (no STT or no matching references)")
    summary_lines.append("")
    summary_lines.append(f"CSV: {csv_path}")
    summary_lines.append(f"Summary: {summary_path}")

    summary_text = "\n".join(summary_lines) + "\n"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    _log()
    _log(summary_text)

    if args.baseline_out is not None:
        _write_baseline(
            baseline_dir=args.baseline_out,
            rows=rows,
            per_fixture=per_fixture,
            stt_enabled=model is not None,
            iterations=args.iterations,
        )

    return 0


def _write_baseline(
    *,
    baseline_dir: Path,
    rows: list[dict],
    per_fixture: dict[str, dict],
    stt_enabled: bool,
    iterations: int,
) -> None:
    """Persist raw per-iteration rows + per-fixture aggregates.

    baseline.csv: every iteration row exactly as in audio_bench_results.csv.
    baseline.json: {fixture_name: {wer, vad_p50_ms, vad_p95_ms, ...}}
    """
    baseline_dir.mkdir(parents=True, exist_ok=True)
    csv_path = baseline_dir / "baseline.csv"
    json_path = baseline_dir / "baseline.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _p(values: list[float], pct: float) -> Optional[float]:
        v = _percentile(values, pct)
        return None if v is None else round(v, 3)

    def _mean(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return round(statistics.mean(values), 4)

    fixtures_out: dict[str, dict] = {}
    for name, fx in per_fixture.items():
        wers = fx["wers"]
        n_pre = len(fx["total_pre_ms"])
        out_dur_ratio: Optional[float] = None
        if fx["vad_output_samples"]:
            mean_out_samples = statistics.mean(fx["vad_output_samples"])
            input_dur = fx["duration_s"]
            if input_dur > 0:
                out_dur_ratio = round((mean_out_samples / TARGET_SAMPLE_RATE) / input_dur, 4)
        fixtures_out[name] = {
            "n_iterations": n_pre,
            "duration_s": round(fx["duration_s"], 3),
            "wer_mean": _mean(wers) if wers else None,
            "wer_n": len(wers),
            "vad_p50_ms": _p(fx["vad_ms"], 50),
            "vad_p95_ms": _p(fx["vad_ms"], 95),
            "denoise_p50_ms": _p(fx["denoise_ms"], 50),
            "denoise_p95_ms": _p(fx["denoise_ms"], 95),
            "total_p50_ms": _p(fx["total_pre_ms"], 50),
            "total_p95_ms": _p(fx["total_pre_ms"], 95),
            "stt_p50_ms": _p(fx["stt_ms"], 50),
            "stt_p95_ms": _p(fx["stt_ms"], 95),
            "output_duration_ratio": out_dur_ratio,
            "n_segments_mean": (
                round(statistics.mean(fx["vad_segments"]), 3)
                if fx["vad_segments"]
                else None
            ),
        }

    payload = {
        "schema_version": 1,
        "stt_enabled": stt_enabled,
        "iterations_per_fixture": iterations,
        "fixtures": fixtures_out,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    _log(f"Baseline CSV : {csv_path}")
    _log(f"Baseline JSON: {json_path}")


if __name__ == "__main__":
    sys.exit(main())
