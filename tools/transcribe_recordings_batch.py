#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from tools.diarization.config import DiarizationConfig
from tools.diarization.io import write_rttm
from tools.diarization.providers import NeMoTelephonyDiarizationProvider
from tools.diarization.providers.base import PreparedAudio, SpeakerTurn

try:
    import soxr
except ImportError:  # pragma: no cover
    soxr = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass(frozen=True)
class AudioJob:
    key: str
    audio_path: Path
    url: str | None = None
    row_indices: tuple[int, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch transcribe recordings with LID probe, denoise, context biasing, and diarization."
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--url-column", default="cr_recording_url")
    parser.add_argument("--id-column", default="job_id")
    parser.add_argument("--start-row", type=int, default=1, help="1-based input row to start from, excluding the header.")
    parser.add_argument("--limit", type=int, help="Maximum input rows/files to include after --start-row.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--clean-csv", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--worker-url", default="http://127.0.0.1:9000/v1/transcribe")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--decoder", default="ctc", choices=("ctc", "rnnt"))
    parser.add_argument("--fallback-language", default="hi")
    parser.add_argument("--max-speakers", type=int, default=2)
    parser.add_argument("--fixed-speakers", type=int)
    parser.add_argument("--chunk-sec", type=float, default=25.0)
    parser.add_argument("--diarization-device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--concurrency", type=int, default=1, help="Number of files to process in parallel.")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress when available. Default: enabled",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.expanduser().resolve().iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def url_audio_name(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    suffix = Path(name).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        suffix = ".mp3"
    stem = sanitize_stem(Path(name or "recording"))
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{digest}{suffix}"


def jobs_from_csv(rows: list[dict[str, str]], *, url_column: str, download_dir: Path) -> list[AudioJob]:
    by_url: dict[str, list[int]] = {}
    for index, row in enumerate(rows, start=1):
        url = str(row.get(url_column) or "").strip()
        if not url:
            continue
        by_url.setdefault(url, []).append(index)
    return [
        AudioJob(
            key=url,
            url=url,
            audio_path=download_dir / url_audio_name(url),
            row_indices=tuple(indices),
        )
        for url, indices in by_url.items()
    ]


def jobs_from_dir(input_dir: Path) -> list[AudioJob]:
    return [
        AudioJob(key=str(path.resolve()), audio_path=path, row_indices=(index,))
        for index, path in enumerate(audio_files(input_dir), start=1)
    ]


def window_rows(rows: list[dict[str, str]], *, start_row: int, limit: int | None) -> list[dict[str, str]]:
    start_index = max(0, start_row - 1)
    if limit is None:
        return rows[start_index:]
    return rows[start_index : start_index + max(0, limit)]


def window_jobs(jobs: list[AudioJob], *, start_row: int, limit: int | None) -> list[AudioJob]:
    start_index = max(0, start_row - 1)
    if limit is None:
        return jobs[start_index:]
    return jobs[start_index : start_index + max(0, limit)]


def download_audio(job: AudioJob, *, timeout_sec: float, retries: int) -> None:
    if job.url is None:
        return
    if job.audio_path.exists() and job.audio_path.stat().st_size > 0:
        return

    job.audio_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = job.audio_path.with_suffix(job.audio_path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            request = urllib.request.Request(job.url, headers={"User-Agent": "CredResolve-ASR-batch/1.0"})
            with urllib.request.urlopen(request, timeout=timeout_sec) as response, tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp_path.replace(job.audio_path)
            return
        except Exception as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(10.0, 1.5 * attempt))
    raise RuntimeError(f"download_failed:{last_error}")


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1, dtype=np.float32)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if int(src_sr) == int(dst_sr):
        return np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if soxr is not None:
        return soxr.resample(audio, int(src_sr), int(dst_sr)).astype(np.float32)

    src_positions = np.arange(audio.shape[0], dtype=np.float32)
    dst_length = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def load_normalized_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    audio, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    mono = to_mono(audio)
    normalized = resample(mono, int(src_sr), int(sample_rate))
    return np.clip(normalized, -1.0, 1.0).astype(np.float32), int(sample_rate)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return np.clip(np.rint(np.asarray(audio) * 32767.0), -32768, 32767).astype("<i2").tobytes()


def post_worker(
    worker_url: str,
    pcm16le: bytes,
    *,
    sample_rate: int,
    decoder: str,
    language: str,
    session_id: str,
    utterance_id: str,
    denoise: bool,
    context_biasing: bool,
    timestamp_type: str = "none",
    timeout_sec: float,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Sample-Rate": str(sample_rate),
        "X-Decoder": decoder,
        "X-Language": language,
        "X-Mode": "final",
        "X-Session-Id": session_id,
        "X-Utterance-Id": utterance_id,
        "X-Denoise-Enabled": "true" if denoise else "false",
        "X-VAD-Enabled": "false",
        "X-Timestamp-Type": timestamp_type,
    }
    if context_biasing:
        headers["X-Context-Biasing-Request"] = json.dumps(
            {"context_biasing": {"mode": "active"}},
            separators=(",", ":"),
        )

    request = urllib.request.Request(worker_url, data=pcm16le, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
    payload = json.loads(body or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("worker returned a non-object response")
    return payload


def safe_language(payload: dict[str, Any], fallback: str) -> str:
    language = str(payload.get("language") or "").strip().lower()
    if language and language != "auto":
        return language
    return fallback


def run_diarization(
    audio_path: Path,
    *,
    waveform: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    max_speakers: int,
    fixed_speakers: int | None,
    device: str,
) -> tuple[list[SpeakerTurn], Path | None, str | None]:
    fixed_mode = fixed_speakers is not None
    diar_config = DiarizationConfig(
        speaker_count_mode="fixed" if fixed_mode else "estimate",
        fixed_speakers=fixed_speakers,
        max_speakers=max_speakers,
        sample_rate=sample_rate,
        device=device,
    )
    work_dir = output_dir / "nemo_work" / audio_path.stem
    prepared_audio = PreparedAudio(
        audio_path=audio_path,
        waveform=np.asarray(waveform, dtype=np.float32),
        sample_rate=sample_rate,
        source_audio_path=audio_path,
        source_kind="normalized_mono_16k",
    )
    try:
        result = NeMoTelephonyDiarizationProvider().diarize(
            prepared_audio,
            diar_config,
            working_dir=work_dir,
        )
    except Exception as exc:
        return [], None, f"diarization_failed:{exc}"

    turns = list(result.turns)
    if not turns:
        return [], None, "diarization_no_speech"

    rttm_path = output_dir / "rttm" / f"{audio_path.stem}.rttm"
    write_rttm(rttm_path, file_id=audio_path.stem, turns=turns)
    return turns, rttm_path, None


def word_start_end(word: dict[str, Any]) -> tuple[float | None, float | None]:
    start = word.get("start_time", word.get("start", word.get("start_sec")))
    end = word.get("end_time", word.get("end", word.get("end_sec")))
    try:
        return float(start), float(end)
    except (TypeError, ValueError):
        return None, None


def label_for_word(word: dict[str, Any], turns: list[SpeakerTurn]) -> str | None:
    start, end = word_start_end(word)
    if start is None or end is None:
        return None
    midpoint = (start + end) / 2.0
    for turn in turns:
        if turn.start_sec <= midpoint <= turn.end_sec:
            return turn.speaker_label
    return None


def offset_word_timestamps(words: list[dict[str, Any]], offset_sec: float) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for word in words:
        item = dict(word)
        for key in ("start_time", "start", "start_sec"):
            if key in item:
                try:
                    item[key] = round(float(item[key]) + offset_sec, 4)
                except (TypeError, ValueError):
                    pass
        for key in ("end_time", "end", "end_sec"):
            if key in item:
                try:
                    item[key] = round(float(item[key]) + offset_sec, 4)
                except (TypeError, ValueError):
                    pass
        shifted.append(item)
    return shifted


def transcribe_chunks(
    *,
    worker_url: str,
    pcm16le: bytes,
    sample_rate: int,
    decoder: str,
    language: str,
    session_id: str,
    utterance_prefix: str,
    denoise: bool,
    context_biasing: bool,
    timestamp_type: str,
    timeout_sec: float,
    chunk_sec: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    samples = np.frombuffer(pcm16le, dtype="<i2")
    chunk_samples = max(int(sample_rate * chunk_sec), sample_rate)
    chunk_latencies_ms: list[int] = []
    if samples.size <= chunk_samples:
        request_started = time.perf_counter()
        payload = post_worker(
            worker_url,
            pcm16le,
            sample_rate=sample_rate,
            decoder=decoder,
            language=language,
            session_id=session_id,
            utterance_id=utterance_prefix,
            denoise=denoise,
            context_biasing=context_biasing,
            timestamp_type=timestamp_type,
            timeout_sec=timeout_sec,
        )
        chunk_latencies_ms.append(int((time.perf_counter() - request_started) * 1000))
        words = list(payload.get("word_timestamps") or [])
        payload["request_latency_ms"] = int(sum(chunk_latencies_ms))
        payload["chunk_request_latency_ms"] = chunk_latencies_ms
        return payload, [payload], words

    chunk_payloads: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []
    texts: list[str] = []
    for chunk_index, start in enumerate(range(0, samples.size, chunk_samples), start=1):
        stop = min(start + chunk_samples, samples.size)
        chunk = samples[start:stop].astype("<i2", copy=False).tobytes()
        request_started = time.perf_counter()
        payload = post_worker(
            worker_url,
            chunk,
            sample_rate=sample_rate,
            decoder=decoder,
            language=language,
            session_id=session_id,
            utterance_id=f"{utterance_prefix}-chunk{chunk_index:03d}",
            denoise=denoise,
            context_biasing=context_biasing,
            timestamp_type=timestamp_type,
            timeout_sec=timeout_sec,
        )
        chunk_latencies_ms.append(int((time.perf_counter() - request_started) * 1000))
        chunk_payloads.append(payload)
        text = str(payload.get("text") or "").strip()
        if text:
            texts.append(text)
        all_words.extend(offset_word_timestamps(list(payload.get("word_timestamps") or []), start / sample_rate))

    first = dict(chunk_payloads[0]) if chunk_payloads else {}
    first["text"] = " ".join(texts).strip()
    first["metrics"] = {
        "chunks": len(chunk_payloads),
        "audio_duration": round(samples.size / float(sample_rate), 4),
    }
    first["chunk_responses"] = chunk_payloads
    first["request_latency_ms"] = int(sum(chunk_latencies_ms))
    first["chunk_request_latency_ms"] = chunk_latencies_ms
    return first, chunk_payloads, all_words


def extract_word_text(word: dict[str, Any]) -> str:
    for key in ("word", "text", "token"):
        value = str(word.get(key) or "").strip()
        if value:
            return value
    return ""


def speaker_segments(words: list[dict[str, Any]], turns: list[SpeakerTurn], fallback_text: str) -> list[dict[str, Any]]:
    if not words:
        return [{"speaker": None, "text": fallback_text.strip(), "start_sec": None, "end_sec": None}]

    segments: list[dict[str, Any]] = []
    current_speaker: str | None = None
    current_words: list[str] = []
    current_start: float | None = None
    current_end: float | None = None

    def flush() -> None:
        nonlocal current_speaker, current_words, current_start, current_end
        if current_words:
            segments.append(
                {
                    "speaker": current_speaker,
                    "text": " ".join(current_words).strip(),
                    "start_sec": current_start,
                    "end_sec": current_end,
                }
            )
        current_speaker = None
        current_words = []
        current_start = None
        current_end = None

    for word in words:
        text = extract_word_text(word)
        if not text:
            continue
        speaker = label_for_word(word, turns)
        start, end = word_start_end(word)
        if current_words and speaker != current_speaker:
            flush()
        if not current_words:
            current_speaker = speaker
            current_start = start
        current_words.append(text)
        current_end = end
    flush()
    return segments


def write_txt(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"file: {result['file']}",
        f"language: {result.get('language')} ({result.get('language_source')})",
        f"lid_probe: {result.get('lid_language')} ({result.get('lid_language_source')})",
        f"context_biasing: {json.dumps(result.get('context_biasing'), ensure_ascii=False)}",
        "",
        "transcript:",
        str(result.get("text") or "").strip(),
        "",
        "diarized:",
    ]
    for segment in result.get("speaker_segments") or []:
        speaker = segment.get("speaker") or "speaker_unknown"
        text = str(segment.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def sanitize_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "recording"


def summary_row_from_result(result: dict[str, Any], *, json_path: Path, processing_sec: float) -> dict[str, Any]:
    turns = list(result.get("speaker_turns") or [])
    return {
        "file": result.get("file") or "",
        "source_audio_path": result.get("source_audio_path") or "",
        "duration_sec": result.get("duration_sec") or "",
        "language": result.get("language") or "",
        "language_source": result.get("language_source") or "",
        "lid_language": result.get("lid_language") or "",
        "lid_language_source": result.get("lid_language_source") or "",
        "diarization_turns": len(turns),
        "diarization_error": result.get("diarization_error") or "",
        "processing_sec": round(float(processing_sec), 3),
        "text": str(result.get("text") or "").strip(),
        "json_path": str(json_path),
        "status": "ok",
        "error": "",
    }


def error_row(*, job: AudioJob, error: str) -> dict[str, Any]:
    return {
        "file": job.audio_path.name,
        "source_audio_path": str(job.audio_path),
        "duration_sec": "",
        "language": "",
        "language_source": "",
        "lid_language": "",
        "lid_language_source": "",
        "diarization_turns": "",
        "diarization_error": "",
        "processing_sec": "",
        "text": "",
        "json_path": "",
        "status": "error",
        "error": error,
    }


def item_from_existing_result(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "result": result,
        "row": summary_row_from_result(result, json_path=result_path, processing_sec=0.0),
        "timing_ms": {
            "load_audio": 0,
            "lid_probe": 0,
            "final_transcribe": 0,
            "timestamp_pass": 0,
            "diarization": 0,
            "file_total": 0,
        },
        "diarization_ms": 0,
        "file_total_ms": 0,
        "audio_name": str(result.get("file") or result_path.stem),
        "resumed": True,
    }


def process_audio_file(
    *,
    index: int,
    total_files: int,
    audio_path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    normalized_dir: Path,
    transcript_dir: Path,
) -> dict[str, Any]:
    file_started = time.perf_counter()
    stem = sanitize_stem(audio_path)
    logging.info("Processing %s/%s %s", index, total_files, audio_path.name)

    t0 = time.perf_counter()
    audio, sample_rate = load_normalized_audio(audio_path, args.sample_rate)
    load_audio_ms = int((time.perf_counter() - t0) * 1000)
    duration_sec = round(float(audio.shape[0]) / float(sample_rate), 4)
    pcm16le = float_to_pcm16(audio)
    normalized_path = normalized_dir / f"{stem}_mono{sample_rate}.wav"
    sf.write(str(normalized_path), audio, sample_rate, subtype="PCM_16")

    session_id = f"batch-{stem}"
    lid_pcm = audio[: int(min(audio.shape[0], sample_rate * min(args.chunk_sec, 20.0)))]
    t0 = time.perf_counter()
    try:
        lid_payload = post_worker(
            args.worker_url,
            float_to_pcm16(lid_pcm),
            sample_rate=sample_rate,
            decoder=args.decoder,
            language="auto",
            session_id=session_id,
            utterance_id=f"{stem}-lid",
            denoise=True,
            context_biasing=False,
            timeout_sec=args.timeout_sec,
        )
    except Exception as exc:
        logging.warning("LID probe failed for %s: %s", audio_path.name, exc)
        lid_payload = {
            "text": "",
            "language": args.fallback_language,
            "language_source": f"lid_probe_failed:{exc}",
        }
    lid_probe_ms = int((time.perf_counter() - t0) * 1000)

    final_language = safe_language(lid_payload, args.fallback_language)
    t0 = time.perf_counter()
    final_payload, final_chunks, _ = transcribe_chunks(
        worker_url=args.worker_url,
        pcm16le=pcm16le,
        sample_rate=sample_rate,
        decoder=args.decoder,
        language=final_language,
        session_id=session_id,
        utterance_prefix=f"{stem}-final",
        denoise=True,
        context_biasing=True,
        timeout_sec=args.timeout_sec,
        timestamp_type="none",
        chunk_sec=args.chunk_sec,
    )
    final_transcribe_ms = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    timestamp_payload, timestamp_chunks, words = transcribe_chunks(
        worker_url=args.worker_url,
        pcm16le=pcm16le,
        sample_rate=sample_rate,
        decoder=args.decoder,
        language=final_language,
        session_id=session_id,
        utterance_prefix=f"{stem}-timestamps",
        denoise=True,
        context_biasing=False,
        timestamp_type="word",
        timeout_sec=args.timeout_sec,
        chunk_sec=args.chunk_sec,
    )
    timestamp_pass_ms = int((time.perf_counter() - t0) * 1000)

    turns: list[SpeakerTurn] = []
    diarization_error = None
    rttm_path = None
    diarization_ms = 0
    if not args.skip_diarization:
        t0 = time.perf_counter()
        turns, rttm_path, diarization_error = run_diarization(
            normalized_path,
            waveform=audio,
            sample_rate=sample_rate,
            output_dir=output_dir,
            max_speakers=args.max_speakers,
            fixed_speakers=args.fixed_speakers,
            device=args.diarization_device,
        )
        diarization_ms = int((time.perf_counter() - t0) * 1000)

    file_total_ms = int((time.perf_counter() - file_started) * 1000)
    timing_ms = {
        "load_audio": load_audio_ms,
        "lid_probe": lid_probe_ms,
        "final_transcribe": final_transcribe_ms,
        "timestamp_pass": timestamp_pass_ms,
        "diarization": diarization_ms,
        "file_total": file_total_ms,
    }

    segments = speaker_segments(words, turns, str(final_payload.get("text") or ""))
    result = {
        "file": audio_path.name,
        "source_audio_path": str(audio_path),
        "normalized_audio_path": str(normalized_path),
        "duration_sec": duration_sec,
        "text": str(final_payload.get("text") or "").strip(),
        "language": final_payload.get("language"),
        "language_source": final_payload.get("language_source"),
        "lid_language": lid_payload.get("language"),
        "lid_language_source": lid_payload.get("language_source"),
        "context_biasing": final_payload.get("context_biasing"),
        "metrics": final_payload.get("metrics"),
        "final_chunk_count": len(final_chunks),
        "timestamp_chunk_count": len(timestamp_chunks),
        "timestamp_text": str(timestamp_payload.get("text") or "").strip(),
        "diarization_provider": "nemo_clustering" if turns else None,
        "diarization_error": diarization_error,
        "rttm_path": str(rttm_path) if rttm_path else None,
        "speaker_turns": [asdict(turn) for turn in turns],
        "speaker_segments": segments,
        "word_timestamps": words,
        "timing_ms": timing_ms,
    }
    result_path = transcript_dir / f"{stem}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_txt(transcript_dir / f"{stem}.txt", result)

    row = summary_row_from_result(result, json_path=result_path, processing_sec=file_total_ms / 1000.0)

    return {
        "result": result,
        "row": row,
        "timing_ms": timing_ms,
        "diarization_ms": diarization_ms,
        "file_total_ms": file_total_ms,
        "audio_name": audio_path.name,
    }


def main() -> int:
    run_started = time.perf_counter()
    args = parse_args()
    configure_logging(args.log_level)
    output_dir = args.output_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else None
    input_csv = args.input_csv.expanduser().resolve() if args.input_csv else None
    if (input_dir is None) == (input_csv is None):
        raise SystemExit("Provide exactly one of --input-dir or --input-csv")

    normalized_dir = output_dir / "normalized_audio"
    transcript_dir = output_dir / "transcripts"
    download_dir = (args.download_dir.expanduser().resolve() if args.download_dir else output_dir / "downloaded_audio")
    for directory in (output_dir, normalized_dir, transcript_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, str]] = []
    if input_csv is not None:
        source_rows = window_rows(csv_rows(input_csv), start_row=args.start_row, limit=args.limit)
        jobs = jobs_from_csv(source_rows, url_column=args.url_column, download_dir=download_dir)
        input_label = str(input_csv)
    else:
        assert input_dir is not None
        jobs = window_jobs(jobs_from_dir(input_dir), start_row=args.start_row, limit=args.limit)
        input_label = str(input_dir)
    if not jobs:
        raise SystemExit(f"No audio files found in {input_label}")

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    progress = tqdm(total=len(jobs), desc="Transcribing", unit="file") if bool(args.progress) and tqdm is not None else None

    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    rows_by_key: dict[str, dict[str, Any]] = {}
    total_stage_ms = {
        "load_audio": 0,
        "lid_probe": 0,
        "final_transcribe": 0,
        "timestamp_pass": 0,
        "diarization": 0,
        "file_total": 0,
    }
    def consume_item(item: dict[str, Any]) -> None:
        if item.get("result") is not None:
            results.append(item["result"])
        rows.append(item["row"])
        rows_by_key[item["job_key"]] = item["row"]
        if "timing_ms" in item:
            for key in total_stage_ms:
                total_stage_ms[key] += int(item["timing_ms"][key])
        if progress is not None:
            progress.update(1)
            resumed = " resumed" if item.get("resumed") else ""
            progress.set_postfix_str(
                f"last={item['audio_name']}{resumed} total={item.get('file_total_ms', 0)/1000.0:.1f}s diar={item.get('diarization_ms', 0)/1000.0:.1f}s"
            )

    def process_job(index: int, job: AudioJob) -> dict[str, Any]:
        try:
            download_audio(job, timeout_sec=args.timeout_sec, retries=args.download_retries)
            stem = sanitize_stem(job.audio_path)
            result_path = transcript_dir / f"{stem}.json"
            if args.resume and result_path.exists():
                item = item_from_existing_result(result_path)
            else:
                item = process_audio_file(
                    index=index,
                    total_files=len(jobs),
                    audio_path=job.audio_path,
                    args=args,
                    output_dir=output_dir,
                    normalized_dir=normalized_dir,
                    transcript_dir=transcript_dir,
                )
            item["job_key"] = job.key
            return item
        except Exception as exc:
            logging.exception("Failed processing %s", job.url or job.audio_path)
            return {
                "result": None,
                "row": error_row(job=job, error=str(exc)),
                "job_key": job.key,
                "audio_name": job.audio_path.name,
                "file_total_ms": 0,
                "diarization_ms": 0,
            }

    if args.concurrency == 1:
        for index, job in enumerate(jobs, start=1):
            consume_item(process_job(index, job))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_map = {
                executor.submit(process_job, index, job): job
                for index, job in enumerate(jobs, start=1)
            }
            for future in as_completed(future_map):
                consume_item(future.result())

    if progress is not None:
        progress.close()

    results.sort(key=lambda item: str(item.get("file") or ""))
    rows.sort(key=lambda item: str(item.get("file") or ""))

    (output_dir / "transcripts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    clean_csv_path = (args.clean_csv.expanduser().resolve() if args.clean_csv else output_dir / "clean_transcripts.csv")
    if input_csv is not None:
        clean_csv_path.parent.mkdir(parents=True, exist_ok=True)
        clean_fields = [args.id_column, args.url_column, "file", "duration_sec", "language", "transcript", "status", "error"]
        with clean_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=clean_fields)
            writer.writeheader()
            for row in source_rows:
                url = str(row.get(args.url_column) or "").strip()
                item = rows_by_key.get(url, {})
                writer.writerow(
                    {
                        args.id_column: row.get(args.id_column) or "",
                        args.url_column: url,
                        "file": item.get("file") or "",
                        "duration_sec": item.get("duration_sec") or "",
                        "language": item.get("language") or "",
                        "transcript": item.get("text") or "",
                        "status": item.get("status") or "error",
                        "error": item.get("error") or "",
                    }
                )
    summary = {
        "input": input_label,
        "output_dir": str(output_dir),
        "worker_url": args.worker_url,
        "concurrency": args.concurrency,
        "jobs": len(jobs),
        "input_rows": len(source_rows) if input_csv is not None else len(jobs),
        "files": len(results),
        "errors": sum(1 for row in rows if row.get("status") == "error"),
        "generated_at_epoch": int(time.time()),
        "runtime_sec": round(time.perf_counter() - run_started, 3),
        "stage_totals_sec": {
            key: round(value / 1000.0, 3) for key, value in total_stage_ms.items()
        },
        "stage_avg_sec_per_file": {
            key: round((value / 1000.0) / max(1, len(results)), 3)
            for key, value in total_stage_ms.items()
        },
        "outputs": {
            "jsonl": str(output_dir / "transcripts.jsonl"),
            "csv": str(output_dir / "summary.csv"),
            "clean_csv": str(clean_csv_path) if input_csv is not None else None,
            "transcripts_dir": str(transcript_dir),
            "normalized_audio_dir": str(normalized_dir),
            "download_dir": str(download_dir) if input_csv is not None else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
