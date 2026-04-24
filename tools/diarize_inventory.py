#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
    import pandas as pd

    from tools.diarization.audio import prepare_row_audio_for_diarization
    from tools.diarization.config import (
        DEFAULT_MULTISCALE_HOP_SEC,
        DEFAULT_MULTISCALE_WINDOW_SEC,
        DiarizationConfig,
    )
    from tools.diarization.io import turn_to_json_row, write_jsonl, write_rttm, write_summary
    from tools.diarization.merge import annotate_interval_with_speaker, speakerize_words
    from tools.diarization.providers import NeMoTelephonyDiarizationProvider
    from tools.progress import ProgressReporter
    from tools.segment_inventory_with_silero import READY_SEGMENTATION_STATUSES, clean_optional_str, load_inventory, read_audio
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python dependency "
        f"`{exc.name}`. Activate the target environment and install diarization dependencies with:\n"
        "  python -m pip install -r worker/requirements-diarization.txt"
    ) from exc

try:
    from eval_indicvoices_wer import resample_linear, to_mono
except ImportError:  # pragma: no cover
    from tools.eval_indicvoices_wer import resample_linear, to_mono


TimestampClient = Callable[[Path, str, str], dict[str, Any]]


def parse_float_list(value: str) -> tuple[float, ...]:
    items = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of floats")
    try:
        return tuple(float(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list `{value}`") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline NeMo telephony diarization on an inventory and optionally speakerize segmented ASR audio."
    )
    parser.add_argument("--inventory", type=Path, required=True, help="Input inventory CSV or Parquet.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diarization outputs.")
    parser.add_argument(
        "--audio-mode",
        choices=("auto", "mono", "borrower", "agent", "channel0", "channel1"),
        default="auto",
        help="How to choose source audio per inventory row. Default: auto",
    )
    parser.add_argument("--segments-jsonl", type=Path, help="Optional segments.jsonl from segment_inventory_with_silero.")
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    parser.add_argument(
        "--respect-ready-status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only process rows ready for downstream audio work. Default: enabled",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite normalized audio and output artifacts.")
    parser.add_argument("--progress-every", type=int, default=25, help="Log progress every N rows.")
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a tqdm-style progress bar when running interactively. Default: enabled",
    )
    parser.add_argument("--min-segment-sec", type=float, default=0.5, help="Minimum speech segment duration. Default: 0.5")
    parser.add_argument("--max-segment-sec", type=float, default=20.0, help="Maximum speech segment duration. Default: 20.0")
    parser.add_argument(
        "--multiscale-window-sec",
        type=parse_float_list,
        default=DEFAULT_MULTISCALE_WINDOW_SEC,
        help="Comma-separated multiscale window lengths. Default: 1.5,1.25,1.0,0.75,0.5",
    )
    parser.add_argument(
        "--multiscale-hop-sec",
        type=parse_float_list,
        default=DEFAULT_MULTISCALE_HOP_SEC,
        help="Comma-separated multiscale hop lengths. Default: 0.75,0.625,0.5,0.375,0.25",
    )
    parser.add_argument(
        "--speaker-count-mode",
        choices=("estimate", "fixed"),
        default="estimate",
        help="Whether to estimate or fix the speaker count. Default: estimate",
    )
    parser.add_argument("--fixed-speakers", type=int, help="Exact speaker count when --speaker-count-mode=fixed.")
    parser.add_argument("--min-speakers", type=int, default=1, help="Minimum speaker count hint. Default: 1")
    parser.add_argument("--max-speakers", type=int, default=2, help="Maximum speaker count hint. Default: 2")
    parser.add_argument(
        "--clustering-threshold",
        type=float,
        default=0.25,
        help="Clustering threshold for NeMo max_rp_threshold. Default: 0.25",
    )
    parser.add_argument(
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable overlap-aware MSDD settings. Default: disabled",
    )
    parser.add_argument("--vad-onset", type=float, default=0.1, help="NeMo VAD onset threshold. Default: 0.1")
    parser.add_argument("--vad-offset", type=float, default=0.1, help="NeMo VAD offset threshold. Default: 0.1")
    parser.add_argument("--vad-pad-onset", type=float, default=0.1, help="NeMo VAD onset padding in seconds. Default: 0.1")
    parser.add_argument("--vad-pad-offset", type=float, default=0.0, help="NeMo VAD offset padding in seconds. Default: 0.0")
    parser.add_argument("--vad-min-duration-on", type=float, default=0.0, help="NeMo VAD min speech duration. Default: 0.0")
    parser.add_argument("--vad-min-duration-off", type=float, default=0.2, help="NeMo VAD min silence duration. Default: 0.2")
    parser.add_argument("--vad-window-sec", type=float, default=0.15, help="NeMo VAD window length. Default: 0.15")
    parser.add_argument("--vad-shift-sec", type=float, default=0.01, help="NeMo VAD shift length. Default: 0.01")
    parser.add_argument("--vad-overlap", type=float, default=0.5, help="NeMo VAD overlap ratio. Default: 0.5")
    parser.add_argument("--nemo-vad-model", default="vad_multilingual_marblenet")
    parser.add_argument("--nemo-speaker-model", default="titanet_large")
    parser.add_argument("--nemo-msdd-model", default="diar_msdd_telephonic")
    parser.add_argument(
        "--timestamp-source",
        choices=("worker-http", "none"),
        default="worker-http",
        help="How to fetch optional word timestamps for speakerized transcripts. Default: worker-http",
    )
    parser.add_argument(
        "--worker-url",
        default="http://localhost:9000/v1/transcribe",
        help="Worker /v1/transcribe endpoint used when --timestamp-source=worker-http",
    )
    parser.add_argument("--timeout-sec", type=float, default=20.0, help="Worker timestamp request timeout. Default: 20")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity. Default: INFO",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_config(args: argparse.Namespace) -> DiarizationConfig:
    fixed_speakers = int(args.fixed_speakers) if args.fixed_speakers is not None else None
    return DiarizationConfig(
        min_segment_sec=float(args.min_segment_sec),
        max_segment_sec=float(args.max_segment_sec),
        multiscale_window_sec=tuple(args.multiscale_window_sec),
        multiscale_hop_sec=tuple(args.multiscale_hop_sec),
        speaker_count_mode=args.speaker_count_mode,
        fixed_speakers=fixed_speakers,
        min_speakers=int(args.min_speakers),
        max_speakers=int(args.max_speakers),
        clustering_threshold=float(args.clustering_threshold),
        overlap=bool(args.overlap),
        vad_window_sec=float(args.vad_window_sec),
        vad_shift_sec=float(args.vad_shift_sec),
        vad_onset=float(args.vad_onset),
        vad_offset=float(args.vad_offset),
        vad_pad_onset=float(args.vad_pad_onset),
        vad_pad_offset=float(args.vad_pad_offset),
        vad_min_duration_on=float(args.vad_min_duration_on),
        vad_min_duration_off=float(args.vad_min_duration_off),
        vad_overlap=float(args.vad_overlap),
        nemo_vad_model=str(args.nemo_vad_model),
        nemo_speaker_model=str(args.nemo_speaker_model),
        nemo_msdd_model=str(args.nemo_msdd_model),
    )


def ensure_inventory_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column_name in (
        "diarization_status",
        "diarization_provider",
        "diarization_turns_path",
        "diarization_rttm_path",
        "speakerized_transcript_path",
    ):
        if column_name not in frame.columns:
            frame[column_name] = pd.Series(pd.NA, index=frame.index, dtype="object")
    return frame


def load_segments_jsonl(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"segments JSONL not found: {resolved}")
    rows_by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid segments JSONL at line {line_number}: {exc}") from exc
            call_id = str(row.get("call_id") or "").strip()
            if not call_id:
                continue
            rows_by_call[call_id].append(dict(row))
    for call_id in rows_by_call:
        rows_by_call[call_id].sort(key=lambda item: (float(item.get("segment_start_sec", 0) or 0), str(item.get("segment_id") or "")))
    return dict(rows_by_call)


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return np.clip(np.rint(clipped * 32767.0), -32768, 32767).astype("<i2").tobytes()


def read_pcm16le_for_worker(audio_path: Path) -> tuple[bytes, int]:
    audio, sample_rate = read_audio(audio_path)
    mono_audio = to_mono(np.asarray(audio, dtype=np.float32))
    if sample_rate not in {8000, 16000}:
        mono_audio = resample_linear(mono_audio, int(sample_rate), 16000)
        sample_rate = 16000
    return float32_to_pcm16(mono_audio), int(sample_rate)


def request_transcription_with_timestamps(
    *,
    worker_url: str,
    audio_path: Path,
    language: str,
    utterance_id: str,
    timeout_sec: float,
) -> dict[str, Any]:
    pcm16le, sample_rate = read_pcm16le_for_worker(audio_path)
    request = urllib.request.Request(
        worker_url,
        data=pcm16le,
        method="POST",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(sample_rate),
            "X-Decoder": "ctc",
            "X-Language": language,
            "X-Mode": "final",
            "X-Session-Id": f"offline-diar-{utterance_id}",
            "X-Utterance-Id": utterance_id,
            "X-Timestamp-Type": "word",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(body) if body else {}
        if not isinstance(payload, dict):
            raise RuntimeError("worker returned a non-object JSON payload")
        return payload


def build_timestamp_client(args: argparse.Namespace) -> TimestampClient | None:
    if args.timestamp_source == "none":
        return None

    def fetch(audio_path: Path, language: str, segment_id: str) -> dict[str, Any]:
        return request_transcription_with_timestamps(
            worker_url=args.worker_url,
            audio_path=audio_path,
            language=language,
            utterance_id=segment_id,
            timeout_sec=float(args.timeout_sec),
        )

    return fetch


def build_speakerized_rows(
    *,
    call_id: str,
    segment_rows: list[dict[str, Any]],
    diarization_turns,
    timestamp_client: TimestampClient | None,
) -> tuple[list[dict[str, Any]], int]:
    speakerized_rows: list[dict[str, Any]] = []
    timestamp_errors = 0
    diarization_turns_list = list(diarization_turns)
    for row in segment_rows:
        start_sec = float(row.get("segment_start_sec", 0) or 0)
        end_sec = float(row.get("segment_end_sec", 0) or 0)
        segment_id = str(row.get("segment_id") or "")
        segment_annotation = annotate_interval_with_speaker(
            diarization_turns_list,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        text = str(row.get("text") or "")
        words: list[dict[str, Any]] = []
        timestamp_source = "none"
        if timestamp_client is not None:
            try:
                payload = timestamp_client(
                    Path(str(row.get("audio_filepath"))).expanduser().resolve(),
                    str(row.get("lang") or "hi"),
                    segment_id or f"{call_id}-{len(speakerized_rows):04d}",
                )
                text = str(payload.get("text") or text).strip()
                words = speakerize_words(
                    list(payload.get("word_timestamps") or []),
                    diarization_turns_list,
                    segment_offset_sec=start_sec,
                )
                timestamp_source = "worker_http"
            except (urllib.error.HTTPError, urllib.error.URLError, OSError, RuntimeError, json.JSONDecodeError) as exc:
                timestamp_errors += 1
                logging.warning("Timestamp fetch failed for segment_id=%s: %s", segment_id or "-", exc)

        speakerized_rows.append(
            {
                "segment_id": segment_id,
                "call_id": call_id,
                "row_id": str(row.get("row_id") or ""),
                "audio_filepath": str(row.get("audio_filepath") or ""),
                "source_audio_filepath": str(row.get("source_audio_filepath") or ""),
                "lang": str(row.get("lang") or ""),
                "text": text,
                "segment_start_sec": round(start_sec, 4),
                "segment_end_sec": round(end_sec, 4),
                "speaker_label": segment_annotation["speaker_label"],
                "speaker_confidence": segment_annotation["speaker_confidence"],
                "overlap_flag": segment_annotation["overlap_flag"],
                "timestamp_source": timestamp_source,
                "words": words,
            }
        )
    return speakerized_rows, timestamp_errors


def process_inventory(
    args: argparse.Namespace,
    *,
    provider=None,
    timestamp_client: TimestampClient | None = None,
) -> dict[str, Any]:
    inventory_path = args.inventory.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    normalized_audio_dir = output_dir / "normalized_audio"
    provider_runs_dir = output_dir / "_provider_runs"
    pred_rttm_dir = output_dir / "pred_rttms"
    turns_jsonl_path = output_dir / "diarization_turns.jsonl"
    speakerized_transcript_path = output_dir / "speakerized_transcript.jsonl"
    inventory_csv_path = output_dir / "data_inventory_diarized.csv"
    inventory_parquet_path = output_dir / "data_inventory_diarized.parquet"
    summary_path = output_dir / "summary.json"

    diar_config = build_config(args)
    active_provider = provider or NeMoTelephonyDiarizationProvider()
    active_timestamp_client = timestamp_client if timestamp_client is not None else build_timestamp_client(args)
    inventory_df = ensure_inventory_columns(load_inventory(inventory_path).copy())
    if args.limit is not None:
        inventory_df = inventory_df.head(max(0, int(args.limit))).copy()

    segments_by_call = load_segments_jsonl(args.segments_jsonl)
    total_rows = len(inventory_df)
    progress_every = max(1, int(args.progress_every))
    status_counts: Counter[str] = Counter()
    all_turn_rows: list[dict[str, Any]] = []
    all_speakerized_rows: list[dict[str, Any]] = []
    timestamp_error_count = 0

    logging.info(
        "Starting diarization rows_total=%s audio_mode=%s inventory=%s",
        total_rows,
        args.audio_mode,
        inventory_path,
    )
    progress = ProgressReporter(
        total=total_rows,
        description="Diarizing",
        enabled=bool(getattr(args, "progress_bar", True)),
        log_every=progress_every,
    )
    try:
        for row_number, (idx, row) in enumerate(inventory_df.iterrows(), start=1):
            current_status = clean_optional_str(row.get("segmentation_status")) or "pending"
            if args.respect_ready_status and current_status not in READY_SEGMENTATION_STATUSES:
                inventory_df.at[idx, "diarization_status"] = "skipped_not_ready"
                status_counts["skipped_not_ready"] += 1
                progress.advance(
                    status="skipped_not_ready",
                    details={"turns": len(all_turn_rows)},
                )
                continue

            call_id = clean_optional_str(row.get("call_id")) or f"row-{row_number:06d}"
            prepared_audio, error_status = prepare_row_audio_for_diarization(
                row,
                audio_mode=args.audio_mode,
                output_dir=normalized_audio_dir / call_id,
                target_sample_rate=diar_config.sample_rate,
                overwrite=bool(args.overwrite),
            )
            if prepared_audio is None:
                blocked_status = error_status or "blocked_missing_diarization_audio"
                inventory_df.at[idx, "diarization_status"] = blocked_status
                status_counts[blocked_status] += 1
                progress.advance(
                    status=blocked_status,
                    details={"turns": len(all_turn_rows)},
                )
                continue

            try:
                result = active_provider.diarize(
                    prepared_audio,
                    diar_config,
                    working_dir=provider_runs_dir / call_id,
                )
            except Exception as exc:  # pragma: no cover
                inventory_df.at[idx, "diarization_status"] = "diarization_failed"
                status_counts["diarization_failed"] += 1
                logging.exception("Diarization failed for call_id=%s: %s", call_id, exc)
                progress.advance(
                    status="diarization_failed",
                    details={"turns": len(all_turn_rows)},
                )
                continue

            if not result.turns:
                inventory_df.at[idx, "diarization_status"] = "diarization_no_speech"
                inventory_df.at[idx, "diarization_provider"] = result.provider
                status_counts["diarization_no_speech"] += 1
                progress.advance(
                    status="diarization_no_speech",
                    details={"turns": len(all_turn_rows)},
                )
                continue

            rttm_path = pred_rttm_dir / f"{call_id}.rttm"
            write_rttm(rttm_path, file_id=call_id, turns=result.turns)
            for turn in result.turns:
                all_turn_rows.append(
                    turn_to_json_row(
                        call_id=call_id,
                        turn=turn,
                        provider=result.provider,
                        source_audio_filepath=str(prepared_audio.source_audio_path),
                    )
                )

            if call_id in segments_by_call:
                speakerized_rows, row_timestamp_errors = build_speakerized_rows(
                    call_id=call_id,
                    segment_rows=segments_by_call[call_id],
                    diarization_turns=result.turns,
                    timestamp_client=active_timestamp_client,
                )
                all_speakerized_rows.extend(speakerized_rows)
                timestamp_error_count += row_timestamp_errors
                inventory_df.at[idx, "speakerized_transcript_path"] = str(speakerized_transcript_path)

            inventory_df.at[idx, "diarization_status"] = "diarized_nemo_telephony"
            inventory_df.at[idx, "diarization_provider"] = result.provider
            inventory_df.at[idx, "diarization_turns_path"] = str(turns_jsonl_path)
            inventory_df.at[idx, "diarization_rttm_path"] = str(rttm_path)
            status_counts["diarized_nemo_telephony"] += 1
            progress.advance(
                status="diarized_nemo_telephony",
                details={
                    "turns": len(all_turn_rows),
                    "speakerized": len(all_speakerized_rows),
                },
            )
    finally:
        progress.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(turns_jsonl_path, all_turn_rows)
    if all_speakerized_rows:
        write_jsonl(speakerized_transcript_path, all_speakerized_rows)
    inventory_df.to_csv(inventory_csv_path, index=False)
    try:
        inventory_df.to_parquet(inventory_parquet_path, index=False, engine="pyarrow")
        parquet_status = "written"
    except Exception:
        parquet_status = "failed"

    summary = {
        "inventory": str(inventory_path),
        "output_dir": str(output_dir),
        "audio_mode": args.audio_mode,
        "rows_seen": int(len(inventory_df)),
        "turns_written": int(len(all_turn_rows)),
        "speakerized_segments_written": int(len(all_speakerized_rows)),
        "timestamp_error_count": int(timestamp_error_count),
        "inventory_csv_path": str(inventory_csv_path),
        "inventory_parquet_path": str(inventory_parquet_path),
        "inventory_parquet_status": parquet_status,
        "turns_jsonl_path": str(turns_jsonl_path),
        "speakerized_transcript_path": str(speakerized_transcript_path) if all_speakerized_rows else None,
        "pred_rttm_dir": str(pred_rttm_dir),
        "segments_jsonl_path": str(args.segments_jsonl.expanduser().resolve()) if args.segments_jsonl else None,
        "timestamp_source": args.timestamp_source,
        "status_counts": dict(sorted(status_counts.items())),
        "min_segment_sec": diar_config.min_segment_sec,
        "max_segment_sec": diar_config.max_segment_sec,
        "multiscale_window_sec": list(diar_config.multiscale_window_sec),
        "multiscale_hop_sec": list(diar_config.multiscale_hop_sec),
        "speaker_count_mode": diar_config.speaker_count_mode,
        "fixed_speakers": diar_config.fixed_speakers,
        "min_speakers": diar_config.min_speakers,
        "max_speakers": diar_config.max_speakers,
        "clustering_threshold": diar_config.clustering_threshold,
        "overlap": diar_config.overlap,
        "vad_onset": diar_config.vad_onset,
        "vad_offset": diar_config.vad_offset,
        "vad_pad_onset": diar_config.vad_pad_onset,
        "vad_pad_offset": diar_config.vad_pad_offset,
        "vad_min_duration_on": diar_config.vad_min_duration_on,
        "vad_min_duration_off": diar_config.vad_min_duration_off,
    }
    write_summary(summary_path, summary)
    return summary


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    summary = process_inventory(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
