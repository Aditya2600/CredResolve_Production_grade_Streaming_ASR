#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diarization.config import DEFAULT_MULTISCALE_HOP_SEC, DEFAULT_MULTISCALE_WINDOW_SEC
from tools.segment_inventory_with_silero import clean_optional_str, load_inventory, process_inventory as segment_inventory_process_inventory


DEFAULT_DIARIZATION_DIR = Path("outputs/results_all/diarization_count_scan")
DEFAULT_OUTPUT_DIR = Path("outputs/results_all/silero_segments_diarized")
DEFAULT_TEXT_COLUMN = "raw_transcript"
DEFAULT_LANGUAGE_COLUMN = "language"
DEFAULT_INVENTORY_CANDIDATES = (
    "data_inventory_diarized.csv",
    "data_inventory_diarized.parquet",
    "data_inventory_normalized.csv",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse an existing diarization run to build training-ready segmented audio "
            "and a NeMo-friendly manifest."
        )
    )
    parser.add_argument(
        "--diarization-dir",
        type=Path,
        default=DEFAULT_DIARIZATION_DIR,
        help=(
            "Directory containing diarization_turns.jsonl and optionally "
            "data_inventory_diarized.csv. Default: outputs/results_all/diarization_count_scan"
        ),
    )
    parser.add_argument(
        "--diarization-turns",
        type=Path,
        help="Optional explicit diarization_turns.jsonl path. Overrides --diarization-dir for turns lookup.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help=(
            "Optional inventory CSV/Parquet. Defaults to the diarized inventory in --diarization-dir, "
            "or falls back to the original inventory recorded in summary.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for segmented training audio and manifests. Default: outputs/results_all/silero_segments_diarized",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Transcript column used for segmentation. Default: raw_transcript",
    )
    parser.add_argument(
        "--language-column",
        default=DEFAULT_LANGUAGE_COLUMN,
        help="Language column copied into the manifest. Default: language",
    )
    parser.add_argument(
        "--audio-mode",
        choices=("auto", "mono", "borrower", "agent", "channel0", "channel1"),
        default="auto",
        help="Audio selection mode passed to the segmenter. Default: auto",
    )
    parser.add_argument("--language", help="Optional fixed language override, for example hi.")
    parser.add_argument("--limit", type=int, help="Optional limit after filtering.")
    parser.add_argument(
        "--respect-ready-status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only segment rows ready for downstream audio work. Default: enabled",
    )
    parser.add_argument(
        "--require-diarized-status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only rows whose diarization_status is diarized_nemo_telephony. Default: enabled",
    )
    parser.add_argument(
        "--require-turn-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only rows whose call_id appears in diarization_turns.jsonl. Default: enabled",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite already-exported segment WAV files if they exist.",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=8000,
        help="Output sample rate for exported segment WAVs. Default: 8000",
    )
    parser.add_argument(
        "--vad-resample-rate",
        type=int,
        default=16000,
        choices=(8000, 16000),
        help="Resample rate used before Silero VAD. Default: 16000",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Silero speech threshold. Default: 0.5")
    parser.add_argument(
        "--min-speech-duration-ms",
        type=int,
        default=250,
        help="Silero minimum speech duration. Default: 250",
    )
    parser.add_argument(
        "--max-speech-duration-s",
        type=float,
        default=20.0,
        help="Silero maximum speech chunk duration. Default: 20.0",
    )
    parser.add_argument(
        "--min-silence-duration-ms",
        type=int,
        default=150,
        help="Silero minimum separating silence duration. Default: 150",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=60,
        help="Silero speech padding on each side. Default: 60",
    )
    parser.add_argument(
        "--diarization-min-segment-sec",
        type=float,
        default=0.5,
        help="Minimum diarization-refined segment duration. Default: 0.5",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Progress logging cadence passed to the segmenter. Default: 25",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a tqdm-style progress bar during segmentation when running interactively. Default: enabled",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON file: {resolved}: {exc}") from exc


def resolve_turns_path(args: argparse.Namespace) -> Path:
    if args.diarization_turns is not None:
        turns_path = args.diarization_turns.expanduser().resolve()
    else:
        turns_path = args.diarization_dir.expanduser().resolve() / "diarization_turns.jsonl"
    if not turns_path.exists():
        raise SystemExit(f"diarization turns file not found: {turns_path}")
    return turns_path


def resolve_inventory_path(args: argparse.Namespace, diarization_summary: dict[str, Any]) -> Path:
    if args.inventory is not None:
        inventory_path = args.inventory.expanduser().resolve()
        if not inventory_path.exists():
            raise SystemExit(f"inventory file not found: {inventory_path}")
        return inventory_path

    diarization_dir = args.diarization_dir.expanduser().resolve()
    for name in DEFAULT_INVENTORY_CANDIDATES:
        candidate = diarization_dir / name
        if candidate.exists():
            return candidate

    for key in ("inventory_csv_path", "inventory"):
        candidate_text = str(diarization_summary.get(key) or "").strip()
        if candidate_text:
            candidate = Path(candidate_text).expanduser().resolve()
            if candidate.exists():
                return candidate

    raise SystemExit(
        "could not resolve an inventory file; pass --inventory explicitly or keep "
        "data_inventory_diarized.csv inside the diarization output directory"
    )


def read_turn_call_ids(turns_path: Path) -> set[str]:
    call_ids: set[str] = set()
    for line_number, raw in enumerate(turns_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid diarization turns JSONL at {turns_path}:{line_number}: {exc}") from exc
        call_id = clean_optional_str(row.get("call_id"))
        if call_id:
            call_ids.add(call_id)
    return call_ids


def filter_inventory_for_training(
    frame: pd.DataFrame,
    *,
    require_diarized_status: bool,
    require_turn_coverage: bool,
    call_ids_with_turns: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows_input = int(len(frame))
    stats = {
        "rows_input": rows_input,
        "rows_after_diarized_status_filter": rows_input,
        "rows_after_turn_coverage_filter": rows_input,
        "rows_excluded_by_diarized_status": 0,
        "rows_excluded_by_turn_coverage": 0,
        "diarized_status_filter": {
            "enabled": bool(require_diarized_status),
            "kept_status": "diarized_nemo_telephony",
            "rows_before": rows_input,
            "rows_after": rows_input,
            "rows_excluded": 0,
            "excluded_status_counts": {},
        },
        "turn_coverage_filter": {
            "enabled": bool(require_turn_coverage),
            "rows_before": rows_input,
            "rows_after": rows_input,
            "rows_excluded": 0,
            "excluded_missing_call_id": 0,
            "excluded_missing_turns_for_call_id": 0,
            "call_ids_with_turns_count": int(len(call_ids_with_turns)),
        },
    }
    filtered = frame.copy()

    if require_diarized_status:
        if "diarization_status" not in filtered.columns:
            raise SystemExit(
                "inventory does not include diarization_status; use data_inventory_diarized.csv "
                "or disable --require-diarized-status"
            )
        normalized_status = filtered["diarization_status"].map(clean_optional_str).fillna("")
        status_mask = normalized_status == "diarized_nemo_telephony"
        excluded_status_counts = {
            str(status or "<empty>"): int(count)
            for status, count in normalized_status.loc[~status_mask].value_counts(dropna=False).sort_index().items()
        }
        filtered = filtered.loc[status_mask].copy()
        stats["rows_after_diarized_status_filter"] = int(len(filtered))
        stats["rows_excluded_by_diarized_status"] = rows_input - int(len(filtered))
        stats["diarized_status_filter"] = {
            "enabled": True,
            "kept_status": "diarized_nemo_telephony",
            "rows_before": rows_input,
            "rows_after": int(len(filtered)),
            "rows_excluded": rows_input - int(len(filtered)),
            "excluded_status_counts": excluded_status_counts,
        }

    turn_rows_before = int(len(filtered))
    stats["turn_coverage_filter"]["rows_before"] = turn_rows_before

    if require_turn_coverage:
        if "call_id" not in filtered.columns:
            raise SystemExit(
                "inventory does not include call_id; disable --require-turn-coverage "
                "or provide a compatible inventory"
            )
        normalized_call_ids = filtered["call_id"].map(clean_optional_str).fillna("")
        turn_mask = normalized_call_ids.isin(call_ids_with_turns)
        excluded_missing_call_id = int((~turn_mask & normalized_call_ids.eq("")).sum())
        excluded_missing_turns = int((~turn_mask & normalized_call_ids.ne("")).sum())
        filtered = filtered.loc[turn_mask].copy()
        stats["rows_after_turn_coverage_filter"] = int(len(filtered))
        stats["rows_excluded_by_turn_coverage"] = turn_rows_before - int(len(filtered))
        stats["turn_coverage_filter"] = {
            "enabled": True,
            "rows_before": turn_rows_before,
            "rows_after": int(len(filtered)),
            "rows_excluded": turn_rows_before - int(len(filtered)),
            "excluded_missing_call_id": excluded_missing_call_id,
            "excluded_missing_turns_for_call_id": excluded_missing_turns,
            "call_ids_with_turns_count": int(len(call_ids_with_turns)),
        }

    return filtered, stats


def build_segmenter_args(
    *,
    filtered_inventory_path: Path,
    output_dir: Path,
    diarization_turns_path: Path,
    wrapper_args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        inventory=filtered_inventory_path,
        output_dir=output_dir,
        audio_mode=wrapper_args.audio_mode,
        text_column=wrapper_args.text_column,
        language_column=wrapper_args.language_column,
        language=wrapper_args.language,
        limit=None,
        respect_ready_status=bool(wrapper_args.respect_ready_status),
        overwrite=bool(wrapper_args.overwrite),
        output_sample_rate=int(wrapper_args.output_sample_rate),
        vad_resample_rate=int(wrapper_args.vad_resample_rate),
        threshold=float(wrapper_args.threshold),
        min_speech_duration_ms=int(wrapper_args.min_speech_duration_ms),
        max_speech_duration_s=float(wrapper_args.max_speech_duration_s),
        min_silence_duration_ms=int(wrapper_args.min_silence_duration_ms),
        speech_pad_ms=int(wrapper_args.speech_pad_ms),
        diarization_turns=diarization_turns_path,
        enable_diarization=False,
        diarization_audio_mode="auto",
        diarization_min_segment_sec=float(wrapper_args.diarization_min_segment_sec),
        diarization_max_segment_sec=20.0,
        diarization_multiscale_window_sec=DEFAULT_MULTISCALE_WINDOW_SEC,
        diarization_multiscale_hop_sec=DEFAULT_MULTISCALE_HOP_SEC,
        diarization_speaker_count_mode="estimate",
        diarization_fixed_speakers=None,
        diarization_min_speakers=1,
        diarization_max_speakers=2,
        diarization_clustering_threshold=0.25,
        diarization_overlap=False,
        diarization_vad_onset=0.1,
        diarization_vad_offset=0.1,
        diarization_vad_pad_onset=0.1,
        diarization_vad_pad_offset=0.0,
        diarization_vad_min_duration_on=0.0,
        diarization_vad_min_duration_off=0.2,
        diarization_vad_window_sec=0.15,
        diarization_vad_shift_sec=0.01,
        diarization_vad_overlap=0.5,
        diarization_vad_model="vad_multilingual_marblenet",
        diarization_speaker_model="titanet_large",
        diarization_msdd_model="diar_msdd_telephonic",
        progress_every=int(wrapper_args.progress_every),
        progress_bar=bool(getattr(wrapper_args, "progress_bar", True)),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    diarization_dir = args.diarization_dir.expanduser().resolve()
    diarization_summary = load_json(diarization_dir / "summary.json")
    diarization_turns_path = resolve_turns_path(args)
    inventory_path = resolve_inventory_path(args, diarization_summary)

    inventory_df = load_inventory(inventory_path)
    call_ids_with_turns = read_turn_call_ids(diarization_turns_path) if args.require_turn_coverage else set()
    filtered_inventory, filter_stats = filter_inventory_for_training(
        inventory_df,
        require_diarized_status=bool(args.require_diarized_status),
        require_turn_coverage=bool(args.require_turn_coverage),
        call_ids_with_turns=call_ids_with_turns,
    )

    if args.limit is not None:
        filtered_inventory = filtered_inventory.head(max(0, int(args.limit))).copy()

    if filtered_inventory.empty:
        raise SystemExit("no rows remain after filtering; nothing to segment")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_inventory_path = output_dir / "input_inventory_diarized_for_training.csv"
    filtered_inventory.to_csv(filtered_inventory_path, index=False)

    segmenter_args = build_segmenter_args(
        filtered_inventory_path=filtered_inventory_path,
        output_dir=output_dir,
        diarization_turns_path=diarization_turns_path,
        wrapper_args=args,
    )
    segmentation_summary = segment_inventory_process_inventory(segmenter_args)

    orchestration_summary = {
        "diarization_dir": str(diarization_dir),
        "source_inventory": str(inventory_path),
        "filtered_inventory_path": str(filtered_inventory_path),
        "diarization_turns_path": str(diarization_turns_path),
        "require_diarized_status": bool(args.require_diarized_status),
        "require_turn_coverage": bool(args.require_turn_coverage),
        **filter_stats,
        "rows_after_limit": int(len(filtered_inventory)),
        "segmentation": segmentation_summary,
    }
    orchestration_summary_path = output_dir / "orchestration_summary.json"
    orchestration_summary_path.write_text(
        json.dumps(orchestration_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(orchestration_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
