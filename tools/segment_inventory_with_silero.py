#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diarization.config import (
    DEFAULT_MULTISCALE_HOP_SEC,
    DEFAULT_MULTISCALE_WINDOW_SEC,
    DiarizationConfig,
)
from tools.diarization.io import turn_to_json_row, write_rttm
from tools.diarization.merge import annotate_interval_with_speaker, annotate_segment_rows, load_diarization_turns
from tools.diarization.providers import NeMoTelephonyDiarizationProvider
from tools.diarization.providers.base import SpeakerTurn
from tools.progress import ProgressReporter

try:
    from normalize_indic_transcripts import extract_transcript_text, infer_language_code_from_text
    from eval_indicvoices_wer import resample_linear, to_mono
except ImportError:  # pragma: no cover - allows package-style imports
    from tools.normalize_indic_transcripts import extract_transcript_text, infer_language_code_from_text
    from tools.eval_indicvoices_wer import resample_linear, to_mono


TEXT_COLUMN_CANDIDATES = (
    "asr_l1_text",
    "normalized_transcript",
    "native_language_transcript",
    "raw_transcript",
)
LANGUAGE_COLUMN_CANDIDATES = ("lang", "language", "language_code", "locale")
TEXT_KEY_CANDIDATES = ("asr_l1_text", "en_text", "text", "transcript", "reference", "utterance")
ROLE_KEY_CANDIDATES = ("role", "speaker", "party")
READY_SEGMENTATION_STATUSES = frozenset(
    {
        "ready_mono_segmentation",
        "ready_split_channels",
        "ready_source_audio",
    }
)
BORROWER_ROLE_ALIASES = frozenset({"borrower", "customer", "user", "callee", "debtor", "listener"})
AGENT_ROLE_ALIASES = frozenset({"agent", "caller", "collector", "executive", "advisor", "representative"})
LANGUAGE_ID_MAP = {
    "assamese": "as",
    "as": "as",
    "bengali": "bn",
    "bn": "bn",
    "bodo": "brx",
    "brx": "brx",
    "dogri": "doi",
    "doi": "doi",
    "english": "en",
    "en": "en",
    "gujarati": "gu",
    "gu": "gu",
    "hindi": "hi",
    "hi": "hi",
    "kannada": "kn",
    "kn": "kn",
    "kashmiri": "ks",
    "ks": "ks",
    "konkani": "gom",
    "gom": "gom",
    "maithili": "mai",
    "mai": "mai",
    "malayalam": "ml",
    "ml": "ml",
    "manipuri": "mni",
    "mni": "mni",
    "marathi": "mr",
    "mr": "mr",
    "nepali": "ne",
    "ne": "ne",
    "odia": "or",
    "oriya": "or",
    "or": "or",
    "punjabi": "pa",
    "pa": "pa",
    "sanskrit": "sa",
    "sa": "sa",
    "santali": "sat",
    "sat": "sat",
    "sindhi": "sd",
    "sd": "sd",
    "tamil": "ta",
    "ta": "ta",
    "telugu": "te",
    "te": "te",
    "urdu": "ur",
    "ur": "ur",
}
MULTISPACE_RE = re.compile(r"\s+")
JSON_LIST_RE = re.compile(r"^\s*\[")
JSON_DICT_RE = re.compile(r"^\s*\{")


@dataclass(frozen=True)
class TranscriptUtterance:
    text: str
    role: str | None
    source_index: int


@dataclass(frozen=True)
class SpeechSegment:
    start_sample: int
    end_sample: int

    @property
    def duration_samples(self) -> int:
        return max(0, self.end_sample - self.start_sample)


@dataclass(frozen=True)
class AudioTarget:
    audio_path: Path
    target_kind: str
    role_filter: str | None
    channel_label: str | None


SpeechTimestampDetector = Callable[[np.ndarray, int], list[dict[str, int]]]


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
        description=(
            "Segment an ASR inventory into utterance-level training audio using Silero VAD, "
            "and emit a NeMo-style manifest plus an updated inventory copy."
        )
    )
    parser.add_argument("--inventory", type=Path, required=True, help="Input inventory CSV or Parquet.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for segmented audio and manifests.")
    parser.add_argument(
        "--audio-mode",
        choices=("auto", "mono", "borrower", "agent", "channel0", "channel1"),
        default="auto",
        help="How to choose source audio per inventory row. Default: auto",
    )
    parser.add_argument(
        "--text-column",
        help=(
            "Transcript column to segment against. Auto-detected from: "
            + ", ".join(TEXT_COLUMN_CANDIDATES)
        ),
    )
    parser.add_argument(
        "--language-column",
        help=(
            "Language column to copy into the manifest. Auto-detected from: "
            + ", ".join(LANGUAGE_COLUMN_CANDIDATES)
        ),
    )
    parser.add_argument("--language", help="Optional fixed language override, for example hi.")
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    parser.add_argument(
        "--respect-ready-status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only process rows whose segmentation_status is ready for segmentation. "
            "Default: enabled"
        ),
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
        help="Resample rate used before Silero VAD if source audio is not already 8 kHz or 16 kHz. Default: 16000",
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
        "--diarization-turns",
        type=Path,
        help="Optional diarization_turns.jsonl path used to attach speaker labels to exported segments.",
    )
    parser.add_argument(
        "--enable-diarization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run optional inline NeMo telephony diarization before final segmentation. "
            "Default: disabled"
        ),
    )
    parser.add_argument(
        "--diarization-audio-mode",
        choices=("auto", "mono", "borrower", "agent", "channel0", "channel1"),
        default="auto",
        help="Audio selection used by inline diarization. Default: auto",
    )
    parser.add_argument(
        "--diarization-min-segment-sec",
        type=float,
        default=0.5,
        help="Minimum diarization-refined segment duration in seconds. Default: 0.5",
    )
    parser.add_argument(
        "--diarization-max-segment-sec",
        type=float,
        default=20.0,
        help="Maximum diarization segment duration hint in seconds. Default: 20.0",
    )
    parser.add_argument(
        "--diarization-multiscale-window-sec",
        type=parse_float_list,
        default=DEFAULT_MULTISCALE_WINDOW_SEC,
        help="Comma-separated multiscale window lengths for diarization. Default: 1.5,1.25,1.0,0.75,0.5",
    )
    parser.add_argument(
        "--diarization-multiscale-hop-sec",
        type=parse_float_list,
        default=DEFAULT_MULTISCALE_HOP_SEC,
        help="Comma-separated multiscale hop lengths for diarization. Default: 0.75,0.625,0.5,0.375,0.25",
    )
    parser.add_argument(
        "--diarization-speaker-count-mode",
        choices=("estimate", "fixed"),
        default="estimate",
        help="Whether inline diarization estimates or fixes the speaker count. Default: estimate",
    )
    parser.add_argument(
        "--diarization-fixed-speakers",
        type=int,
        help="Exact speaker count for inline diarization when --diarization-speaker-count-mode=fixed.",
    )
    parser.add_argument(
        "--diarization-min-speakers",
        type=int,
        default=1,
        help="Minimum speaker count hint for inline diarization. Default: 1",
    )
    parser.add_argument(
        "--diarization-max-speakers",
        type=int,
        default=2,
        help="Maximum speaker count hint for inline diarization. Default: 2",
    )
    parser.add_argument(
        "--diarization-clustering-threshold",
        type=float,
        default=0.25,
        help="Clustering threshold for NeMo max_rp_threshold. Default: 0.25",
    )
    parser.add_argument(
        "--diarization-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable overlap-aware inline diarization. Default: disabled",
    )
    parser.add_argument(
        "--diarization-vad-onset",
        type=float,
        default=0.1,
        help="NeMo VAD onset threshold for inline diarization. Default: 0.1",
    )
    parser.add_argument(
        "--diarization-vad-offset",
        type=float,
        default=0.1,
        help="NeMo VAD offset threshold for inline diarization. Default: 0.1",
    )
    parser.add_argument(
        "--diarization-vad-pad-onset",
        type=float,
        default=0.1,
        help="NeMo VAD onset padding in seconds. Default: 0.1",
    )
    parser.add_argument(
        "--diarization-vad-pad-offset",
        type=float,
        default=0.0,
        help="NeMo VAD offset padding in seconds. Default: 0.0",
    )
    parser.add_argument(
        "--diarization-vad-min-duration-on",
        type=float,
        default=0.0,
        help="NeMo VAD minimum speech duration in seconds. Default: 0.0",
    )
    parser.add_argument(
        "--diarization-vad-min-duration-off",
        type=float,
        default=0.2,
        help="NeMo VAD minimum silence duration in seconds. Default: 0.2",
    )
    parser.add_argument(
        "--diarization-vad-window-sec",
        type=float,
        default=0.15,
        help="NeMo VAD window length in seconds. Default: 0.15",
    )
    parser.add_argument(
        "--diarization-vad-shift-sec",
        type=float,
        default=0.01,
        help="NeMo VAD shift length in seconds. Default: 0.01",
    )
    parser.add_argument(
        "--diarization-vad-overlap",
        type=float,
        default=0.5,
        help="NeMo VAD overlap ratio for inline diarization. Default: 0.5",
    )
    parser.add_argument("--diarization-vad-model", default="vad_multilingual_marblenet")
    parser.add_argument("--diarization-speaker-model", default="titanet_large")
    parser.add_argument("--diarization-msdd-model", default="diar_msdd_telephonic")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log progress every N processed rows. Default: 25",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a tqdm-style progress bar when running interactively. Default: enabled",
    )
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


def build_diarization_config(args: argparse.Namespace) -> DiarizationConfig:
    fixed_speakers = (
        int(args.diarization_fixed_speakers)
        if getattr(args, "diarization_fixed_speakers", None) is not None
        else None
    )
    return DiarizationConfig(
        min_segment_sec=float(args.diarization_min_segment_sec),
        max_segment_sec=float(args.diarization_max_segment_sec),
        multiscale_window_sec=tuple(args.diarization_multiscale_window_sec),
        multiscale_hop_sec=tuple(args.diarization_multiscale_hop_sec),
        speaker_count_mode=str(args.diarization_speaker_count_mode),
        fixed_speakers=fixed_speakers,
        min_speakers=int(args.diarization_min_speakers),
        max_speakers=int(args.diarization_max_speakers),
        clustering_threshold=float(args.diarization_clustering_threshold),
        overlap=bool(args.diarization_overlap),
        vad_window_sec=float(args.diarization_vad_window_sec),
        vad_shift_sec=float(args.diarization_vad_shift_sec),
        vad_onset=float(args.diarization_vad_onset),
        vad_offset=float(args.diarization_vad_offset),
        vad_pad_onset=float(args.diarization_vad_pad_onset),
        vad_pad_offset=float(args.diarization_vad_pad_offset),
        vad_min_duration_on=float(args.diarization_vad_min_duration_on),
        vad_min_duration_off=float(args.diarization_vad_min_duration_off),
        vad_overlap=float(args.diarization_vad_overlap),
        nemo_vad_model=str(args.diarization_vad_model),
        nemo_speaker_model=str(args.diarization_speaker_model),
        nemo_msdd_model=str(args.diarization_msdd_model),
    )


def is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except TypeError:
        return False
    if isinstance(result, bool):
        return result
    return False


def normalize_space(value: str) -> str:
    return MULTISPACE_RE.sub(" ", str(value or "")).strip()


def clean_optional_str(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = normalize_space(str(value))
    return text or None


def sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "item"


def ensure_diarization_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column_name in (
        "diarization_status",
        "diarization_provider",
        "diarization_turns_path",
        "diarization_rttm_path",
    ):
        if column_name not in frame.columns:
            frame[column_name] = pd.Series(pd.NA, index=frame.index, dtype="object")
    return frame


def load_inventory(path: Path) -> pd.DataFrame:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"inventory file not found: {resolved}")
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved, dtype=str, keep_default_na=True, encoding="utf-8-sig")


def detect_existing_column(frame: pd.DataFrame, explicit: str | None, candidates: tuple[str, ...]) -> str | None:
    if explicit:
        if explicit not in frame.columns:
            raise SystemExit(f"column not found in inventory: {explicit}")
        return explicit
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def try_parse_json(value: str) -> Any | None:
    text = value.strip()
    if not text:
        return None
    if not JSON_LIST_RE.match(text) and not JSON_DICT_RE.match(text):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_utterance_role(payload: dict[str, Any]) -> str | None:
    for key in ROLE_KEY_CANDIDATES:
        role = clean_optional_str(payload.get(key))
        if role:
            return role.casefold()
    return None


def extract_utterance_text(payload: dict[str, Any]) -> str:
    for key in TEXT_KEY_CANDIDATES:
        value = payload.get(key)
        if not is_missing(value):
            text = normalize_space(str(value))
            if text:
                return text
    return extract_transcript_text(payload)


def parse_transcript_utterances(value: Any) -> list[TranscriptUtterance]:
    if is_missing(value):
        return []

    parsed = value
    if isinstance(value, str):
        candidate = try_parse_json(value)
        if candidate is not None:
            parsed = candidate

    if isinstance(parsed, dict):
        interaction = parsed.get("interaction_transcript")
        if isinstance(interaction, list):
            utterances: list[TranscriptUtterance] = []
            for index, item in enumerate(interaction):
                if isinstance(item, dict):
                    text = extract_utterance_text(item)
                    if text:
                        utterances.append(
                            TranscriptUtterance(
                                text=text,
                                role=extract_utterance_role(item),
                                source_index=index,
                            )
                        )
                else:
                    text = extract_transcript_text(item)
                    if text:
                        utterances.append(TranscriptUtterance(text=text, role=None, source_index=index))
            if utterances:
                return utterances
        flattened = extract_transcript_text(parsed)
        return [TranscriptUtterance(text=flattened, role=None, source_index=0)] if flattened else []

    if isinstance(parsed, list):
        utterances = []
        for index, item in enumerate(parsed):
            if isinstance(item, dict):
                text = extract_utterance_text(item)
                if text:
                    utterances.append(
                        TranscriptUtterance(
                            text=text,
                            role=extract_utterance_role(item),
                            source_index=index,
                        )
                    )
            else:
                text = extract_transcript_text(item)
                if text:
                    utterances.append(TranscriptUtterance(text=text, role=None, source_index=index))
        return utterances

    flattened = extract_transcript_text(parsed)
    return [TranscriptUtterance(text=flattened, role=None, source_index=0)] if flattened else []


def normalize_channel_label(value: Any) -> str | None:
    text = clean_optional_str(value)
    if not text:
        return None
    aliases = {
        "0": "ch0",
        "1": "ch1",
        "ch0": "ch0",
        "ch1": "ch1",
        "channel0": "ch0",
        "channel1": "ch1",
        "channel_0": "ch0",
        "channel_1": "ch1",
    }
    return aliases.get(text.casefold())


def other_channel(channel_label: str | None) -> str | None:
    if channel_label == "ch0":
        return "ch1"
    if channel_label == "ch1":
        return "ch0"
    return None


def resolve_channel_path(row: pd.Series, channel_label: str) -> Path | None:
    column_name = "channel_0_path" if channel_label == "ch0" else "channel_1_path"
    path_text = clean_optional_str(row.get(column_name))
    if not path_text:
        return None
    return Path(path_text).expanduser().resolve()


def infer_role_filter(channel_label: str | None, borrower_channel: str | None) -> str | None:
    if channel_label is None or borrower_channel is None:
        return None
    if channel_label == borrower_channel:
        return "borrower"
    if channel_label == other_channel(borrower_channel):
        return "agent"
    return None


def resolve_audio_target(row: pd.Series, *, audio_mode: str) -> tuple[AudioTarget | None, str | None]:
    borrower_channel = normalize_channel_label(row.get("borrower_channel"))
    wav_audio_path = clean_optional_str(row.get("wav_audio_path"))
    local_audio_path = clean_optional_str(row.get("local_audio_path"))

    if audio_mode == "auto":
        recommended = clean_optional_str(row.get("recommended_next_step"))
        if recommended == "split_channels_first":
            if borrower_channel is None:
                return None, "blocked_missing_borrower_channel"
            borrower_path = resolve_channel_path(row, borrower_channel)
            if borrower_path is None:
                return None, "blocked_missing_channel_audio"
            return (
                AudioTarget(
                    audio_path=borrower_path,
                    target_kind="borrower_channel",
                    role_filter="borrower",
                    channel_label=borrower_channel,
                ),
                None,
            )
        if wav_audio_path:
            return (
                AudioTarget(
                    audio_path=Path(wav_audio_path).expanduser().resolve(),
                    target_kind="mono_wav",
                    role_filter=None,
                    channel_label=None,
                ),
                None,
            )
        if local_audio_path and local_audio_path.lower().endswith(".wav"):
            return (
                AudioTarget(
                    audio_path=Path(local_audio_path).expanduser().resolve(),
                    target_kind="source_wav",
                    role_filter=None,
                    channel_label=None,
                ),
                None,
            )
        return None, "blocked_missing_segment_audio"

    if audio_mode == "mono":
        candidate = wav_audio_path or (local_audio_path if local_audio_path and local_audio_path.lower().endswith(".wav") else None)
        if not candidate:
            return None, "blocked_missing_mono_audio"
        return (
            AudioTarget(
                audio_path=Path(candidate).expanduser().resolve(),
                target_kind="mono_wav",
                role_filter=None,
                channel_label=None,
            ),
            None,
        )

    if audio_mode == "borrower":
        if borrower_channel is None:
            return None, "blocked_missing_borrower_channel"
        channel_path = resolve_channel_path(row, borrower_channel)
        if channel_path is None:
            return None, "blocked_missing_channel_audio"
        return (
            AudioTarget(
                audio_path=channel_path,
                target_kind="borrower_channel",
                role_filter="borrower",
                channel_label=borrower_channel,
            ),
            None,
        )

    if audio_mode == "agent":
        agent_channel = other_channel(borrower_channel)
        if agent_channel is None:
            return None, "blocked_missing_borrower_channel"
        channel_path = resolve_channel_path(row, agent_channel)
        if channel_path is None:
            return None, "blocked_missing_channel_audio"
        return (
            AudioTarget(
                audio_path=channel_path,
                target_kind="agent_channel",
                role_filter="agent",
                channel_label=agent_channel,
            ),
            None,
        )

    explicit_channel = "ch0" if audio_mode == "channel0" else "ch1"
    channel_path = resolve_channel_path(row, explicit_channel)
    if channel_path is None:
        return None, "blocked_missing_channel_audio"
    return (
        AudioTarget(
            audio_path=channel_path,
            target_kind=f"explicit_{explicit_channel}",
            role_filter=infer_role_filter(explicit_channel, borrower_channel),
            channel_label=explicit_channel,
        ),
        None,
    )


def normalize_role(value: str | None) -> str | None:
    if not value:
        return None
    text = normalize_space(value).casefold()
    if text in BORROWER_ROLE_ALIASES:
        return "borrower"
    if text in AGENT_ROLE_ALIASES:
        return "agent"
    return text or None


def filter_utterances(utterances: list[TranscriptUtterance], *, role_filter: str | None) -> list[TranscriptUtterance]:
    if role_filter is None:
        return utterances
    filtered = [utterance for utterance in utterances if normalize_role(utterance.role) == role_filter]
    if filtered:
        return filtered
    has_any_known_role = any(normalize_role(utterance.role) in {"borrower", "agent"} for utterance in utterances)
    if has_any_known_role:
        return []
    return utterances


def normalize_language_code(value: Any) -> str | None:
    text = clean_optional_str(value)
    if not text:
        return None
    normalized = text.replace("-", " ").replace("_", " ").casefold()
    return LANGUAGE_ID_MAP.get(normalized)


def resolve_language_code(
    row: pd.Series,
    *,
    explicit_language: str | None,
    language_column: str | None,
    fallback_text: str,
) -> str | None:
    if explicit_language:
        explicit = normalize_language_code(explicit_language)
        if explicit:
            return explicit
    if language_column is not None:
        inferred = normalize_language_code(row.get(language_column))
        if inferred:
            return inferred
    for candidate in LANGUAGE_COLUMN_CANDIDATES:
        if candidate == language_column:
            continue
        if candidate in row.index:
            inferred = normalize_language_code(row.get(candidate))
            if inferred:
                return inferred
    if fallback_text:
        return infer_language_code_from_text(fallback_text)
    return None


def parse_slice_tags(value: Any) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [normalize_space(str(item)) for item in value if normalize_space(str(item))]
    text = str(value).strip()
    if not text:
        return []
    parsed = try_parse_json(text)
    if isinstance(parsed, list):
        return [normalize_space(str(item)) for item in parsed if normalize_space(str(item))]
    return [normalize_space(text)]


def add_slice_tags(value: Any, *new_tags: str) -> str:
    tags = parse_slice_tags(value)
    for tag in new_tags:
        cleaned = normalize_space(tag)
        if cleaned and cleaned not in tags:
            tags.append(cleaned)
    return json.dumps(tags, ensure_ascii=False)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def prepare_audio_for_vad(audio: np.ndarray, sample_rate: int, *, vad_resample_rate: int) -> tuple[np.ndarray, int]:
    mono_audio = to_mono(np.asarray(audio, dtype=np.float32))
    if sample_rate in {8000, 16000}:
        return mono_audio, sample_rate
    return resample_linear(mono_audio, sample_rate, vad_resample_rate), vad_resample_rate


def build_silero_detector(args: argparse.Namespace) -> SpeechTimestampDetector:
    logging.info("Loading Silero VAD runtime")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "torch is required to run the Silero segmentation pipeline. "
            "Install it in the active environment before running this script."
        ) from exc

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "silero-vad is required to run this script. Install it in the active environment first."
        ) from exc

    torch.set_num_threads(1)
    logging.info("Loading Silero VAD model weights")
    model = load_silero_vad()
    logging.info("Silero VAD model ready")

    def detect(audio: np.ndarray, sample_rate: int) -> list[dict[str, int]]:
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
        timestamps = get_speech_timestamps(
            waveform,
            model,
            threshold=float(args.threshold),
            sampling_rate=int(sample_rate),
            min_speech_duration_ms=int(args.min_speech_duration_ms),
            max_speech_duration_s=float(args.max_speech_duration_s),
            min_silence_duration_ms=int(args.min_silence_duration_ms),
            speech_pad_ms=int(args.speech_pad_ms),
            return_seconds=False,
        )
        if hasattr(model, "reset_states"):
            model.reset_states()
        return [dict(item) for item in timestamps]

    return detect


def parse_speech_segments(timestamps: list[dict[str, int]]) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    for item in timestamps:
        start = int(item.get("start", 0) or 0)
        end = int(item.get("end", 0) or 0)
        if end <= start:
            continue
        segments.append(SpeechSegment(start_sample=start, end_sample=end))
    return segments


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return max(0, int(round(float(seconds) * int(sample_rate))))


def describe_interval_speakers(
    turns: list[SpeakerTurn],
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    labels: set[str] = set()
    overlap_flag = False
    for turn in turns:
        overlap_start = max(float(start_sec), float(turn.start_sec))
        overlap_end = min(float(end_sec), float(turn.end_sec))
        if overlap_end <= overlap_start:
            continue
        labels.add(turn.speaker_label)
        if turn.overlap_flag:
            overlap_flag = True
    speaker_count = len(labels)
    return {
        "speaker_count": speaker_count,
        "mixed_speaker_flag": speaker_count > 1 or overlap_flag,
    }


def segment_merge_score(
    left: SpeechSegment,
    right: SpeechSegment,
    *,
    speaker_turns: list[SpeakerTurn] | None,
    sample_rate: int | None,
) -> tuple[int, int, int]:
    gap = max(0, right.start_sample - left.end_sample)
    if not speaker_turns or not sample_rate:
        return (0, 0, gap)
    left_annotation = annotate_interval_with_speaker(
        speaker_turns,
        start_sec=left.start_sample / sample_rate,
        end_sec=left.end_sample / sample_rate,
    )
    right_annotation = annotate_interval_with_speaker(
        speaker_turns,
        start_sec=right.start_sample / sample_rate,
        end_sec=right.end_sample / sample_rate,
    )
    left_label = str(left_annotation.get("speaker_label") or "")
    right_label = str(right_annotation.get("speaker_label") or "")
    speaker_change_penalty = 0 if not left_label or not right_label or left_label == right_label else 1
    overlap_penalty = 1 if left_annotation.get("overlap_flag") or right_annotation.get("overlap_flag") else 0
    return (speaker_change_penalty, overlap_penalty, gap)


def merge_segments_to_count(
    segments: list[SpeechSegment],
    target_count: int,
    *,
    speaker_turns: list[SpeakerTurn] | None = None,
    sample_rate: int | None = None,
) -> list[SpeechSegment]:
    if target_count <= 0:
        return []
    merged = list(segments)
    while len(merged) > target_count and len(merged) > 1:
        merge_index = min(
            range(len(merged) - 1),
            key=lambda index: segment_merge_score(
                merged[index],
                merged[index + 1],
                speaker_turns=speaker_turns,
                sample_rate=sample_rate,
            ),
        )
        merged_segment = SpeechSegment(
            start_sample=merged[merge_index].start_sample,
            end_sample=merged[merge_index + 1].end_sample,
        )
        merged = merged[:merge_index] + [merged_segment] + merged[merge_index + 2 :]
    return merged


def merge_short_refined_segments(
    segments: list[SpeechSegment],
    *,
    speaker_turns: list[SpeakerTurn],
    sample_rate: int,
    min_duration_samples: int,
) -> list[SpeechSegment]:
    if not segments:
        return []

    def dominant_label(segment: SpeechSegment) -> str:
        annotation = annotate_interval_with_speaker(
            speaker_turns,
            start_sec=segment.start_sample / sample_rate,
            end_sec=segment.end_sample / sample_rate,
        )
        return str(annotation.get("speaker_label") or "")

    working = list(segments)
    refined: list[SpeechSegment] = []
    index = 0
    while index < len(working):
        segment = working[index]
        if segment.duration_samples >= min_duration_samples or len(working) == 1:
            refined.append(segment)
            index += 1
            continue

        previous = refined[-1] if refined else None
        following = working[index + 1] if index + 1 < len(working) else None
        current_label = dominant_label(segment)

        if previous is None and following is None:
            refined.append(segment)
            break

        if previous is None and following is not None:
            following_label = dominant_label(following)
            if current_label and following_label and current_label != following_label:
                refined.append(segment)
            else:
                working[index + 1] = SpeechSegment(start_sample=segment.start_sample, end_sample=following.end_sample)
            index += 1
            continue

        if previous is not None and following is None:
            previous_label = dominant_label(previous)
            if current_label and previous_label and current_label != previous_label:
                refined.append(segment)
            else:
                refined[-1] = SpeechSegment(start_sample=previous.start_sample, end_sample=segment.end_sample)
            index += 1
            continue

        assert previous is not None and following is not None
        previous_label = dominant_label(previous)
        following_label = dominant_label(following)
        if (
            current_label
            and previous_label
            and following_label
            and current_label != previous_label
            and current_label != following_label
        ):
            refined.append(segment)
            index += 1
            continue
        previous_score = segment_merge_score(
            previous,
            segment,
            speaker_turns=speaker_turns,
            sample_rate=sample_rate,
        )
        following_score = segment_merge_score(
            segment,
            following,
            speaker_turns=speaker_turns,
            sample_rate=sample_rate,
        )
        if previous_score <= following_score:
            refined[-1] = SpeechSegment(start_sample=previous.start_sample, end_sample=segment.end_sample)
        else:
            working[index + 1] = SpeechSegment(start_sample=segment.start_sample, end_sample=following.end_sample)
        index += 1
    return refined


def refine_segments_with_speaker_turns(
    segments: list[SpeechSegment],
    *,
    speaker_turns: list[SpeakerTurn],
    sample_rate: int,
    min_segment_sec: float,
) -> list[SpeechSegment]:
    if not segments or not speaker_turns:
        return segments

    min_duration_samples = max(1, seconds_to_samples(float(min_segment_sec), sample_rate))
    refined_segments: list[SpeechSegment] = []

    for segment in segments:
        segment_start_sec = segment.start_sample / sample_rate
        segment_end_sec = segment.end_sample / sample_rate
        boundaries = {segment.start_sample, segment.end_sample}

        for turn in speaker_turns:
            overlap_start_sec = max(segment_start_sec, float(turn.start_sec))
            overlap_end_sec = min(segment_end_sec, float(turn.end_sec))
            if overlap_end_sec <= overlap_start_sec:
                continue
            boundaries.add(max(segment.start_sample, seconds_to_samples(overlap_start_sec, sample_rate)))
            boundaries.add(min(segment.end_sample, seconds_to_samples(overlap_end_sec, sample_rate)))

        ordered_boundaries = sorted(boundaries)
        if len(ordered_boundaries) <= 2:
            refined_segments.append(segment)
            continue

        split_segments = [
            SpeechSegment(start_sample=start_sample, end_sample=end_sample)
            for start_sample, end_sample in zip(ordered_boundaries, ordered_boundaries[1:])
            if end_sample > start_sample
        ]
        if not split_segments:
            refined_segments.append(segment)
            continue
        refined_segments.extend(
            merge_short_refined_segments(
                split_segments,
                speaker_turns=speaker_turns,
                sample_rate=sample_rate,
                min_duration_samples=min_duration_samples,
            )
        )

    return refined_segments


def partition_utterances(utterances: list[TranscriptUtterance], segments: list[SpeechSegment]) -> list[list[TranscriptUtterance]]:
    if not segments:
        return []
    if not utterances:
        return [[] for _ in segments]
    if len(segments) >= len(utterances):
        groups = [[utterance] for utterance in utterances]
        groups.extend([[] for _ in range(len(segments) - len(utterances))])
        return groups

    weights = [max(1, len(utterance.text.replace(" ", ""))) for utterance in utterances]
    total_weight = sum(weights)
    total_duration = sum(segment.duration_samples for segment in segments)
    if total_weight <= 0 or total_duration <= 0:
        return [utterances] + [[] for _ in range(len(segments) - 1)]

    groups: list[list[TranscriptUtterance]] = []
    start_index = 0
    used_weight = 0
    cumulative_duration = 0

    for segment_index, segment in enumerate(segments):
        remaining_segments = len(segments) - segment_index
        if segment_index == len(segments) - 1 or remaining_segments == 1:
            groups.append(utterances[start_index:])
            break

        cumulative_duration += segment.duration_samples
        max_end = len(utterances) - (remaining_segments - 1)
        target_weight = round(total_weight * (cumulative_duration / total_duration))
        segment_target_weight = max(1, round(total_weight * (segment.duration_samples / total_duration)))

        best_end = start_index + 1
        best_delta: tuple[int, int] | None = None
        running_weight = 0
        for end_index in range(start_index + 1, max_end + 1):
            running_weight += weights[end_index - 1]
            current_weight = used_weight + running_weight
            delta = abs(current_weight - target_weight)
            candidate = (delta, abs(running_weight - segment_target_weight))
            if best_delta is None or candidate < best_delta:
                best_delta = candidate
                best_end = end_index

        groups.append(utterances[start_index:best_end])
        used_weight += sum(weights[start_index:best_end])
        start_index = best_end

    while len(groups) < len(segments):
        groups.append([])
    return groups


def align_segments_to_utterances(
    segments: list[SpeechSegment],
    utterances: list[TranscriptUtterance],
    *,
    speaker_turns: list[SpeakerTurn] | None = None,
    sample_rate: int | None = None,
) -> list[tuple[SpeechSegment, list[TranscriptUtterance]]]:
    if not segments or not utterances:
        return []
    working_segments = list(segments)
    if len(working_segments) > len(utterances):
        working_segments = merge_segments_to_count(
            working_segments,
            len(utterances),
            speaker_turns=speaker_turns,
            sample_rate=sample_rate,
        )
    grouped_utterances = partition_utterances(utterances, working_segments)
    aligned: list[tuple[SpeechSegment, list[TranscriptUtterance]]] = []
    for segment, grouped in zip(working_segments, grouped_utterances):
        if not grouped:
            continue
        aligned.append((segment, grouped))
    return aligned


def build_segment_text(utterances: list[TranscriptUtterance]) -> str:
    return normalize_space(" ".join(utterance.text for utterance in utterances if utterance.text))


def build_segment_roles(utterances: list[TranscriptUtterance]) -> list[str]:
    roles: list[str] = []
    for utterance in utterances:
        role = normalize_role(utterance.role)
        if role and role not in roles:
            roles.append(role)
    return roles


def apply_diarization_metadata_to_segments(
    segment_rows: list[dict[str, Any]],
    *,
    speaker_turns: list[SpeakerTurn],
    diarization_source: str,
    diarization_provider: str,
    boundary_refined: bool,
) -> list[dict[str, Any]]:
    annotated_rows = annotate_segment_rows(
        segment_rows,
        {
            str(row.get("call_id") or "").strip(): speaker_turns
            for row in segment_rows
            if str(row.get("call_id") or "").strip()
        }
        if speaker_turns
        else None,
    )
    enriched_rows: list[dict[str, Any]] = []
    for row in annotated_rows:
        start_sec = float(row.get("segment_start_sec", 0) or 0)
        end_sec = float(row.get("segment_end_sec", 0) or 0)
        speaker_stats = describe_interval_speakers(
            speaker_turns,
            start_sec=start_sec,
            end_sec=end_sec,
        ) if speaker_turns else {"speaker_count": 0, "mixed_speaker_flag": False}
        enriched = dict(row)
        enriched.update(
            {
                "diarization_used": bool(speaker_turns),
                "diarization_source": diarization_source if speaker_turns else "",
                "diarization_provider": diarization_provider if speaker_turns else "",
                "speaker_alignment_method": "audio_interval_overlap" if speaker_turns else "",
                "diarization_boundary_refined": bool(boundary_refined and speaker_turns),
                "diarization_speaker_count": int(speaker_stats["speaker_count"]),
                "diarization_mixed_speaker_flag": bool(speaker_stats["mixed_speaker_flag"]),
            }
        )
        enriched_rows.append(enriched)
    return enriched_rows


def run_inline_diarization_for_row(
    *,
    row: pd.Series,
    audio_mode: str,
    diarization_config: DiarizationConfig,
    provider: NeMoTelephonyDiarizationProvider,
    overwrite: bool,
    normalized_audio_dir: Path,
    provider_runs_dir: Path,
    pred_rttm_dir: Path,
) -> tuple[list[SpeakerTurn], dict[str, Any]]:
    from tools.diarization.audio import prepare_row_audio_for_diarization

    fallback_row_id = clean_optional_str(row.get("row_id")) or "row"
    call_id = clean_optional_str(row.get("call_id")) or fallback_row_id
    safe_call_id = sanitize_filename_component(call_id)
    prepared_audio, error_status = prepare_row_audio_for_diarization(
        row,
        audio_mode=audio_mode,
        output_dir=normalized_audio_dir / safe_call_id,
        target_sample_rate=diarization_config.sample_rate,
        overwrite=overwrite,
    )
    if prepared_audio is None:
        return [], {
            "status": error_status or "blocked_missing_diarization_audio",
            "provider": "",
            "rttm_path": None,
            "turn_rows": [],
        }

    result = provider.diarize(
        prepared_audio,
        diarization_config,
        working_dir=provider_runs_dir / safe_call_id,
    )
    if not result.turns:
        return [], {
            "status": "diarization_no_speech",
            "provider": result.provider,
            "rttm_path": None,
            "turn_rows": [],
        }

    rttm_path = pred_rttm_dir / f"{safe_call_id}.rttm"
    write_rttm(rttm_path, file_id=safe_call_id, turns=result.turns)
    turn_rows = [
        turn_to_json_row(
            call_id=call_id,
            turn=turn,
            provider=result.provider,
            source_audio_filepath=str(prepared_audio.source_audio_path),
        )
        for turn in result.turns
    ]
    return list(result.turns), {
        "status": "diarized_nemo_telephony",
        "provider": result.provider,
        "rttm_path": str(rttm_path),
        "turn_rows": turn_rows,
    }


def segment_audio_for_row(
    *,
    row: pd.Series,
    audio_target: AudioTarget,
    utterances: list[TranscriptUtterance],
    language_code: str,
    detector: SpeechTimestampDetector,
    output_audio_dir: Path,
    output_sample_rate: int,
    vad_resample_rate: int,
    overwrite: bool,
    text_column: str,
    speaker_turns_by_call: dict[str, list] | None,
    diarization_refine_segments: bool,
    diarization_min_segment_sec: float,
    diarization_source: str,
    diarization_provider: str,
) -> tuple[list[dict[str, Any]], str]:
    audio, sample_rate = read_audio(audio_target.audio_path)
    vad_audio, vad_sample_rate = prepare_audio_for_vad(audio, sample_rate, vad_resample_rate=vad_resample_rate)
    segments = parse_speech_segments(detector(vad_audio, vad_sample_rate))
    if not segments:
        return [], "silero_no_speech"

    call_id = clean_optional_str(row.get("call_id")) or ""
    row_speaker_turns = list((speaker_turns_by_call or {}).get(call_id, []))
    segments_refined = False
    if row_speaker_turns and diarization_refine_segments:
        refined_segments = refine_segments_with_speaker_turns(
            segments,
            speaker_turns=row_speaker_turns,
            sample_rate=vad_sample_rate,
            min_segment_sec=diarization_min_segment_sec,
        )
        if refined_segments:
            segments_refined = len(refined_segments) != len(segments) or any(
                refined.start_sample != original.start_sample or refined.end_sample != original.end_sample
                for refined, original in zip(refined_segments[: len(segments)], segments[: len(refined_segments)])
            )
            segments = refined_segments

    aligned = align_segments_to_utterances(
        segments,
        utterances,
        speaker_turns=row_speaker_turns,
        sample_rate=vad_sample_rate,
    )
    if not aligned:
        return [], "blocked_missing_transcript_alignment"

    fallback_row_id = clean_optional_str(row.get("row_id")) or "row"
    safe_call_id = sanitize_filename_component(clean_optional_str(row.get("call_id")) or fallback_row_id)
    source_stub = sanitize_filename_component(audio_target.target_kind)
    segment_rows: list[dict[str, Any]] = []

    for segment_index, (segment, segment_utterances) in enumerate(aligned, start=1):
        segment_audio = vad_audio[segment.start_sample : segment.end_sample]
        if output_sample_rate != vad_sample_rate:
            segment_audio = resample_linear(segment_audio, vad_sample_rate, output_sample_rate)
        segment_duration = float(len(segment_audio) / output_sample_rate) if output_sample_rate > 0 else 0.0
        if segment_duration <= 0:
            continue

        segment_filename = f"{safe_call_id}_{source_stub}_seg{segment_index:04d}.wav"
        segment_path = output_audio_dir / segment_filename
        if overwrite or not segment_path.exists():
            segment_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(segment_path, segment_audio, output_sample_rate, subtype="PCM_16")

        segment_text = build_segment_text(segment_utterances)
        if not segment_text:
            continue
        roles = build_segment_roles(segment_utterances)
        segment_rows.append(
            {
                "audio_filepath": str(segment_path),
                "duration": round(segment_duration, 4),
                "text": segment_text,
                "lang": language_code,
                "segment_id": f"{safe_call_id}-{source_stub}-{segment_index:04d}",
                "row_id": clean_optional_str(row.get("row_id")) or "",
                "call_id": clean_optional_str(row.get("call_id")) or "",
                "segment_index": segment_index,
                "segment_start_sec": round(segment.start_sample / vad_sample_rate, 4),
                "segment_end_sec": round(segment.end_sample / vad_sample_rate, 4),
                "segment_audio_source": audio_target.target_kind,
                "segment_channel": audio_target.channel_label or "",
                "transcript_column": text_column,
                "transcript_alignment_method": "role_order_heuristic",
                "utterance_count": len(segment_utterances),
                "transcript_roles": roles,
                "source_audio_filepath": str(audio_target.audio_path),
            }
        )

    segment_rows = apply_diarization_metadata_to_segments(
        segment_rows,
        speaker_turns=row_speaker_turns,
        diarization_source=diarization_source,
        diarization_provider=diarization_provider,
        boundary_refined=segments_refined,
    )

    if not segment_rows:
        return [], "blocked_empty_segment_manifest"
    return segment_rows, "segmented_silero_vad"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def process_inventory(
    args: argparse.Namespace,
    *,
    detector: SpeechTimestampDetector | None = None,
    diarization_provider: NeMoTelephonyDiarizationProvider | None = None,
) -> dict[str, Any]:
    inventory_path = args.inventory.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_audio_dir = output_dir / "audio"
    output_inventory_csv = output_dir / "data_inventory_segmented.csv"
    output_inventory_parquet = output_dir / "data_inventory_segmented.parquet"
    output_manifest_path = output_dir / "manifest.jsonl"
    output_segments_jsonl = output_dir / "segments.jsonl"
    output_summary_path = output_dir / "summary.json"
    inline_diarization_turns_path = output_dir / "diarization_turns.jsonl"
    inline_pred_rttm_dir = output_dir / "pred_rttms"
    inline_diarization_audio_dir = output_dir / "_diarization_audio"
    inline_diarization_runs_dir = output_dir / "_diarization_runs"

    if getattr(args, "enable_diarization", False) and getattr(args, "diarization_turns", None):
        raise SystemExit("choose either --enable-diarization or --diarization-turns, not both")

    inventory_df = load_inventory(inventory_path).copy()
    if getattr(args, "enable_diarization", False) or getattr(args, "diarization_turns", None):
        inventory_df = ensure_diarization_columns(inventory_df)
    if args.limit is not None:
        inventory_df = inventory_df.head(max(0, int(args.limit))).copy()

    text_column = detect_existing_column(inventory_df, args.text_column, TEXT_COLUMN_CANDIDATES)
    if text_column is None:
        raise SystemExit(
            "could not detect a transcript column in the inventory; tried: "
            + ", ".join(TEXT_COLUMN_CANDIDATES)
        )
    language_column = detect_existing_column(inventory_df, args.language_column, LANGUAGE_COLUMN_CANDIDATES)
    active_detector = detector or build_silero_detector(args)
    progress_every = max(1, int(getattr(args, "progress_every", 25)))
    external_speaker_turns_by_call = (
        load_diarization_turns(args.diarization_turns.expanduser().resolve())
        if getattr(args, "diarization_turns", None)
        else None
    )
    active_diarization_provider = diarization_provider or NeMoTelephonyDiarizationProvider()
    diarization_config = build_diarization_config(args)

    segment_manifest_rows: list[dict[str, Any]] = []
    segment_metadata_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    diarization_status_counts: Counter[str] = Counter()
    inline_diarization_rows: list[dict[str, Any]] = []
    total_rows = len(inventory_df)
    ready_rows = int(
        inventory_df["segmentation_status"].map(clean_optional_str).isin(READY_SEGMENTATION_STATUSES).sum()
    ) if "segmentation_status" in inventory_df.columns else total_rows
    duration_seconds = (
        pd.to_numeric(inventory_df.get("duration_sec"), errors="coerce").fillna(0).sum()
        if "duration_sec" in inventory_df.columns
        else 0.0
    )
    logging.info(
        "Starting Silero segmentation rows_total=%s rows_ready=%s approx_audio_hours=%.2f audio_mode=%s text_column=%s language_column=%s",
        total_rows,
        ready_rows,
        float(duration_seconds) / 3600.0,
        args.audio_mode,
        text_column,
        language_column or "auto",
    )
    progress = ProgressReporter(
        total=total_rows,
        description="Segmenting",
        enabled=bool(getattr(args, "progress_bar", True)),
        log_every=progress_every,
    )
    try:
        for row_number, (idx, row) in enumerate(inventory_df.iterrows(), start=1):
            current_status = clean_optional_str(row.get("segmentation_status")) or "pending"
            if args.respect_ready_status and current_status not in READY_SEGMENTATION_STATUSES:
                status_counts["skipped_not_ready"] += 1
                progress.advance(
                    status="skipped_not_ready",
                    details={"segments": len(segment_manifest_rows)},
                )
                continue

            audio_target, error_status = resolve_audio_target(row, audio_mode=args.audio_mode)
            if audio_target is None:
                inventory_df.at[idx, "segmentation_status"] = error_status
                blocked_status = error_status or "blocked_missing_segment_audio"
                status_counts[blocked_status] += 1
                progress.advance(
                    status=blocked_status,
                    details={"segments": len(segment_manifest_rows)},
                )
                continue

            if not audio_target.audio_path.exists():
                inventory_df.at[idx, "segmentation_status"] = "blocked_missing_segment_audio"
                status_counts["blocked_missing_segment_audio"] += 1
                progress.advance(
                    status="blocked_missing_segment_audio",
                    details={"segments": len(segment_manifest_rows)},
                )
                continue

            utterances = parse_transcript_utterances(row.get(text_column))
            utterances = filter_utterances(utterances, role_filter=audio_target.role_filter)
            if not utterances:
                inventory_df.at[idx, "segmentation_status"] = "blocked_missing_transcript"
                status_counts["blocked_missing_transcript"] += 1
                progress.advance(
                    status="blocked_missing_transcript",
                    details={"segments": len(segment_manifest_rows)},
                )
                continue

            fallback_text = build_segment_text(utterances)
            language_code = resolve_language_code(
                row,
                explicit_language=args.language,
                language_column=language_column,
                fallback_text=fallback_text,
            )
            if not language_code:
                inventory_df.at[idx, "segmentation_status"] = "blocked_missing_language"
                status_counts["blocked_missing_language"] += 1
                progress.advance(
                    status="blocked_missing_language",
                    details={"segments": len(segment_manifest_rows)},
                )
                continue

            row_call_id = clean_optional_str(row.get("call_id")) or ""
            row_speaker_turns = list((external_speaker_turns_by_call or {}).get(row_call_id, []))
            row_diarization_source = ""
            row_diarization_provider = ""

            if getattr(args, "enable_diarization", False):
                try:
                    row_speaker_turns, diarization_result = run_inline_diarization_for_row(
                        row=row,
                        audio_mode=str(args.diarization_audio_mode),
                        diarization_config=diarization_config,
                        provider=active_diarization_provider,
                        overwrite=bool(args.overwrite),
                        normalized_audio_dir=inline_diarization_audio_dir,
                        provider_runs_dir=inline_diarization_runs_dir,
                        pred_rttm_dir=inline_pred_rttm_dir,
                    )
                except Exception as exc:  # pragma: no cover - defensive runtime path
                    logging.exception(
                        "Inline diarization failed row_id=%s call_id=%s: %s",
                        clean_optional_str(row.get("row_id")) or "-",
                        row_call_id or "-",
                        exc,
                    )
                    diarization_result = {
                        "status": "diarization_failed",
                        "provider": "",
                        "rttm_path": None,
                        "turn_rows": [],
                    }
                    row_speaker_turns = []

                inventory_df.at[idx, "diarization_status"] = diarization_result["status"]
                inventory_df.at[idx, "diarization_provider"] = diarization_result["provider"] or pd.NA
                inventory_df.at[idx, "diarization_turns_path"] = (
                    str(inline_diarization_turns_path)
                    if row_speaker_turns
                    else pd.NA
                )
                inventory_df.at[idx, "diarization_rttm_path"] = diarization_result["rttm_path"] or pd.NA
                diarization_status_counts[str(diarization_result["status"])] += 1
                inline_diarization_rows.extend(list(diarization_result["turn_rows"]))
                if row_speaker_turns:
                    row_diarization_source = "inline_nemo_telephony"
                    row_diarization_provider = str(diarization_result["provider"] or "")
            elif external_speaker_turns_by_call is not None:
                external_status = "loaded_external_turns" if row_speaker_turns else "diarization_turns_missing_for_call"
                inventory_df.at[idx, "diarization_status"] = external_status
                inventory_df.at[idx, "diarization_provider"] = "external_jsonl"
                inventory_df.at[idx, "diarization_turns_path"] = str(args.diarization_turns.expanduser().resolve())
                inventory_df.at[idx, "diarization_rttm_path"] = pd.NA
                diarization_status_counts[external_status] += 1
                if row_speaker_turns:
                    row_diarization_source = "external_turns_jsonl"
                    row_diarization_provider = "external_jsonl"

            try:
                row_segments, row_status = segment_audio_for_row(
                    row=row,
                    audio_target=audio_target,
                    utterances=utterances,
                    language_code=language_code,
                    detector=active_detector,
                    output_audio_dir=output_audio_dir,
                    output_sample_rate=int(args.output_sample_rate),
                    vad_resample_rate=int(args.vad_resample_rate),
                    overwrite=bool(args.overwrite),
                    text_column=text_column,
                    speaker_turns_by_call={row_call_id: row_speaker_turns} if row_call_id and row_speaker_turns else None,
                    diarization_refine_segments=bool(row_speaker_turns),
                    diarization_min_segment_sec=float(args.diarization_min_segment_sec),
                    diarization_source=row_diarization_source,
                    diarization_provider=row_diarization_provider,
                )
            except Exception as exc:  # pragma: no cover - defensive runtime path
                inventory_df.at[idx, "segmentation_status"] = "silero_segmentation_failed"
                status_counts["silero_segmentation_failed"] += 1
                segment_metadata_rows.append(
                    {
                        "row_id": clean_optional_str(row.get("row_id")) or "",
                        "call_id": clean_optional_str(row.get("call_id")) or "",
                        "status": "silero_segmentation_failed",
                        "error": str(exc),
                    }
                )
                progress.advance(
                    status="silero_segmentation_failed",
                    details={"segments": len(segment_manifest_rows)},
                )
                continue

            inventory_df.at[idx, "segmentation_status"] = row_status
            if row_status == "segmented_silero_vad":
                inventory_df.at[idx, "slice_tags"] = add_slice_tags(
                    row.get("slice_tags"),
                    "silero_vad",
                    audio_target.target_kind,
                    "diarization_aware" if row_speaker_turns else "",
                )
            status_counts[row_status] += 1

            segment_manifest_rows.extend(row_segments)
            for segment_row in row_segments:
                segment_metadata_rows.append(dict(segment_row))
            progress.advance(
                status=row_status,
                details={
                    "segments": len(segment_manifest_rows),
                    "last_segments": len(row_segments),
                },
            )
    finally:
        progress.close()

    if getattr(args, "enable_diarization", False):
        write_jsonl(inline_diarization_turns_path, inline_diarization_rows)
    write_jsonl(output_manifest_path, segment_manifest_rows)
    write_jsonl(output_segments_jsonl, segment_metadata_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(output_inventory_csv, index=False)
    try:
        inventory_df.to_parquet(output_inventory_parquet, index=False, engine="pyarrow")
        parquet_status = "written"
    except Exception:
        parquet_status = "failed"

    summary = {
        "inventory": str(inventory_path),
        "output_dir": str(output_dir),
        "text_column": text_column,
        "language_column": language_column,
        "audio_mode": args.audio_mode,
        "output_sample_rate": int(args.output_sample_rate),
        "vad_threshold": float(args.threshold),
        "min_speech_duration_ms": int(args.min_speech_duration_ms),
        "max_speech_duration_s": float(args.max_speech_duration_s),
        "min_silence_duration_ms": int(args.min_silence_duration_ms),
        "speech_pad_ms": int(args.speech_pad_ms),
        "rows_seen": int(len(inventory_df)),
        "segments_written": int(len(segment_manifest_rows)),
        "status_counts": dict(sorted(status_counts.items())),
        "enable_diarization": bool(getattr(args, "enable_diarization", False)),
        "diarization_audio_mode": str(getattr(args, "diarization_audio_mode", "")) if getattr(args, "enable_diarization", False) else None,
        "diarization_status_counts": dict(sorted(diarization_status_counts.items())),
        "diarization_turns_written": int(len(inline_diarization_rows)),
        "manifest_path": str(output_manifest_path),
        "segments_path": str(output_segments_jsonl),
        "inventory_csv_path": str(output_inventory_csv),
        "inventory_parquet_path": str(output_inventory_parquet),
        "inventory_parquet_status": parquet_status,
        "diarization_turns_path": (
            str(inline_diarization_turns_path)
            if getattr(args, "enable_diarization", False)
            else str(args.diarization_turns.expanduser().resolve()) if getattr(args, "diarization_turns", None) else None
        ),
        "pred_rttm_dir": str(inline_pred_rttm_dir) if getattr(args, "enable_diarization", False) else None,
        "diarization_min_segment_sec": float(args.diarization_min_segment_sec),
        "diarization_max_segment_sec": float(args.diarization_max_segment_sec),
        "diarization_multiscale_window_sec": list(args.diarization_multiscale_window_sec),
        "diarization_multiscale_hop_sec": list(args.diarization_multiscale_hop_sec),
        "diarization_speaker_count_mode": str(args.diarization_speaker_count_mode),
        "diarization_fixed_speakers": args.diarization_fixed_speakers,
        "diarization_min_speakers": int(args.diarization_min_speakers),
        "diarization_max_speakers": int(args.diarization_max_speakers),
        "diarization_clustering_threshold": float(args.diarization_clustering_threshold),
        "diarization_overlap": bool(args.diarization_overlap),
    }
    output_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    summary = process_inventory(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
