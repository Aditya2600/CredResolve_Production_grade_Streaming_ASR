#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import pandas as pd
import requests


NORMALIZED_COLUMNS = [
    "row_id",
    "call_id",
    "audio_url",
    "local_audio_path",
    "wav_audio_path",
    "channel_0_path",
    "channel_1_path",
    "borrower_channel",
    "channel_assignment_method",
    "channel_assignment_confidence",
    "channels",
    "sample_rate",
    "duration_sec",
    "audio_structure",
    "recommended_next_step",
    "lender",
    "portfolio",
    "borrower_name",
    "event_date",
    "amount_1",
    "amount_2",
    "amount_3",
    "transcript_source",
    "raw_transcript",
    "normalized_transcript",
    "download_status",
    "audio_convert_status",
    "segmentation_status",
    "alignment_status",
    "quality_tier",
    "split",
    "slice_tags",
]

SOURCE_DERIVED_COLUMNS = (
    "call_id",
    "audio_url",
    "lender",
    "portfolio",
    "borrower_name",
    "event_date",
    "amount_1",
    "amount_2",
    "amount_3",
)

REQUEST_USER_AGENT = "build_asr_inventory/1.0"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class ColumnProfile:
    original_name: str
    canonical_name: str
    tokens: frozenset[str]
    sample_values: tuple[str, ...]


@dataclass(frozen=True)
class CandidateMatch:
    source_column: str
    score: int


@dataclass(frozen=True)
class MappingResult:
    mapped_columns: dict[str, str]
    missing_columns: list[str]
    source_export_names: list[str]
    source_rename_map: dict[str, str]


@dataclass(frozen=True)
class AudioProbeResult:
    channels: int | None
    sample_rate: int | None
    duration_sec: float | None
    audio_structure: str
    recommended_next_step: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a normalized ASR inventory from a raw call-export CSV, download audio, "
            "inspect the downloaded recording with ffprobe, split stereo recordings into "
            "per-channel WAV files, convert audio to mono 8 kHz WAV, and export the "
            "inventory as CSV and Parquet."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw call-export CSV.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory used for downloaded source audio. Default: data/raw",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("data/wav"),
        help="Directory used for normalized WAV audio. Default: data/wav",
    )
    parser.add_argument(
        "--channel-dir",
        type=Path,
        default=Path("data/channels"),
        help="Directory used for split channel WAV audio. Default: data/channels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory used for CSV and Parquet exports. Default: outputs",
    )
    parser.add_argument(
        "--channel-map",
        type=Path,
        help=(
            "Optional CSV with at least call_id and borrower_channel columns. "
            "Extra columns are ignored."
        ),
    )
    parser.add_argument(
        "--default-borrower-channel",
        choices=("ch0", "ch1"),
        help=(
            "Optional fallback borrower channel applied to split stereo calls that are not "
            "listed in --channel-map."
        ),
    )
    parser.add_argument(
        "--audit-sample-size",
        type=int,
        default=None,
        help=(
            "Optional number of split stereo calls to include in the borrower channel audit "
            "CSV. Default: include all eligible calls."
        ),
    )
    parser.add_argument(
        "--audit-seed",
        type=int,
        default=42,
        help="Random seed used for borrower audit sampling. Default: 42",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=None,
        help=(
            "Optional borrower audit CSV path. Default: <output-dir>/borrower_channel_audit.csv"
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke tests or partial processing.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP read timeout in seconds for each audio download attempt. Default: 60",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download attempts per audio file. Default: 3",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff in seconds between failed download attempts. Default: 2.0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and re-convert audio even if output files already exist.",
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


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg"):
        return
    raise SystemExit("ffmpeg was not found on PATH. Please install ffmpeg before running this script.")


def ensure_ffprobe_available() -> None:
    if shutil.which("ffprobe"):
        return
    raise SystemExit("ffprobe was not found on PATH. Please install ffprobe before running this script.")


def read_source_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    try:
        source_df = pd.read_csv(path, dtype=str, keep_default_na=True, encoding="utf-8-sig")
    except Exception as exc:  # pragma: no cover - defensive exit path
        raise SystemExit(f"Failed to read CSV {path}: {exc}") from exc

    if source_df.empty:
        raise SystemExit(f"Input CSV is empty: {path}")

    return source_df


def clean_nullable_string(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "unknown_call_id"


def canonicalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def build_column_profiles(source_df: pd.DataFrame, *, sample_limit: int = 3) -> list[ColumnProfile]:
    profiles: list[ColumnProfile] = []
    for column in source_df.columns:
        series = source_df[column].dropna().astype(str)
        sample_values = tuple(value.strip() for value in series.head(sample_limit) if value.strip())
        canonical_name = canonicalize_column_name(column)
        profiles.append(
            ColumnProfile(
                original_name=column,
                canonical_name=canonical_name,
                tokens=frozenset(token for token in canonical_name.split("_") if token),
                sample_values=sample_values,
            )
        )
    return profiles


def looks_like_datetime(profile: ColumnProfile) -> bool:
    if not profile.sample_values:
        return False
    for value in profile.sample_values:
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            continue
        if not pd.isna(parsed):
            return True
    return False


def numericish_ratio(profile: ColumnProfile) -> float:
    if not profile.sample_values:
        return 0.0

    numeric_matches = 0
    for value in profile.sample_values:
        cleaned = value.replace(",", "").replace(" ", "")
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", cleaned):
            numeric_matches += 1
    return numeric_matches / len(profile.sample_values)


def score_call_id(profile: ColumnProfile) -> int:
    exact_alias_scores = {
        "call_sid": 220,
        "call_id": 210,
        "session_id": 180,
        "session_sid": 175,
        "audio_id": 165,
        "recording_id": 160,
        "id": 60,
    }
    score = exact_alias_scores.get(profile.canonical_name, 0)
    id_tokens = {"id", "sid"}

    if "call" in profile.tokens and profile.tokens.intersection(id_tokens):
        score += 120
    if "session" in profile.tokens and profile.tokens.intersection(id_tokens):
        score += 95
    if ("audio" in profile.tokens or "recording" in profile.tokens) and profile.tokens.intersection(id_tokens):
        score += 80
    if any(re.fullmatch(r"\d{6,}", value.replace(",", "")) for value in profile.sample_values):
        score += 10

    if {"client", "campaign", "task", "lead", "caller", "phone", "number"}.intersection(profile.tokens):
        score -= 90
    if {"duration", "time", "date", "amount"}.intersection(profile.tokens):
        score -= 40

    return score


def score_audio_url(profile: ColumnProfile) -> int:
    exact_alias_scores = {
        "cr_recording_url": 240,
        "recording_url": 220,
        "audio_url": 210,
        "recording_file_url": 190,
        "file_url": 140,
    }
    score = exact_alias_scores.get(profile.canonical_name, 0)

    if "url" in profile.tokens:
        score += 70
    if "recording" in profile.tokens:
        score += 75
    if "audio" in profile.tokens:
        score += 70
    if any(value.startswith(("http://", "https://")) for value in profile.sample_values):
        score += 60
    if any((".mp3" in value.lower()) or (".wav" in value.lower()) for value in profile.sample_values):
        score += 25

    if {"duration", "durtion", "talk", "time"}.intersection(profile.tokens):
        score -= 100

    return score


def score_lender(profile: ColumnProfile) -> int:
    exact_alias_scores = {
        "lender": 220,
        "client": 170,
        "client_name": 165,
        "client_code": 150,
        "client_sid": 105,
        "workspace_code": 95,
    }
    score = exact_alias_scores.get(profile.canonical_name, 0)

    if "lender" in profile.tokens:
        score += 110
    if "client" in profile.tokens:
        score += 80
    if "workspace" in profile.tokens:
        score += 30

    if {"campaign", "portfolio", "dialer", "group", "description"}.intersection(profile.tokens):
        score -= 70
    if {"id", "sid"}.intersection(profile.tokens):
        score -= 35

    return score


def score_portfolio(profile: ColumnProfile) -> int:
    exact_alias_scores = {
        "portfolio": 220,
        "campaign": 210,
        "campaign_name": 205,
        "campaign_sid": 150,
        "dialer": 195,
        "group_description": 205,
        "institute_name": 185,
    }
    score = exact_alias_scores.get(profile.canonical_name, 0)

    if "portfolio" in profile.tokens:
        score += 110
    if "campaign" in profile.tokens:
        score += 95
    if "dialer" in profile.tokens:
        score += 95
    if "group" in profile.tokens:
        score += 70
    if "description" in profile.tokens:
        score += 40
    if "institute" in profile.tokens:
        score += 55

    if "lender" in profile.tokens:
        score -= 40

    return score


def score_borrower_name(profile: ColumnProfile) -> int:
    exact_alias_scores = {
        "borrower_name": 240,
        "customer_name": 235,
        "borrower": 180,
        "customer": 175,
        "name": 160,
        "lead_name": 150,
    }
    score = exact_alias_scores.get(profile.canonical_name, 0)

    if "borrower" in profile.tokens:
        score += 110
    if "customer" in profile.tokens:
        score += 110
    if "name" in profile.tokens:
        score += 85
    if any(re.search(r"[A-Za-z]", value) and " " in value for value in profile.sample_values):
        score += 15

    if {"client", "lender", "caller", "phone", "number", "agent", "campaign", "group"}.intersection(profile.tokens):
        score -= 100

    return score


def score_event_date(profile: ColumnProfile) -> int:
    exact_alias_scores = {
        "time_of_call": 240,
        "call_start_time": 230,
        "call_start": 225,
        "call_time": 220,
        "event_date": 210,
        "start_time": 205,
        "created_at": 175,
        "dms_time_stamp": 165,
        "call_date": 160,
        "date": 100,
        "end_time_of_call": 150,
        "due_date": 90,
    }
    score = exact_alias_scores.get(profile.canonical_name, 0)

    if "call" in profile.tokens and "time" in profile.tokens:
        score += 130
    if "call" in profile.tokens and "start" in profile.tokens:
        score += 120
    if "event" in profile.tokens and "date" in profile.tokens:
        score += 100
    if "start" in profile.tokens and "time" in profile.tokens:
        score += 90
    if "created" in profile.tokens:
        score += 50
    if "timestamp" in profile.tokens:
        score += 40
    if looks_like_datetime(profile):
        score += 25

    if {"duration", "talk", "amount", "emi"}.intersection(profile.tokens):
        score -= 110

    return score


def score_amount(profile: ColumnProfile, target_column: str) -> int:
    exact_alias_scores = {
        "amount_1": {
            "principle": 250,
            "principal": 250,
            "loan_amount": 200,
            "amount_due": 170,
            "due_amount": 165,
            "amount": 120,
        },
        "amount_2": {
            "emi_amount": 250,
            "emi": 240,
            "installment_amount": 225,
            "instalment_amount": 225,
            "emi_due": 220,
        },
        "amount_3": {
            "total_due": 250,
            "total_amount": 225,
            "balance_due": 220,
            "outstanding_amount": 220,
            "net_due": 215,
            "outstanding": 200,
        },
    }
    score = exact_alias_scores[target_column].get(profile.canonical_name, 0)

    if "amount" in profile.tokens:
        score += 60
    if "principal" in profile.tokens or "principle" in profile.tokens:
        score += 95 if target_column == "amount_1" else 35
    if "emi" in profile.tokens:
        score += 95 if target_column == "amount_2" else 25
    if "total" in profile.tokens:
        score += 90 if target_column == "amount_3" else 30
    if "due" in profile.tokens:
        score += 70 if target_column in {"amount_1", "amount_3"} else 30
    if "outstanding" in profile.tokens or "balance" in profile.tokens or "payable" in profile.tokens:
        score += 80 if target_column == "amount_3" else 35
    if "loan" in profile.tokens:
        score += 45 if target_column == "amount_1" else 15
    if "installment" in profile.tokens or "instalment" in profile.tokens:
        score += 90 if target_column == "amount_2" else 20
    if "bucket" in profile.tokens:
        score += 35

    if numericish_ratio(profile) >= 0.66:
        score += 25

    if {"id", "sid", "date", "time", "duration", "phone", "number"}.intersection(profile.tokens):
        score -= 140

    return score


def pick_best_match(
    profiles: list[ColumnProfile],
    *,
    used_columns: set[str],
    scorer: Callable[[ColumnProfile], int],
    minimum_score: int,
) -> CandidateMatch | None:
    best_match: CandidateMatch | None = None
    for profile in profiles:
        if profile.original_name in used_columns:
            continue
        score = scorer(profile)
        if score < minimum_score:
            continue
        if best_match is None or score > best_match.score:
            best_match = CandidateMatch(source_column=profile.original_name, score=score)
    return best_match


def infer_schema_mapping(source_df: pd.DataFrame) -> MappingResult:
    profiles = build_column_profiles(source_df)
    used_columns: set[str] = set()
    mapped_columns: dict[str, str] = {}
    missing_columns: list[str] = []

    field_specs: list[tuple[str, Callable[[ColumnProfile], int], int]] = [
        ("call_id", score_call_id, 90),
        ("audio_url", score_audio_url, 120),
        ("lender", score_lender, 70),
        ("portfolio", score_portfolio, 75),
        ("borrower_name", score_borrower_name, 90),
        ("event_date", score_event_date, 100),
    ]

    for normalized_column, scorer, minimum_score in field_specs:
        best_match = pick_best_match(
            profiles,
            used_columns=used_columns,
            scorer=scorer,
            minimum_score=minimum_score,
        )
        if best_match is None:
            missing_columns.append(normalized_column)
            continue
        mapped_columns[normalized_column] = best_match.source_column
        used_columns.add(best_match.source_column)

    for normalized_column in ("amount_1", "amount_2", "amount_3"):
        best_match = pick_best_match(
            profiles,
            used_columns=used_columns,
            scorer=lambda profile, target=normalized_column: score_amount(profile, target),
            minimum_score=110,
        )
        if best_match is None:
            missing_columns.append(normalized_column)
            continue
        mapped_columns[normalized_column] = best_match.source_column
        used_columns.add(best_match.source_column)

    source_export_names, source_rename_map = build_source_export_names(source_df.columns)
    return MappingResult(
        mapped_columns=mapped_columns,
        missing_columns=missing_columns,
        source_export_names=source_export_names,
        source_rename_map=source_rename_map,
    )


def build_source_export_names(source_columns: pd.Index) -> tuple[list[str], dict[str, str]]:
    reserved_names = set(NORMALIZED_COLUMNS)
    export_names: list[str] = []
    rename_map: dict[str, str] = {}

    for column in source_columns:
        export_name = column
        if export_name in reserved_names or export_name in export_names:
            base_name = f"source__{column}"
            export_name = base_name
            suffix = 2
            while export_name in reserved_names or export_name in export_names:
                export_name = f"{base_name}_{suffix}"
                suffix += 1
            rename_map[column] = export_name
        export_names.append(export_name)

    return export_names, rename_map


def normalize_event_date_value(value: object) -> object:
    cleaned = clean_nullable_string(value)
    if pd.isna(cleaned):
        return pd.NA

    try:
        parsed = pd.to_datetime(cleaned, errors="coerce")
    except Exception:
        parsed = pd.NaT

    if pd.isna(parsed):
        return cleaned

    if isinstance(parsed, pd.Timestamp) and parsed.tzinfo is not None:
        return parsed.isoformat()

    if isinstance(parsed, pd.Timestamp) and parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0:
        return parsed.strftime("%Y-%m-%d")

    return parsed.isoformat()


def build_path_series(call_ids: pd.Series, base_dir: Path, suffix: str) -> pd.Series:
    def path_for_call_id(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        call_id = str(value)
        return str(base_dir / f"{sanitize_filename_component(call_id)}{suffix}")

    return call_ids.map(path_for_call_id)


def build_inventory_frame(
    source_df: pd.DataFrame,
    mapping: MappingResult,
    *,
    raw_dir: Path,
    wav_dir: Path,
    channel_dir: Path,
) -> pd.DataFrame:
    inventory = pd.DataFrame(index=source_df.index)
    inventory["row_id"] = range(1, len(source_df) + 1)

    for normalized_column in SOURCE_DERIVED_COLUMNS:
        source_column = mapping.mapped_columns.get(normalized_column)
        if source_column is None:
            inventory[normalized_column] = pd.Series([pd.NA] * len(source_df), dtype="object")
            continue

        source_series = source_df[source_column]
        if normalized_column == "event_date":
            inventory[normalized_column] = source_series.map(normalize_event_date_value)
        else:
            inventory[normalized_column] = source_series.map(clean_nullable_string)

    inventory["local_audio_path"] = build_path_series(inventory["call_id"], raw_dir, ".mp3")
    inventory["wav_audio_path"] = build_path_series(inventory["call_id"], wav_dir, ".wav")
    inventory["channel_0_path"] = build_path_series(inventory["call_id"], channel_dir, "_ch0.wav")
    inventory["channel_1_path"] = build_path_series(inventory["call_id"], channel_dir, "_ch1.wav")
    inventory["borrower_channel"] = pd.NA
    inventory["channel_assignment_method"] = pd.NA
    inventory["channel_assignment_confidence"] = pd.NA
    inventory["channels"] = pd.Series([pd.NA] * len(source_df), dtype="Int64")
    inventory["sample_rate"] = pd.Series([pd.NA] * len(source_df), dtype="Int64")
    inventory["duration_sec"] = pd.Series([pd.NA] * len(source_df), dtype="Float64")
    inventory["audio_structure"] = "unknown"
    inventory["recommended_next_step"] = "manual_review"
    inventory["transcript_source"] = pd.NA
    inventory["raw_transcript"] = pd.NA
    inventory["normalized_transcript"] = pd.NA
    inventory["download_status"] = "pending"
    inventory["audio_convert_status"] = "pending"
    inventory["segmentation_status"] = "pending"
    inventory["alignment_status"] = "pending"
    inventory["quality_tier"] = "unreviewed"
    inventory["split"] = "unset"
    inventory["slice_tags"] = "[]"

    source_export_df = source_df.copy()
    source_export_df.columns = mapping.source_export_names

    ordered_columns = NORMALIZED_COLUMNS + list(source_export_df.columns)
    return pd.concat([inventory[NORMALIZED_COLUMNS], source_export_df], axis=1)[ordered_columns]


def is_valid_http_url(value: object) -> bool:
    if pd.isna(value):
        return False
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def parse_positive_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def parse_non_negative_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return round(parsed, 3)


def derive_audio_routing(channels: int | None) -> tuple[str, str]:
    if channels == 1:
        return "mono", "diarization_or_role_filter_first"
    if channels == 2:
        return "stereo", "split_channels_first"
    if channels is not None and channels > 2:
        return "multi_channel", "manual_review"
    return "unknown", "manual_review"


def build_audio_probe_result(
    *,
    channels: int | None,
    sample_rate: int | None,
    duration_sec: float | None,
) -> AudioProbeResult:
    audio_structure, recommended_next_step = derive_audio_routing(channels)
    return AudioProbeResult(
        channels=channels,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        audio_structure=audio_structure,
        recommended_next_step=recommended_next_step,
    )


def probe_audio_metadata(path: Path) -> AudioProbeResult:
    default_result = build_audio_probe_result(channels=None, sample_rate=None, duration_sec=None)
    if not path.exists():
        logging.warning("Audio probe skipped because the source file is missing: %s", path)
        return default_result

    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]

    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logging.warning("ffprobe failed for %s: %s", path, exc.stderr.strip())
        return default_result

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        logging.warning("Failed to parse ffprobe JSON for %s: %s", path, exc)
        return default_result

    streams = payload.get("streams") or []
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if audio_stream is None and streams:
        audio_stream = streams[0]

    if audio_stream is None:
        logging.warning("ffprobe returned no audio stream for %s", path)
        return default_result

    channels = parse_positive_int(audio_stream.get("channels"))
    sample_rate = parse_positive_int(audio_stream.get("sample_rate"))
    duration_sec = parse_non_negative_float(audio_stream.get("duration"))
    if duration_sec is None:
        duration_sec = parse_non_negative_float((payload.get("format") or {}).get("duration"))

    probe_result = build_audio_probe_result(
        channels=channels,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
    )
    logging.info(
        "Audio probe for %s: channels=%s sample_rate=%s duration_sec=%s audio_structure=%s recommended_next_step=%s",
        path,
        probe_result.channels if probe_result.channels is not None else "unknown",
        probe_result.sample_rate if probe_result.sample_rate is not None else "unknown",
        probe_result.duration_sec if probe_result.duration_sec is not None else "unknown",
        probe_result.audio_structure,
        probe_result.recommended_next_step,
    )
    return probe_result


def download_audio(
    session: requests.Session,
    *,
    audio_url: str,
    output_path: Path,
    timeout: int,
    retries: int,
    retry_backoff_seconds: float,
    overwrite: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        logging.debug("Reusing existing audio file: %s", output_path)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with session.get(
                audio_url,
                stream=True,
                timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, timeout),
            ) as response:
                response.raise_for_status()
                with temp_output_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temp_output_path.replace(output_path)
            return True
        except (requests.RequestException, OSError) as exc:
            logging.warning(
                "Download attempt %s/%s failed for %s: %s",
                attempt,
                retries,
                audio_url,
                exc,
            )
            if temp_output_path.exists():
                temp_output_path.unlink()
            if output_path.exists() and overwrite:
                output_path.unlink()
            if attempt < retries:
                time.sleep(retry_backoff_seconds * attempt)

    return False


def convert_audio_to_wav(*, input_path: Path, output_path: Path, overwrite: bool) -> bool:
    if not input_path.exists():
        return False
    if output_path.exists() and not overwrite:
        logging.debug("Reusing existing wav file: %s", output_path)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "8000",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        logging.error("ffmpeg conversion failed for %s: %s", input_path, exc.stderr.strip())
        if output_path.exists():
            output_path.unlink()
        return False


def split_stereo_channels(
    *,
    input_path: Path,
    channel_0_path: Path,
    channel_1_path: Path,
    overwrite: bool,
) -> bool:
    if not input_path.exists():
        logging.warning("Channel split skipped because source audio is missing: %s", input_path)
        return False

    if channel_0_path.exists() and channel_1_path.exists() and not overwrite:
        logging.debug("Reusing existing split channel files for %s", input_path)
        return True

    channel_0_path.parent.mkdir(parents=True, exist_ok=True)
    channel_1_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        "[0:a]pan=mono|c0=c0[ch0];[0:a]pan=mono|c0=c1[ch1]",
        "-map",
        "[ch0]",
        "-ar",
        "8000",
        "-ac",
        "1",
        str(channel_0_path),
        "-map",
        "[ch1]",
        "-ar",
        "8000",
        "-ac",
        "1",
        str(channel_1_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        logging.error("ffmpeg channel split failed for %s: %s", input_path, exc.stderr.strip())
        for output_path in (channel_0_path, channel_1_path):
            if output_path.exists():
                output_path.unlink()
        return False


def path_exists(value: object) -> bool:
    if pd.isna(value):
        return False
    return Path(str(value)).exists()


def normalize_borrower_channel(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    aliases = {
        "0": "ch0",
        "1": "ch1",
        "ch0": "ch0",
        "ch1": "ch1",
        "channel_0": "ch0",
        "channel_1": "ch1",
        "channel0": "ch0",
        "channel1": "ch1",
    }
    return aliases.get(text)


def derive_agent_channel(value: object) -> object:
    normalized = normalize_borrower_channel(value)
    if normalized == "ch0":
        return "ch1"
    if normalized == "ch1":
        return "ch0"
    return pd.NA


def load_channel_map(path: Path) -> dict[str, dict[str, str]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"Channel map file not found: {resolved}")

    try:
        channel_map_df = pd.read_csv(resolved, dtype=str, keep_default_na=True, encoding="utf-8-sig")
    except Exception as exc:  # pragma: no cover - defensive exit path
        raise SystemExit(f"Failed to read channel map CSV {resolved}: {exc}") from exc

    canonical_columns = {canonicalize_column_name(column): column for column in channel_map_df.columns}
    call_id_column = canonical_columns.get("call_id")
    borrower_channel_column = canonical_columns.get("borrower_channel")
    confidence_column = canonical_columns.get("channel_assignment_confidence") or canonical_columns.get("confidence")

    if call_id_column is None or borrower_channel_column is None:
        raise SystemExit(
            f"{resolved}: channel map must include call_id and borrower_channel columns."
        )

    channel_map: dict[str, dict[str, str]] = {}
    for row_index, row in channel_map_df.iterrows():
        call_id = clean_nullable_string(row[call_id_column])
        if pd.isna(call_id):
            logging.warning("Skipping channel map row %s because call_id is empty.", row_index + 2)
            continue

        borrower_channel = normalize_borrower_channel(row[borrower_channel_column])
        if borrower_channel is None:
            logging.warning(
                "Skipping channel map row %s for call_id=%s because borrower_channel is invalid.",
                row_index + 2,
                call_id,
            )
            continue

        confidence = clean_nullable_string(row[confidence_column]) if confidence_column else pd.NA
        channel_map[str(call_id)] = {
            "borrower_channel": borrower_channel,
            "channel_assignment_method": "manual_channel_map",
            "channel_assignment_confidence": (
                str(confidence) if not pd.isna(confidence) else "high"
            ),
        }

    logging.info("Loaded %s borrower channel assignments from %s", len(channel_map), resolved)
    return channel_map


def apply_borrower_channel_assignments(
    inventory_df: pd.DataFrame,
    *,
    channel_map: dict[str, dict[str, str]],
    default_borrower_channel: str | None,
) -> pd.DataFrame:
    matched_call_ids: set[str] = set()

    for idx, row in inventory_df.iterrows():
        if row["recommended_next_step"] != "split_channels_first":
            continue

        call_id = clean_nullable_string(row["call_id"])
        if pd.isna(call_id):
            continue

        assignment = channel_map.get(str(call_id))
        if assignment is not None:
            matched_call_ids.add(str(call_id))
            inventory_df.at[idx, "borrower_channel"] = assignment["borrower_channel"]
            inventory_df.at[idx, "channel_assignment_method"] = assignment["channel_assignment_method"]
            inventory_df.at[idx, "channel_assignment_confidence"] = assignment["channel_assignment_confidence"]
            continue

        if default_borrower_channel is not None:
            inventory_df.at[idx, "borrower_channel"] = default_borrower_channel
            inventory_df.at[idx, "channel_assignment_method"] = "global_default"
            inventory_df.at[idx, "channel_assignment_confidence"] = "medium"
            continue

        inventory_df.at[idx, "borrower_channel"] = pd.NA
        inventory_df.at[idx, "channel_assignment_method"] = "manual_audit_required"
        inventory_df.at[idx, "channel_assignment_confidence"] = "unassigned"

    unused_call_ids = sorted(set(channel_map) - matched_call_ids)
    for call_id in unused_call_ids:
        logging.warning("Channel map entry for call_id=%s was not found in the current inventory.", call_id)

    return inventory_df


def build_borrower_audit_frame(
    inventory_df: pd.DataFrame,
    *,
    audit_sample_size: int | None,
    audit_seed: int,
) -> pd.DataFrame:
    eligible_df = inventory_df.loc[
        inventory_df["recommended_next_step"] == "split_channels_first"
    ].copy()

    eligible_df["agent_channel"] = eligible_df["borrower_channel"].map(derive_agent_channel)
    eligible_df["channel_split_ready"] = eligible_df.apply(
        lambda row: path_exists(row["channel_0_path"]) and path_exists(row["channel_1_path"]),
        axis=1,
    )
    eligible_df["audit_status"] = "needs_manual_review"
    eligible_df.loc[eligible_df["borrower_channel"].isin(("ch0", "ch1")), "audit_status"] = "assigned"
    eligible_df.loc[~eligible_df["channel_split_ready"], "audit_status"] = "split_missing"
    eligible_df["borrower_channel_options"] = "ch0|ch1"
    eligible_df["audit_notes"] = ""

    audit_columns = [
        "call_id",
        "borrower_channel",
        "agent_channel",
        "channel_assignment_method",
        "channel_assignment_confidence",
        "audit_status",
        "borrower_channel_options",
        "channel_0_path",
        "channel_1_path",
        "local_audio_path",
        "duration_sec",
        "sample_rate",
        "audio_structure",
        "recommended_next_step",
        "channel_split_ready",
        "audit_notes",
    ]
    eligible_df = eligible_df[audit_columns].sort_values("call_id", kind="stable").reset_index(drop=True)

    if audit_sample_size is None or audit_sample_size >= len(eligible_df):
        return eligible_df
    if audit_sample_size <= 0:
        raise SystemExit("--audit-sample-size must be greater than 0 when provided.")
    return (
        eligible_df.sample(n=audit_sample_size, random_state=audit_seed)
        .sort_values("call_id", kind="stable")
        .reset_index(drop=True)
    )


def write_borrower_audit_report(
    inventory_df: pd.DataFrame,
    *,
    output_dir: Path,
    audit_report: Path | None,
    audit_sample_size: int | None,
    audit_seed: int,
) -> Path:
    report_path = (
        audit_report.expanduser()
        if audit_report is not None
        else output_dir / "borrower_channel_audit.csv"
    )
    audit_df = build_borrower_audit_frame(
        inventory_df,
        audit_sample_size=audit_sample_size,
        audit_seed=audit_seed,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(report_path, index=False)
    return report_path


def process_audio(inventory_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    with requests.Session() as session:
        session.headers.update({"User-Agent": REQUEST_USER_AGENT})

        total_rows = len(inventory_df)
        for idx, row in inventory_df.iterrows():
            row_id = row["row_id"]
            call_id = row["call_id"]
            audio_url = row["audio_url"]
            local_audio_path = row["local_audio_path"]
            wav_audio_path = row["wav_audio_path"]
            channel_0_path = row["channel_0_path"]
            channel_1_path = row["channel_1_path"]

            if (
                pd.isna(call_id)
                or pd.isna(local_audio_path)
                or pd.isna(wav_audio_path)
                or pd.isna(channel_0_path)
                or pd.isna(channel_1_path)
            ):
                logging.warning("Row %s is missing call_id; audio processing skipped.", row_id)
                inventory_df.at[idx, "download_status"] = "failed"
                inventory_df.at[idx, "audio_convert_status"] = "failed"
                continue

            if not is_valid_http_url(audio_url):
                logging.warning("Row %s (%s) is missing or has an invalid audio_url.", row_id, call_id)
                inventory_df.at[idx, "download_status"] = "failed"
                inventory_df.at[idx, "audio_convert_status"] = "failed"
                continue

            logging.info("Processing row %s/%s for call_id=%s", row_id, total_rows, call_id)
            download_ok = download_audio(
                session,
                audio_url=str(audio_url),
                output_path=Path(str(local_audio_path)),
                timeout=args.timeout,
                retries=args.retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                overwrite=args.overwrite,
            )
            inventory_df.at[idx, "download_status"] = "done" if download_ok else "failed"

            if not download_ok:
                inventory_df.at[idx, "audio_convert_status"] = "failed"
                continue

            probe_result = probe_audio_metadata(Path(str(local_audio_path)))
            inventory_df.at[idx, "channels"] = (
                probe_result.channels if probe_result.channels is not None else pd.NA
            )
            inventory_df.at[idx, "sample_rate"] = (
                probe_result.sample_rate if probe_result.sample_rate is not None else pd.NA
            )
            inventory_df.at[idx, "duration_sec"] = (
                probe_result.duration_sec if probe_result.duration_sec is not None else pd.NA
            )
            inventory_df.at[idx, "audio_structure"] = probe_result.audio_structure
            inventory_df.at[idx, "recommended_next_step"] = probe_result.recommended_next_step

            if probe_result.channels == 2:
                split_ok = split_stereo_channels(
                    input_path=Path(str(local_audio_path)),
                    channel_0_path=Path(str(channel_0_path)),
                    channel_1_path=Path(str(channel_1_path)),
                    overwrite=args.overwrite,
                )
                if split_ok:
                    logging.info(
                        "Channel split succeeded for call_id=%s -> %s, %s",
                        call_id,
                        channel_0_path,
                        channel_1_path,
                    )
                else:
                    logging.warning("Channel split failed for call_id=%s", call_id)

            convert_ok = convert_audio_to_wav(
                input_path=Path(str(local_audio_path)),
                output_path=Path(str(wav_audio_path)),
                overwrite=args.overwrite,
            )
            inventory_df.at[idx, "audio_convert_status"] = "done" if convert_ok else "failed"

    return inventory_df


def save_outputs(inventory_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "data_inventory.csv"
    parquet_path = output_dir / "data_inventory.parquet"

    inventory_df.to_csv(csv_path, index=False)
    try:
        inventory_df.to_parquet(parquet_path, index=False, engine="pyarrow")
    except ImportError as exc:  # pragma: no cover - dependency error
        raise SystemExit("pyarrow is required to write Parquet output. Install it with pip install pyarrow.") from exc

    return csv_path, parquet_path


def log_mapping_summary(mapping: MappingResult) -> None:
    if mapping.mapped_columns:
        logging.info("Columns mapped automatically:")
        for normalized_column, source_column in mapping.mapped_columns.items():
            logging.info("  %s <- %s", normalized_column, source_column)
    else:
        logging.warning("No source columns were mapped automatically.")

    if mapping.missing_columns:
        logging.warning("Missing normalized source fields: %s", ", ".join(mapping.missing_columns))
    else:
        logging.info("All normalized source-derived fields were mapped.")

    if mapping.source_rename_map:
        logging.info("Renamed source columns to avoid export collisions:")
        for original_name, renamed_name in mapping.source_rename_map.items():
            logging.info("  %s -> %s", original_name, renamed_name)


def log_processing_summary(inventory_df: pd.DataFrame) -> None:
    total_rows = len(inventory_df)
    download_success = int((inventory_df["download_status"] == "done").sum())
    download_failure = int((inventory_df["download_status"] == "failed").sum())
    wav_success = int((inventory_df["audio_convert_status"] == "done").sum())
    wav_failure = int((inventory_df["audio_convert_status"] == "failed").sum())
    split_channels_first = int((inventory_df["recommended_next_step"] == "split_channels_first").sum())
    diarization_first = int(
        (inventory_df["recommended_next_step"] == "diarization_or_role_filter_first").sum()
    )
    manual_review = int((inventory_df["recommended_next_step"] == "manual_review").sum())
    channel_split_success = int(
        inventory_df.apply(
            lambda row: row["recommended_next_step"] == "split_channels_first"
            and path_exists(row["channel_0_path"])
            and path_exists(row["channel_1_path"]),
            axis=1,
        ).sum()
    )
    borrower_assigned = int(inventory_df["borrower_channel"].isin(("ch0", "ch1")).sum())
    borrower_unassigned = int(
        inventory_df.apply(
            lambda row: row["recommended_next_step"] == "split_channels_first"
            and normalize_borrower_channel(row["borrower_channel"]) is None,
            axis=1,
        ).sum()
    )

    logging.info("Summary report:")
    logging.info("  total_rows=%s", total_rows)
    logging.info("  download_success=%s", download_success)
    logging.info("  download_failure=%s", download_failure)
    logging.info("  wav_conversion_success=%s", wav_success)
    logging.info("  wav_conversion_failure=%s", wav_failure)
    logging.info("  split_channels_first=%s", split_channels_first)
    logging.info("  channel_split_success=%s", channel_split_success)
    logging.info("  diarization_or_role_filter_first=%s", diarization_first)
    logging.info("  manual_review=%s", manual_review)
    logging.info("  borrower_assigned=%s", borrower_assigned)
    logging.info("  borrower_unassigned=%s", borrower_unassigned)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_ffmpeg_available()
    ensure_ffprobe_available()

    input_path = args.input.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser()
    wav_dir = args.wav_dir.expanduser()
    channel_dir = args.channel_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    channel_map = load_channel_map(args.channel_map) if args.channel_map else {}

    logging.info("Reading source CSV from %s", input_path)
    source_df = read_source_csv(input_path)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise SystemExit("--max-rows must be greater than 0 when provided.")
        source_df = source_df.head(args.max_rows).copy()
        logging.info("Limiting processing to the first %s rows", len(source_df))
    logging.info("Loaded %s rows from input CSV", len(source_df))

    mapping = infer_schema_mapping(source_df)
    log_mapping_summary(mapping)

    inventory_df = build_inventory_frame(
        source_df,
        mapping,
        raw_dir=raw_dir,
        wav_dir=wav_dir,
        channel_dir=channel_dir,
    )
    inventory_df = process_audio(inventory_df, args)
    inventory_df = apply_borrower_channel_assignments(
        inventory_df,
        channel_map=channel_map,
        default_borrower_channel=args.default_borrower_channel,
    )

    csv_path, parquet_path = save_outputs(inventory_df, output_dir)
    audit_report_path = write_borrower_audit_report(
        inventory_df,
        output_dir=output_dir,
        audit_report=args.audit_report,
        audit_sample_size=args.audit_sample_size,
        audit_seed=args.audit_seed,
    )
    log_processing_summary(inventory_df)

    logging.info("Wrote inventory CSV to %s", csv_path)
    logging.info("Wrote inventory Parquet to %s", parquet_path)
    logging.info("Wrote borrower audit CSV to %s", audit_report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
