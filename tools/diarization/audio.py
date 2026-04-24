from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    from eval_indicvoices_wer import resample_linear, to_mono
except ImportError:  # pragma: no cover
    from tools.eval_indicvoices_wer import resample_linear, to_mono

from tools.diarization.providers.base import PreparedAudio
from tools.segment_inventory_with_silero import (
    clean_optional_str,
    read_audio,
    resolve_audio_target,
    resolve_channel_path,
)


def _write_normalized_audio(
    *,
    waveform: np.ndarray,
    sample_rate: int,
    output_path: Path,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not output_path.exists():
        sf.write(str(output_path), np.asarray(waveform, dtype=np.float32), int(sample_rate), subtype="PCM_16")


def prepare_audio_for_diarization(
    audio_path: Path,
    *,
    output_dir: Path,
    target_sample_rate: int = 16000,
    overwrite: bool = False,
) -> PreparedAudio:
    waveform, sample_rate = read_audio(audio_path)
    mono_waveform = to_mono(np.asarray(waveform, dtype=np.float32))
    if sample_rate != target_sample_rate:
        mono_waveform = resample_linear(mono_waveform, int(sample_rate), int(target_sample_rate))
        sample_rate = target_sample_rate
    mono_waveform = np.asarray(mono_waveform, dtype=np.float32)

    normalized_path = output_dir / f"{audio_path.stem}_mono{target_sample_rate}.wav"
    _write_normalized_audio(
        waveform=mono_waveform,
        sample_rate=int(sample_rate),
        output_path=normalized_path,
        overwrite=overwrite,
    )
    return PreparedAudio(
        audio_path=normalized_path,
        waveform=mono_waveform,
        sample_rate=int(sample_rate),
        source_audio_path=audio_path,
        source_kind="normalized_mono_16k",
    )


def prepare_split_channels_for_diarization(
    channel_0_path: Path,
    channel_1_path: Path,
    *,
    output_dir: Path,
    target_sample_rate: int = 16000,
    overwrite: bool = False,
) -> PreparedAudio:
    waveform_0, sample_rate_0 = read_audio(channel_0_path)
    waveform_1, sample_rate_1 = read_audio(channel_1_path)
    mono_0 = to_mono(np.asarray(waveform_0, dtype=np.float32))
    mono_1 = to_mono(np.asarray(waveform_1, dtype=np.float32))
    if sample_rate_0 != target_sample_rate:
        mono_0 = resample_linear(mono_0, int(sample_rate_0), int(target_sample_rate))
    if sample_rate_1 != target_sample_rate:
        mono_1 = resample_linear(mono_1, int(sample_rate_1), int(target_sample_rate))

    max_length = max(len(mono_0), len(mono_1))
    if len(mono_0) < max_length:
        mono_0 = np.pad(mono_0, (0, max_length - len(mono_0)))
    if len(mono_1) < max_length:
        mono_1 = np.pad(mono_1, (0, max_length - len(mono_1)))
    downmixed = ((mono_0 + mono_1) / 2.0).astype(np.float32, copy=False)

    normalized_path = output_dir / "split_channels_downmixed_mono16k.wav"
    _write_normalized_audio(
        waveform=downmixed,
        sample_rate=int(target_sample_rate),
        output_path=normalized_path,
        overwrite=overwrite,
    )
    return PreparedAudio(
        audio_path=normalized_path,
        waveform=downmixed,
        sample_rate=int(target_sample_rate),
        source_audio_path=normalized_path,
        source_kind="downmixed_split_channels",
    )


def prepare_row_audio_for_diarization(
    row: Any,
    *,
    audio_mode: str,
    output_dir: Path,
    target_sample_rate: int = 16000,
    overwrite: bool = False,
) -> tuple[PreparedAudio | None, str | None]:
    wav_audio_path = clean_optional_str(row.get("wav_audio_path"))
    local_audio_path = clean_optional_str(row.get("local_audio_path"))

    if audio_mode == "auto":
        if wav_audio_path:
            return (
                prepare_audio_for_diarization(
                    Path(wav_audio_path).expanduser().resolve(),
                    output_dir=output_dir,
                    target_sample_rate=target_sample_rate,
                    overwrite=overwrite,
                ),
                None,
            )
        if local_audio_path and local_audio_path.lower().endswith(".wav"):
            return (
                prepare_audio_for_diarization(
                    Path(local_audio_path).expanduser().resolve(),
                    output_dir=output_dir,
                    target_sample_rate=target_sample_rate,
                    overwrite=overwrite,
                ),
                None,
            )
        channel_0_path = resolve_channel_path(row, "ch0")
        channel_1_path = resolve_channel_path(row, "ch1")
        if channel_0_path is not None and channel_1_path is not None:
            return (
                prepare_split_channels_for_diarization(
                    channel_0_path,
                    channel_1_path,
                    output_dir=output_dir,
                    target_sample_rate=target_sample_rate,
                    overwrite=overwrite,
                ),
                None,
            )
        return None, "blocked_missing_diarization_audio"

    target, error_status = resolve_audio_target(row, audio_mode=audio_mode)
    if target is None:
        return None, error_status or "blocked_missing_diarization_audio"
    return (
        prepare_audio_for_diarization(
            target.audio_path,
            output_dir=output_dir,
            target_sample_rate=target_sample_rate,
            overwrite=overwrite,
        ),
        None,
    )
