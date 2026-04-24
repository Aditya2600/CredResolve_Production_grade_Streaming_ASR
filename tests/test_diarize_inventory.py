from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf

from tools.diarization.providers.base import DiarizationResult, SpeakerTurn
from tools.diarize_inventory import process_inventory


class _FakeProvider:
    def __init__(self, turns):
        self.turns = tuple(turns)
        self.calls = 0

    def diarize(self, prepared_audio, config, *, working_dir: Path):
        del prepared_audio, config, working_dir
        self.calls += 1
        return DiarizationResult(provider="nemo_telephony", turns=self.turns)


def build_args(inventory: Path, output_dir: Path, **overrides):
    values = {
        "inventory": inventory,
        "output_dir": output_dir,
        "audio_mode": "auto",
        "segments_jsonl": None,
        "limit": None,
        "respect_ready_status": True,
        "overwrite": False,
        "progress_every": 25,
        "min_segment_sec": 0.5,
        "max_segment_sec": 20.0,
        "multiscale_window_sec": (1.5, 1.25, 1.0, 0.75, 0.5),
        "multiscale_hop_sec": (0.75, 0.625, 0.5, 0.375, 0.25),
        "speaker_count_mode": "estimate",
        "fixed_speakers": None,
        "min_speakers": 1,
        "max_speakers": 2,
        "clustering_threshold": 0.25,
        "overlap": False,
        "vad_onset": 0.1,
        "vad_offset": 0.1,
        "vad_pad_onset": 0.1,
        "vad_pad_offset": 0.0,
        "vad_min_duration_on": 0.0,
        "vad_min_duration_off": 0.2,
        "vad_window_sec": 0.15,
        "vad_shift_sec": 0.01,
        "vad_overlapping": 0.5,
        "vad_overlap": 0.5,
        "nemo_vad_model": "vad_multilingual_marblenet",
        "nemo_speaker_model": "titanet_large",
        "nemo_msdd_model": "diar_msdd_telephonic",
        "timestamp_source": "none",
        "worker_url": "http://localhost:9000/v1/transcribe",
        "timeout_sec": 20.0,
        "log_level": "INFO",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_inventory(tmp_path: Path) -> tuple[Path, Path]:
    sample_rate = 8000
    wav_path = tmp_path / "call.wav"
    inventory_path = tmp_path / "inventory.csv"
    audio = np.zeros(sample_rate, dtype=np.float32)
    sf.write(wav_path, audio, sample_rate, subtype="PCM_16")
    inventory = pd.DataFrame(
        [
            {
                "row_id": "1",
                "call_id": "call-001",
                "wav_audio_path": str(wav_path),
                "local_audio_path": str(wav_path),
                "recommended_next_step": "diarization_or_role_filter_first",
                "segmentation_status": "ready_mono_segmentation",
            }
        ]
    )
    inventory.to_csv(inventory_path, index=False)
    return inventory_path, wav_path


def _write_segments_jsonl(tmp_path: Path, wav_path: Path) -> Path:
    segments_jsonl_path = tmp_path / "segments.jsonl"
    rows = [
        {
            "segment_id": "call-001-mono-0001",
            "call_id": "call-001",
            "row_id": "1",
            "audio_filepath": str(wav_path),
            "source_audio_filepath": str(wav_path),
            "lang": "hi",
            "text": "नमस्ते",
            "segment_start_sec": 0.0,
            "segment_end_sec": 0.4,
        },
        {
            "segment_id": "call-001-mono-0002",
            "call_id": "call-001",
            "row_id": "1",
            "audio_filepath": str(wav_path),
            "source_audio_filepath": str(wav_path),
            "lang": "hi",
            "text": "जी बोलिए",
            "segment_start_sec": 0.5,
            "segment_end_sec": 0.9,
        },
    ]
    segments_jsonl_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return segments_jsonl_path


def test_process_inventory_marks_no_speech_when_provider_returns_no_turns(tmp_path: Path):
    inventory_path, _wav_path = _write_inventory(tmp_path)
    output_dir = tmp_path / "diarized"
    provider = _FakeProvider(())

    summary = process_inventory(build_args(inventory_path, output_dir), provider=provider)

    assert provider.calls == 1
    assert summary["status_counts"] == {"diarization_no_speech": 1}
    updated_inventory = pd.read_csv(output_dir / "data_inventory_diarized.csv", dtype=str, keep_default_na=True)
    assert updated_inventory.loc[0, "diarization_status"] == "diarization_no_speech"


def test_process_inventory_writes_turns_and_speakerized_transcript_schema(tmp_path: Path):
    inventory_path, wav_path = _write_inventory(tmp_path)
    output_dir = tmp_path / "diarized"
    segments_jsonl_path = _write_segments_jsonl(tmp_path, wav_path)
    provider = _FakeProvider(
        (
            SpeakerTurn(speaker_label="speaker_0", start_sec=0.0, end_sec=0.4),
            SpeakerTurn(speaker_label="speaker_1", start_sec=0.5, end_sec=0.9),
        )
    )

    def fake_timestamp_client(audio_path: Path, language: str, segment_id: str):
        del audio_path, language
        if segment_id.endswith("0001"):
            return {
                "text": "नमस्ते",
                "word_timestamps": [{"word": "नमस्ते", "start_sec": 0.0, "end_sec": 0.35, "word_index": 0}],
            }
        return {
            "text": "जी बोलिए",
            "word_timestamps": [
                {"word": "जी", "start_sec": 0.0, "end_sec": 0.12, "word_index": 0},
                {"word": "बोलिए", "start_sec": 0.12, "end_sec": 0.32, "word_index": 1},
            ],
        }

    summary = process_inventory(
        build_args(inventory_path, output_dir, segments_jsonl=segments_jsonl_path),
        provider=provider,
        timestamp_client=fake_timestamp_client,
    )

    assert summary["turns_written"] == 2
    assert summary["speakerized_segments_written"] == 2
    diarization_rows = [
        json.loads(line)
        for line in (output_dir / "diarization_turns.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["speaker_label"] for row in diarization_rows] == ["speaker_0", "speaker_1"]

    speakerized_rows = [
        json.loads(line)
        for line in (output_dir / "speakerized_transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["speaker_label"] for row in speakerized_rows] == ["speaker_0", "speaker_1"]
    assert speakerized_rows[0]["words"][0]["speaker_label"] == "speaker_0"
    assert speakerized_rows[1]["words"][0]["speaker_label"] == "speaker_1"
    assert set(speakerized_rows[0].keys()) >= {
        "segment_id",
        "call_id",
        "text",
        "segment_start_sec",
        "segment_end_sec",
        "speaker_label",
        "overlap_flag",
        "words",
    }


def test_process_inventory_handles_one_speaker_audio(tmp_path: Path):
    inventory_path, wav_path = _write_inventory(tmp_path)
    output_dir = tmp_path / "diarized"
    segments_jsonl_path = _write_segments_jsonl(tmp_path, wav_path)
    provider = _FakeProvider((SpeakerTurn(speaker_label="speaker_0", start_sec=0.0, end_sec=1.0),))

    summary = process_inventory(
        build_args(inventory_path, output_dir, segments_jsonl=segments_jsonl_path, timestamp_source="none"),
        provider=provider,
    )

    assert summary["status_counts"]["diarized_nemo_telephony"] == 1
    speakerized_rows = [
        json.loads(line)
        for line in (output_dir / "speakerized_transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row["speaker_label"] == "speaker_0" for row in speakerized_rows)
