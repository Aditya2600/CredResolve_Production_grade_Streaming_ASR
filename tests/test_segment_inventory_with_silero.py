import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf

from tools.diarization.providers.base import DiarizationResult, SpeakerTurn
from tools.segment_inventory_with_silero import (
    AudioTarget,
    SpeechSegment,
    TranscriptUtterance,
    align_segments_to_utterances,
    parse_transcript_utterances,
    process_inventory,
    resolve_audio_target,
)


def build_args(inventory: Path, output_dir: Path, **overrides):
    values = {
        "inventory": inventory,
        "output_dir": output_dir,
        "audio_mode": "auto",
        "text_column": None,
        "language_column": None,
        "language": None,
        "limit": None,
        "respect_ready_status": True,
        "overwrite": False,
        "output_sample_rate": 8000,
        "vad_resample_rate": 16000,
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 20.0,
        "min_silence_duration_ms": 150,
        "speech_pad_ms": 60,
        "diarization_turns": None,
        "enable_diarization": False,
        "diarization_audio_mode": "auto",
        "diarization_min_segment_sec": 0.5,
        "diarization_max_segment_sec": 20.0,
        "diarization_multiscale_window_sec": (1.5, 1.25, 1.0, 0.75, 0.5),
        "diarization_multiscale_hop_sec": (0.75, 0.625, 0.5, 0.375, 0.25),
        "diarization_speaker_count_mode": "estimate",
        "diarization_fixed_speakers": None,
        "diarization_min_speakers": 1,
        "diarization_max_speakers": 2,
        "diarization_clustering_threshold": 0.25,
        "diarization_overlap": False,
        "diarization_vad_onset": 0.1,
        "diarization_vad_offset": 0.1,
        "diarization_vad_pad_onset": 0.1,
        "diarization_vad_pad_offset": 0.0,
        "diarization_vad_min_duration_on": 0.0,
        "diarization_vad_min_duration_off": 0.2,
        "diarization_vad_window_sec": 0.15,
        "diarization_vad_shift_sec": 0.01,
        "diarization_vad_overlap": 0.5,
        "diarization_vad_model": "vad_multilingual_marblenet",
        "diarization_speaker_model": "titanet_large",
        "diarization_msdd_model": "diar_msdd_telephonic",
        "progress_every": 25,
        "log_level": "INFO",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeDiarizationProvider:
    def __init__(self, turns):
        self.turns = tuple(turns)
        self.calls = 0

    def diarize(self, prepared_audio, config, *, working_dir: Path):
        del prepared_audio, config, working_dir
        self.calls += 1
        return DiarizationResult(provider="nemo_telephony", turns=self.turns)


def test_parse_transcript_utterances_preserves_roles_and_order():
    payload = json.dumps(
        {
            "interaction_transcript": [
                {"role": "agent", "en_text": "नमस्ते"},
                {"role": "user", "en_text": "जी बोलिए"},
            ]
        },
        ensure_ascii=False,
    )

    utterances = parse_transcript_utterances(payload)

    assert [utterance.text for utterance in utterances] == ["नमस्ते", "जी बोलिए"]
    assert [utterance.role for utterance in utterances] == ["agent", "user"]
    assert [utterance.source_index for utterance in utterances] == [0, 1]


def test_resolve_audio_target_auto_prefers_borrower_channel_for_split_rows():
    row = pd.Series(
        {
            "recommended_next_step": "split_channels_first",
            "borrower_channel": "ch1",
            "channel_0_path": "/tmp/demo_ch0.wav",
            "channel_1_path": "/tmp/demo_ch1.wav",
            "wav_audio_path": "/tmp/demo.wav",
        }
    )

    target, error_status = resolve_audio_target(row, audio_mode="auto")

    assert error_status is None
    assert isinstance(target, AudioTarget)
    assert target.audio_path == Path("/tmp/demo_ch1.wav")
    assert target.target_kind == "borrower_channel"
    assert target.role_filter == "borrower"
    assert target.channel_label == "ch1"


def test_align_segments_to_utterances_merges_extra_vad_segments():
    segments = [
        SpeechSegment(start_sample=0, end_sample=100),
        SpeechSegment(start_sample=110, end_sample=200),
        SpeechSegment(start_sample=400, end_sample=520),
    ]
    utterances = [
        TranscriptUtterance(text="पहला", role="agent", source_index=0),
        TranscriptUtterance(text="दूसरा", role="user", source_index=1),
    ]

    aligned = align_segments_to_utterances(segments, utterances)

    assert len(aligned) == 2
    assert aligned[0][0].start_sample == 0
    assert aligned[0][0].end_sample == 200
    assert [item.text for item in aligned[0][1]] == ["पहला"]
    assert [item.text for item in aligned[1][1]] == ["दूसरा"]


def test_process_inventory_writes_manifest_and_updates_status_with_fake_detector(tmp_path: Path):
    sample_rate = 16000
    wav_path = tmp_path / "call.wav"
    output_dir = tmp_path / "segmented"
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
                "language": "HINDI",
                "slice_tags": "[]",
                "raw_transcript": json.dumps(
                    {
                        "interaction_transcript": [
                            {"role": "agent", "en_text": "नमस्ते"},
                            {"role": "user", "en_text": "जी बोलिए"},
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
        ]
    )
    inventory.to_csv(inventory_path, index=False)

    def fake_detector(audio_values, detected_sample_rate):
        assert detected_sample_rate == 16000
        assert len(audio_values) == sample_rate
        return [
            {"start": 0, "end": 3200},
            {"start": 8000, "end": 12800},
        ]

    summary = process_inventory(
        build_args(inventory_path, output_dir),
        detector=fake_detector,
    )

    assert summary["segments_written"] == 2
    assert summary["status_counts"]["segmented_silero_vad"] == 1

    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["text"] for row in manifest_rows] == ["नमस्ते", "जी बोलिए"]
    assert all(Path(row["audio_filepath"]).exists() for row in manifest_rows)

    updated_inventory = pd.read_csv(output_dir / "data_inventory_segmented.csv", dtype=str, keep_default_na=True)
    assert updated_inventory.loc[0, "segmentation_status"] == "segmented_silero_vad"
    assert "silero_vad" in updated_inventory.loc[0, "slice_tags"]


def test_process_inventory_attaches_diarization_labels_to_segments(tmp_path: Path):
    sample_rate = 16000
    wav_path = tmp_path / "call.wav"
    output_dir = tmp_path / "segmented"
    inventory_path = tmp_path / "inventory.csv"
    diarization_turns_path = tmp_path / "diarization_turns.jsonl"

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
                "language": "HINDI",
                "slice_tags": "[]",
                "raw_transcript": json.dumps(
                    {
                        "interaction_transcript": [
                            {"role": "agent", "en_text": "नमस्ते"},
                            {"role": "user", "en_text": "जी बोलिए"},
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
        ]
    )
    inventory.to_csv(inventory_path, index=False)
    diarization_turns_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "call_id": "call-001",
                        "speaker_label": "speaker_0",
                        "start_sec": 0.0,
                        "end_sec": 0.2,
                        "confidence": 0.98,
                        "overlap_flag": False,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "call_id": "call-001",
                        "speaker_label": "speaker_1",
                        "start_sec": 0.5,
                        "end_sec": 0.8,
                        "confidence": 0.96,
                        "overlap_flag": False,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_detector(audio_values, detected_sample_rate):
        assert detected_sample_rate == 16000
        assert len(audio_values) == sample_rate
        return [
            {"start": 0, "end": 3200},
            {"start": 8000, "end": 12800},
        ]

    summary = process_inventory(
        build_args(inventory_path, output_dir, diarization_turns=diarization_turns_path),
        detector=fake_detector,
    )

    assert summary["segments_written"] == 2
    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["speaker_label"] for row in manifest_rows] == ["speaker_0", "speaker_1"]
    assert [row["overlap_flag"] for row in manifest_rows] == [False, False]
    assert all(row["diarization_used"] is True for row in manifest_rows)
    assert all(row["diarization_source"] == "external_turns_jsonl" for row in manifest_rows)
    assert all(row["speaker_alignment_method"] == "audio_interval_overlap" for row in manifest_rows)


def test_process_inventory_inline_diarization_refines_mono_segments(tmp_path: Path):
    sample_rate = 16000
    wav_path = tmp_path / "call.wav"
    output_dir = tmp_path / "segmented"
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
                "language": "HINDI",
                "slice_tags": "[]",
                "raw_transcript": json.dumps(
                    {
                        "interaction_transcript": [
                            {"role": "agent", "en_text": "नमस्ते"},
                            {"role": "user", "en_text": "जी बोलिए"},
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
        ]
    )
    inventory.to_csv(inventory_path, index=False)

    def fake_detector(audio_values, detected_sample_rate):
        assert detected_sample_rate == 16000
        assert len(audio_values) == sample_rate
        return [{"start": 0, "end": sample_rate}]

    provider = _FakeDiarizationProvider(
        (
            SpeakerTurn(speaker_label="speaker_0", start_sec=0.0, end_sec=0.45),
            SpeakerTurn(speaker_label="speaker_1", start_sec=0.45, end_sec=1.0),
        )
    )

    summary = process_inventory(
        build_args(inventory_path, output_dir, enable_diarization=True),
        detector=fake_detector,
        diarization_provider=provider,
    )

    assert provider.calls == 1
    assert summary["diarization_turns_written"] == 2
    assert summary["diarization_status_counts"]["diarized_nemo_telephony"] == 1
    assert Path(summary["pred_rttm_dir"]).joinpath("call-001.rttm").exists()

    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(manifest_rows) == 2
    assert [row["speaker_label"] for row in manifest_rows] == ["speaker_0", "speaker_1"]
    assert all(row["diarization_used"] is True for row in manifest_rows)
    assert all(row["diarization_source"] == "inline_nemo_telephony" for row in manifest_rows)
    assert all(row["diarization_boundary_refined"] is True for row in manifest_rows)
    assert all(row["transcript_alignment_method"] == "role_order_heuristic" for row in manifest_rows)
    assert all(row["speaker_alignment_method"] == "audio_interval_overlap" for row in manifest_rows)

    updated_inventory = pd.read_csv(output_dir / "data_inventory_segmented.csv", dtype=str, keep_default_na=True)
    assert updated_inventory.loc[0, "segmentation_status"] == "segmented_silero_vad"
    assert updated_inventory.loc[0, "diarization_status"] == "diarized_nemo_telephony"


def test_process_inventory_inline_diarization_no_speech_keeps_segmentation(tmp_path: Path):
    sample_rate = 16000
    wav_path = tmp_path / "call.wav"
    output_dir = tmp_path / "segmented"
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
                "language": "HINDI",
                "slice_tags": "[]",
                "raw_transcript": json.dumps(
                    {
                        "interaction_transcript": [
                            {"role": "agent", "en_text": "नमस्ते"},
                            {"role": "user", "en_text": "जी बोलिए"},
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
        ]
    )
    inventory.to_csv(inventory_path, index=False)

    def fake_detector(audio_values, detected_sample_rate):
        assert detected_sample_rate == 16000
        assert len(audio_values) == sample_rate
        return [
            {"start": 0, "end": 3200},
            {"start": 8000, "end": 12800},
        ]

    provider = _FakeDiarizationProvider(())

    summary = process_inventory(
        build_args(inventory_path, output_dir, enable_diarization=True),
        detector=fake_detector,
        diarization_provider=provider,
    )

    assert provider.calls == 1
    assert summary["segments_written"] == 2
    assert summary["diarization_status_counts"]["diarization_no_speech"] == 1
    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["speaker_label"] for row in manifest_rows] == ["", ""]
    assert all(row["diarization_used"] is False for row in manifest_rows)
