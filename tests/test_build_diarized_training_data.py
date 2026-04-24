from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.build_diarized_training_data import build_segmenter_args, filter_inventory_for_training, main


def test_filter_inventory_keeps_only_successful_diarized_rows_with_turns() -> None:
    frame = pd.DataFrame(
        [
            {"call_id": "call-001", "diarization_status": "diarized_nemo_telephony"},
            {"call_id": "call-002", "diarization_status": "diarization_failed"},
            {"call_id": "call-003", "diarization_status": "diarized_nemo_telephony"},
        ]
    )

    filtered, stats = filter_inventory_for_training(
        frame,
        require_diarized_status=True,
        require_turn_coverage=True,
        call_ids_with_turns={"call-001"},
    )

    assert list(filtered["call_id"]) == ["call-001"]
    assert stats == {
        "rows_input": 3,
        "rows_after_diarized_status_filter": 2,
        "rows_after_turn_coverage_filter": 1,
        "rows_excluded_by_diarized_status": 1,
        "rows_excluded_by_turn_coverage": 1,
        "diarized_status_filter": {
            "enabled": True,
            "kept_status": "diarized_nemo_telephony",
            "rows_before": 3,
            "rows_after": 2,
            "rows_excluded": 1,
            "excluded_status_counts": {"diarization_failed": 1},
        },
        "turn_coverage_filter": {
            "enabled": True,
            "rows_before": 2,
            "rows_after": 1,
            "rows_excluded": 1,
            "excluded_missing_call_id": 0,
            "excluded_missing_turns_for_call_id": 1,
            "call_ids_with_turns_count": 1,
        },
    }


def test_build_segmenter_args_reuses_external_diarization_turns(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.csv"
    turns_path = tmp_path / "diarization_turns.jsonl"
    output_dir = tmp_path / "segmented"
    inventory_path.write_text("call_id\ncall-001\n", encoding="utf-8")
    turns_path.write_text("", encoding="utf-8")

    class _Args:
        audio_mode = "auto"
        text_column = "raw_transcript"
        language_column = "language"
        language = None
        respect_ready_status = True
        overwrite = False
        output_sample_rate = 8000
        vad_resample_rate = 16000
        threshold = 0.5
        min_speech_duration_ms = 250
        max_speech_duration_s = 20.0
        min_silence_duration_ms = 150
        speech_pad_ms = 60
        diarization_min_segment_sec = 0.5
        progress_every = 25

    namespace = build_segmenter_args(
        filtered_inventory_path=inventory_path,
        output_dir=output_dir,
        diarization_turns_path=turns_path,
        wrapper_args=_Args(),
    )

    assert namespace.inventory == inventory_path
    assert namespace.output_dir == output_dir
    assert namespace.diarization_turns == turns_path
    assert namespace.enable_diarization is False
    assert namespace.text_column == "raw_transcript"
    assert namespace.language_column == "language"


def test_main_filters_inventory_and_invokes_segmenter(tmp_path: Path, monkeypatch) -> None:
    diarization_dir = tmp_path / "diarization"
    diarization_dir.mkdir()
    output_dir = tmp_path / "training"

    inventory_path = diarization_dir / "data_inventory_diarized.csv"
    inventory = pd.DataFrame(
        [
            {
                "call_id": "call-001",
                "row_id": "1",
                "language": "HINDI",
                "raw_transcript": json.dumps({"interaction_transcript": [{"role": "agent", "en_text": "hello"}]}),
                "segmentation_status": "ready_mono_segmentation",
                "diarization_status": "diarized_nemo_telephony",
            },
            {
                "call_id": "call-002",
                "row_id": "2",
                "language": "HINDI",
                "raw_transcript": json.dumps({"interaction_transcript": [{"role": "agent", "en_text": "bye"}]}),
                "segmentation_status": "ready_mono_segmentation",
                "diarization_status": "diarization_failed",
            },
        ]
    )
    inventory.to_csv(inventory_path, index=False)

    turns_path = diarization_dir / "diarization_turns.jsonl"
    turns_path.write_text(
        json.dumps(
            {
                "call_id": "call-001",
                "speaker_label": "speaker_0",
                "start_sec": 0.0,
                "end_sec": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    (diarization_dir / "summary.json").write_text(
        json.dumps({"inventory_csv_path": str(inventory_path)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_segment_inventory(args):
        captured["args"] = args
        return {
            "manifest_path": str(output_dir / "manifest.jsonl"),
            "segments_written": 1,
            "diarization_turns_path": str(turns_path),
        }

    monkeypatch.setattr(
        "tools.build_diarized_training_data.segment_inventory_process_inventory",
        fake_segment_inventory,
    )

    exit_code = main(
        [
            "--diarization-dir",
            str(diarization_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    filtered_inventory = pd.read_csv(output_dir / "input_inventory_diarized_for_training.csv", dtype=str)
    assert list(filtered_inventory["call_id"]) == ["call-001"]

    segment_args = captured["args"]
    assert segment_args.inventory == output_dir / "input_inventory_diarized_for_training.csv"
    assert segment_args.diarization_turns == turns_path.resolve()

    orchestration_summary = json.loads((output_dir / "orchestration_summary.json").read_text(encoding="utf-8"))
    assert orchestration_summary["rows_input"] == 2
    assert orchestration_summary["rows_after_diarized_status_filter"] == 1
    assert orchestration_summary["rows_after_turn_coverage_filter"] == 1
    assert orchestration_summary["rows_excluded_by_diarized_status"] == 1
    assert orchestration_summary["rows_excluded_by_turn_coverage"] == 0
    assert orchestration_summary["diarized_status_filter"] == {
        "enabled": True,
        "kept_status": "diarized_nemo_telephony",
        "rows_before": 2,
        "rows_after": 1,
        "rows_excluded": 1,
        "excluded_status_counts": {"diarization_failed": 1},
    }
    assert orchestration_summary["turn_coverage_filter"] == {
        "enabled": True,
        "rows_before": 1,
        "rows_after": 1,
        "rows_excluded": 0,
        "excluded_missing_call_id": 0,
        "excluded_missing_turns_for_call_id": 0,
        "call_ids_with_turns_count": 1,
    }
