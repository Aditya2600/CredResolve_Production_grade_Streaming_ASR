from __future__ import annotations

from tools.diarization.merge import annotate_interval_with_speaker, speakerize_words
from tools.diarization.providers.base import SpeakerTurn


def test_annotate_interval_returns_empty_for_silence_only_audio():
    annotation = annotate_interval_with_speaker([], start_sec=0.0, end_sec=1.0)

    assert annotation == {
        "speaker_label": "",
        "speaker_confidence": None,
        "overlap_flag": False,
    }


def test_annotate_interval_handles_one_speaker_audio():
    turns = [
        SpeakerTurn(speaker_label="speaker_0", start_sec=0.0, end_sec=1.5, confidence=1.0),
    ]

    annotation = annotate_interval_with_speaker(turns, start_sec=0.2, end_sec=1.0)

    assert annotation["speaker_label"] == "speaker_0"
    assert annotation["speaker_confidence"] == 1.0
    assert annotation["overlap_flag"] is False


def test_annotate_interval_prefers_majority_speaker_with_short_backchannel():
    turns = [
        SpeakerTurn(speaker_label="speaker_0", start_sec=0.0, end_sec=2.0),
        SpeakerTurn(speaker_label="speaker_1", start_sec=0.95, end_sec=1.1),
    ]

    annotation = annotate_interval_with_speaker(turns, start_sec=0.8, end_sec=1.3)

    assert annotation["speaker_label"] == "speaker_0"
    assert annotation["overlap_flag"] is True


def test_speakerize_words_marks_overlap_for_interruption_case():
    turns = [
        SpeakerTurn(speaker_label="speaker_0", start_sec=0.0, end_sec=1.0),
        SpeakerTurn(speaker_label="speaker_1", start_sec=0.9, end_sec=1.5),
    ]
    words = [
        {"word": "haan", "start_sec": 0.95, "end_sec": 1.05, "word_index": 0},
    ]

    speakerized = speakerize_words(words, turns)

    assert speakerized == [
        {
            "word": "haan",
            "start_sec": 0.95,
            "end_sec": 1.05,
            "word_index": 0,
            "speaker_label": "speaker_1",
            "speaker_confidence": 1.0,
            "overlap_flag": True,
        }
    ]
