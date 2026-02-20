from __future__ import annotations

import re

import numpy as np

from asr.models.load_indicconformer import MockIndicConformerModel
from asr.postprocess.script_fix import fix_script_confusion, fix_text
from asr.postprocess.transliteration import transliterate_telugu_to_devanagari
from asr.service.streaming_server import DecodeOptions, StreamingASREngine


DEV_RE = re.compile(r"[\u0900-\u097F]")
TEL_RE = re.compile(r"[\u0C00-\u0C7F]")


def test_transliteration_telugu_to_devanagari_changes_script() -> None:
    out = transliterate_telugu_to_devanagari("హలో")
    assert DEV_RE.search(out) is not None


def test_script_fix_applies_only_for_hi_tagged_telugu_token() -> None:
    tokens = ["తెలుగు", "హలో"]
    langs = ["te", "hi"]
    scripts = ["telugu", "telugu"]
    fixed, events = fix_script_confusion(tokens, langs, scripts)
    assert fixed[0] == "తెలుగు"
    assert DEV_RE.search(fixed[1]) is not None
    assert events[1].action == "telugu_to_devanagari"


def test_script_fix_does_not_modify_telugu_when_lang_is_te() -> None:
    tokens = ["తెలుగు", "హలో"]
    langs = ["te", "te"]
    scripts = ["telugu", "telugu"]
    fixed, events = fix_script_confusion(tokens, langs, scripts)
    assert fixed == tokens
    assert all(e.action == "keep" for e in events)


def test_fix_text_auto_heads_can_fix_known_hindi_word_in_telugu_script() -> None:
    out, events = fix_text("తెలుగు హలో")
    assert TEL_RE.search(out) is not None
    assert DEV_RE.search(out) is not None
    assert any(e.action == "telugu_to_devanagari" for e in events)


def test_end_to_end_mixed_language_output_contains_telugu_and_devanagari() -> None:
    model = MockIndicConformerModel(scripted_outputs=["తెలుగు హలో"])
    engine = StreamingASREngine(model=model, options=DecodeOptions(decode_mode="rnnt", use_lid_bias=False))

    # Synthetic short 8k telephony-like audio.
    sr = 8000
    t = np.arange(int(sr * 1.0), dtype=np.float32) / float(sr)
    audio = 0.2 * np.sin(2 * np.pi * 220.0 * t)

    out = engine.transcribe(audio, sample_rate=sr)
    text = out["text"]
    assert TEL_RE.search(text) is not None
    assert DEV_RE.search(text) is not None
