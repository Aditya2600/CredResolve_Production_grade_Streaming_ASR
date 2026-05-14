"""Unit tests for `runtime.script_router`.

Covers `detect_script` over all ten supported scripts, the Common /
empty edge cases, and Hinglish / mixed-script behaviour. Also covers
the priority cascade in `route_language` and the IndicLID stub.
"""

from __future__ import annotations

import pytest

from itn_service.runtime.script_router import (
    RouteResult,
    detect_script,
    indiclid_predict,
    route_language,
)


# --- detect_script: the ten scripts ------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # Devanagari (Hindi, Marathi)
        ("नमस्ते", "Devanagari"),
        ("एक हजार रुपये", "Devanagari"),
        ("मराठी भाषा", "Devanagari"),
        # Bengali
        ("নমস্কার", "Bengali"),
        ("এক হাজার", "Bengali"),
        # Gurmukhi (Punjabi)
        ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "Gurmukhi"),
        ("ਇੱਕ ਹਜ਼ਾਰ", "Gurmukhi"),
        # Gujarati
        ("નમસ્તે", "Gujarati"),
        ("એક હજાર", "Gujarati"),
        # Tamil
        ("வணக்கம்", "Tamil"),
        ("ஆயிரம்", "Tamil"),
        # Telugu
        ("నమస్కారం", "Telugu"),
        ("వెయ్యి", "Telugu"),
        # Kannada
        ("ನಮಸ್ಕಾರ", "Kannada"),
        ("ಒಂದು ಸಾವಿರ", "Kannada"),
        # Malayalam
        ("നമസ്കാരം", "Malayalam"),
        ("ആയിരം", "Malayalam"),
        # Arabic (Urdu)
        ("السلام علیکم", "Arabic"),
        ("ایک ہزار", "Arabic"),
        # Latin
        ("Hello world", "Latin"),
        ("one thousand rupees", "Latin"),
    ],
)
def test_detect_script_for_supported_scripts(text: str, expected: str) -> None:
    assert detect_script(text) == expected


# --- detect_script: edge cases -----------------------------------------------


def test_empty_string_is_common() -> None:
    assert detect_script("") == "Common"


def test_pure_digits_is_common() -> None:
    assert detect_script("9876543210") == "Common"


def test_pure_punctuation_is_common() -> None:
    assert detect_script("--- !!! ???") == "Common"


def test_only_whitespace_is_common() -> None:
    assert detect_script("   \t\n  ") == "Common"


def test_devanagari_with_punctuation_still_devanagari() -> None:
    assert detect_script("नमस्ते, कैसे हैं?") == "Devanagari"


def test_devanagari_with_digits_still_devanagari() -> None:
    assert detect_script("मेरे पास 100 रुपये हैं") == "Devanagari"


# --- detect_script: mixed / Hinglish -----------------------------------------


def test_majority_devanagari_with_a_few_latin_letters() -> None:
    # 12 Devanagari chars vs 2 Latin chars -> Devanagari wins.
    assert detect_script("नमस्ते दुनिया OK") == "Devanagari"


def test_majority_latin_with_one_devanagari_char() -> None:
    assert detect_script("Hello there अ") == "Latin"


def test_hinglish_with_more_latin_than_devanagari() -> None:
    # Romanised Hindi with a few native characters dropped in.
    assert detect_script("Aap kaise ho? मैं ठीक") == "Latin"


def test_tie_between_indic_and_latin_prefers_indic() -> None:
    # Two Devanagari letters, two Latin letters -> Indic wins per the
    # tie-break policy documented in script_router.
    assert detect_script("ab नम") == "Devanagari"


def test_bengali_dominant_in_mixed_with_latin() -> None:
    assert detect_script("Bonjour নমস্কার বন্ধু") == "Bengali"


# --- route_language: priority cascade ----------------------------------------


def test_asr_hint_wins_over_script() -> None:
    # ASR says Marathi, script is Devanagari. Marathi is a trusted hint
    # and wins over the default 'hi' that Devanagari maps to.
    res = route_language("नमस्कार", asr_hint="mr")
    assert isinstance(res, RouteResult)
    assert res.lang == "mr"
    assert res.script == "Devanagari"
    assert res.source == "asr_hint"
    assert res.needs_indiclid is False


def test_untrusted_asr_hint_ignored() -> None:
    res = route_language("नमस्ते", asr_hint="zz")
    assert res.lang == "hi"
    assert res.source == "script_majority"


def test_script_majority_devanagari_defaults_to_hindi() -> None:
    res = route_language("एक हजार रुपये")
    assert res.lang == "hi"
    assert res.script == "Devanagari"
    assert res.source == "script_majority"
    assert res.needs_indiclid is False


# --- Marathi vs Hindi disambiguation on Devanagari ---------------------------
#
# Both languages share the Devanagari script. The router upgrades to
# Marathi only when distinctive Marathi-only lexical cues are present.


@pytest.mark.parametrize(
    "text",
    [
        # Time cues — the strongest Marathi signals.
        "पाच वाजता",
        "साडे पाच वाजता",
        "पाच वाजून तीस मिनिटे",
        "सकाळी सात वाजता",
        "संध्याकाळी साडे पाच वाजता",
        "रात्री दहा वाजता",
        # Hundred compounds — Marathi-only structural shape.
        "पाचशे रुपये",
        "एकशे पंचवीस",
        "दोनशे टक्के",
        # Half/quarter compounds.
        "दीड हजार रुपये",
        "अडीच लाख",
        "पावणे पाच हजार",
        # Percent cue.
        "बारा टक्के",
        "पंचवीस टक्के झाले",
        # Month names.
        "एक जानेवारी दोन हजार पंचवीस",
        "पंधरा ऑगस्ट दोन हजार चोवीस",
        "बारा डिसेंबर",
    ],
)
def test_devanagari_with_marathi_cues_routes_to_mr(text: str) -> None:
    res = route_language(text)
    assert res.lang == "mr", f"{text!r} should route to mr, got {res.lang!r}"
    assert res.script == "Devanagari"
    assert res.source == "script_majority_mr_keywords"


@pytest.mark.parametrize(
    "text",
    [
        # No distinctive cues — defaults to Hindi.
        "नमस्ते",
        "एक हजार",
        "एक हजार रुपये",
        "मेरे पास सौ रुपये हैं",
        # Hindi-specific cues — must stay Hindi.
        "पाँच बजे",
        "पाँच बजकर तीस मिनट",
        "डेढ़ हज़ार रुपये",
        "ढाई लाख",
        "पौने पाँच हज़ार",
        "बारह प्रतिशत",
        "बारह फीसदी",
        "एक करोड़",
        "पंद्रह अगस्त",
        "बारह मई दो हज़ार छब्बीस",
    ],
)
def test_devanagari_with_hindi_cues_or_no_cues_routes_to_hi(text: str) -> None:
    res = route_language(text)
    assert res.lang == "hi", f"{text!r} should route to hi, got {res.lang!r}"
    assert res.script == "Devanagari"


def test_asr_hint_overrides_keyword_score() -> None:
    """Marathi cues in the text must NOT override an explicit hi ASR hint."""
    res = route_language("पाच वाजता", asr_hint="hi")
    assert res.lang == "hi"
    assert res.source == "asr_hint"


def test_marathi_wins_when_evidence_outweighs_hindi() -> None:
    """Direction-of-tiebreak test: code-switched Devanagari with more
    Marathi-only cues than Hindi-only cues flips to mr. The default is
    Hindi, but a positive Marathi margin overrides it."""
    text = "बारह बजे आणि पाच वाजून दहा मिनिटे"
    res = route_language(text)
    assert res.lang == "mr"


def test_lone_hindi_cue_keeps_default() -> None:
    """One Hindi cue with no Marathi cues stays Hindi (the default)."""
    res = route_language("एक बजे")
    assert res.lang == "hi"
    assert res.source == "script_majority"


def test_script_majority_bengali_to_bn() -> None:
    assert route_language("এক হাজার").lang == "bn"


def test_script_majority_gurmukhi_to_pa() -> None:
    assert route_language("ਇੱਕ ਹਜ਼ਾਰ").lang == "pa"


def test_script_majority_arabic_to_ur() -> None:
    assert route_language("ایک ہزار").lang == "ur"


def test_latin_without_romanized_hint_flags_indiclid() -> None:
    res = route_language("Hello aap kaise ho")
    assert res.script == "Latin"
    assert res.lang == "en"
    assert res.needs_indiclid is True


def test_latin_with_romanized_hint_uses_hint() -> None:
    res = route_language("aap kaise ho", romanized_hint="hi")
    assert res.lang == "hi"
    assert res.source == "romanized_hint"
    assert res.needs_indiclid is False


def test_pure_digits_falls_through_to_und() -> None:
    res = route_language("12345")
    assert res.script == "Common"
    assert res.lang == "und"
    assert res.needs_indiclid is True


def test_empty_text_falls_through_to_und() -> None:
    res = route_language("")
    assert res.lang == "und"
    assert res.needs_indiclid is True


# --- IndicLID stub -----------------------------------------------------------


def test_indiclid_predict_is_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        indiclid_predict("aap kaise ho")
