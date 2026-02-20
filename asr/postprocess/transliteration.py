from __future__ import annotations

import re


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")


# Minimal mapping sufficient for common Hindi words typed in Telugu script.
TELUGU_TO_DEVANAGARI = {
    "అ": "अ",
    "ఆ": "आ",
    "ఇ": "इ",
    "ఈ": "ई",
    "ఉ": "उ",
    "ఊ": "ऊ",
    "ఎ": "ए",
    "ఏ": "ए",
    "ఐ": "ऐ",
    "ఒ": "ओ",
    "ఓ": "ओ",
    "ఔ": "औ",
    "క": "क",
    "ఖ": "ख",
    "గ": "ग",
    "ఘ": "घ",
    "చ": "च",
    "జ": "ज",
    "ట": "ट",
    "డ": "ड",
    "త": "त",
    "ద": "द",
    "న": "न",
    "ప": "प",
    "బ": "ब",
    "మ": "म",
    "య": "य",
    "ర": "र",
    "ల": "ल",
    "వ": "व",
    "శ": "श",
    "స": "स",
    "హ": "ह",
    "ా": "ा",
    "ి": "ि",
    "ీ": "ी",
    "ు": "ु",
    "ూ": "ू",
    "ె": "े",
    "ే": "े",
    "ై": "ै",
    "ొ": "ो",
    "ో": "ो",
    "ౌ": "ौ",
    "్": "्",
    "ం": "ं",
    "ః": "ः",
}


def detect_script(token: str) -> str:
    if TELUGU_RE.search(token):
        return "telugu"
    if DEVANAGARI_RE.search(token):
        return "devanagari"
    if any("a" <= ch.lower() <= "z" for ch in token):
        return "latin"
    return "other"


def transliterate_telugu_to_devanagari(token: str) -> str:
    # Try external transliteration libs first when available.
    try:
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

        out = UnicodeIndicTransliterator.transliterate(token, "te", "hi")
        if out:
            return str(out)
    except Exception:
        pass

    try:
        from ai4bharat.transliteration import XlitEngine

        engine = XlitEngine(src_script_type="native", beam_width=1)
        out = engine.translit_sentence(token, lang_code="hi")
        if out:
            return str(out)
    except Exception:
        pass

    return "".join(TELUGU_TO_DEVANAGARI.get(ch, ch) for ch in token)
