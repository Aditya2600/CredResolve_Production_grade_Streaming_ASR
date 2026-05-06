#!/usr/bin/env python3
from __future__ import annotations

import re
import unicodedata


ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
TAG_RE = re.compile(r"<\s*/?\s*[^>\s]+[^>]*>")
BRACKETED_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")
BRACED_TEXT_RE = re.compile(r"\{([^{}]*)\}")
ANNOTATION_MARKER_RE = re.compile(r"(?:--+|\.{2,}|\u2026+)")
INDIC_PARTIAL_WORD_RE = re.compile(r"(?<!\S)\S*[\u094d\u09cd\u0a4d\u0acd\u0b4d\u0c4d\u0ccd\u0d4d]-+")
MULTISPACE_RE = re.compile(r"\s+")
UNINTELLIGIBLE_TAG_RE = re.compile(r"<\s*/?\s*(?:unintelligible|inaudible|unclear)\s*/?\s*>", re.IGNORECASE)
UNINTELLIGIBLE_ONLY_RE = re.compile(
    r"^\s*(?:<|\[|\()?[\s_-]*(?:unintelligible|inaudible|unclear)[\s_-]*(?:>|\]|\))?\s*$",
    re.IGNORECASE,
)
NUKTA_MARKS = frozenset(
    {
        "\u093c",  # Devanagari
        "\u09bc",  # Bengali
        "\u0a3c",  # Gurmukhi
        "\u0abc",  # Gujarati
        "\u0b3c",  # Odia
        "\u0c3c",  # Telugu
        "\u0cbc",  # Kannada
    }
)


def _normalize_space(value: str) -> str:
    return MULTISPACE_RE.sub(" ", value).strip()


def _replace_punctuation_and_symbols_with_space(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if unicodedata.category(char)[0] in {"P", "S"}:
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)


def _normalize_digits(value: str) -> str:
    chars: list[str] = []
    for char in value:
        try:
            chars.append(str(unicodedata.decimal(char)))
        except (TypeError, ValueError):
            chars.append(char)
    return "".join(chars)


def _normalize_nukta(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if char not in NUKTA_MARKS:
            chars.append(char)
            continue
        if not chars or chars[-1].isspace() or chars[-1] in NUKTA_MARKS:
            continue
        chars.append(char)
    return "".join(chars)


def _is_latin_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L") and "LATIN" in unicodedata.name(char, "")


def _is_latin_only_gloss(value: str) -> bool:
    letters = [char for char in value if unicodedata.category(char).startswith("L")]
    return bool(letters) and all(_is_latin_letter(char) for char in letters)


def _replace_braced_text(match: re.Match[str]) -> str:
    value = match.group(1).strip()
    if not value or _is_latin_only_gloss(value):
        return " "
    return f" {value} "


def _remove_latin_letters(value: str) -> str:
    return "".join(" " if _is_latin_letter(char) else char for char in value)


def normalize_asr_text(text: str, language: str | None = None) -> str:
    """Normalize a Vaani/Indic ASR target without transliteration."""
    del language
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = ZERO_WIDTH_RE.sub("", value)
    value = _normalize_digits(value)
    value = _normalize_nukta(value)
    if UNINTELLIGIBLE_ONLY_RE.match(value) or UNINTELLIGIBLE_TAG_RE.search(value):
        return ""

    value = TAG_RE.sub(" ", value)
    value = BRACKETED_ANNOTATION_RE.sub(" ", value)
    value = INDIC_PARTIAL_WORD_RE.sub(" ", value)

    previous = None
    while previous != value:
        previous = value
        value = BRACED_TEXT_RE.sub(_replace_braced_text, value)

    value = _remove_latin_letters(value)
    value = ANNOTATION_MARKER_RE.sub(" ", value)
    value = _replace_punctuation_and_symbols_with_space(value)
    value = _normalize_nukta(value)
    return _normalize_space(value)
