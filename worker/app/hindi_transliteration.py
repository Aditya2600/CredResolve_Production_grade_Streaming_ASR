from __future__ import annotations

import re

LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+")
UPPERCASE_ACRONYM_RE = re.compile(r"[A-Z]{2,8}")

INDEPENDENT_VOWELS = {
    "a": "अ",
    "aa": "आ",
    "ai": "ऐ",
    "au": "औ",
    "e": "ए",
    "ee": "ई",
    "i": "इ",
    "ii": "ई",
    "o": "ओ",
    "oa": "ओ",
    "oo": "ऊ",
    "ou": "औ",
    "u": "उ",
    "uu": "ऊ",
}
DEPENDENT_VOWELS = {
    "a": "",
    "aa": "ा",
    "ai": "ै",
    "au": "ौ",
    "e": "े",
    "ee": "ी",
    "i": "ि",
    "ii": "ी",
    "o": "ो",
    "oa": "ो",
    "oo": "ू",
    "ou": "ौ",
    "u": "ु",
    "uu": "ू",
}
VOWEL_PATTERNS = tuple(
    sorted(
        INDEPENDENT_VOWELS.keys(),
        key=len,
        reverse=True,
    )
)
CONSONANTS = {
    "b": "ब",
    "bh": "भ",
    "c": "क",
    "ch": "च",
    "chh": "छ",
    "d": "द",
    "dh": "ध",
    "f": "फ",
    "g": "ग",
    "gh": "घ",
    "h": "ह",
    "j": "ज",
    "jh": "झ",
    "k": "क",
    "kh": "ख",
    "ksh": "क्ष",
    "l": "ल",
    "m": "म",
    "n": "न",
    "p": "प",
    "ph": "फ",
    "q": "क",
    "r": "र",
    "s": "स",
    "sh": "श",
    "t": "त",
    "th": "थ",
    "v": "व",
    "w": "व",
    "x": "क्स",
    "y": "य",
    "z": "ज",
}
CONSONANT_PATTERNS = tuple(sorted(CONSONANTS.keys(), key=len, reverse=True))
LETTER_NAMES = {
    "A": "ए",
    "B": "बी",
    "C": "सी",
    "D": "डी",
    "E": "ई",
    "F": "एफ",
    "G": "जी",
    "H": "एच",
    "I": "आई",
    "J": "जे",
    "K": "के",
    "L": "एल",
    "M": "एम",
    "N": "एन",
    "O": "ओ",
    "P": "पी",
    "Q": "क्यू",
    "R": "आर",
    "S": "एस",
    "T": "टी",
    "U": "यू",
    "V": "वी",
    "W": "डब्ल्यू",
    "X": "एक्स",
    "Y": "वाई",
    "Z": "जेड",
}
DEVANAGARI_DIGIT_TRANSLATION = str.maketrans("0123456789", "०१२३४५६७८९")
HINDI_MONTH_VARIANTS = {
    1: ("जनवरी",),
    2: ("फरवरी", "फ़रवरी"),
    3: ("मार्च",),
    4: ("अप्रैल",),
    5: ("मई",),
    6: ("जून",),
    7: ("जुलाई",),
    8: ("अगस्त",),
    9: ("सितंबर", "सितम्बर"),
    10: ("अक्टूबर",),
    11: ("नवंबर", "नवम्बर"),
    12: ("दिसंबर", "दिसम्बर"),
}


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    variants: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        variants.append(cleaned)
    return tuple(variants)


def _match_pattern(text: str, index: int, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        if text.startswith(pattern, index):
            return pattern
    return None


def _tokenize(token: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    index = 0
    while index < len(token):
        vowel = _match_pattern(token, index, VOWEL_PATTERNS)
        if vowel is not None:
            units.append(("vowel", vowel))
            index += len(vowel)
            continue

        consonant = _match_pattern(token, index, CONSONANT_PATTERNS)
        if consonant is not None:
            units.append(("consonant", consonant))
            index += len(consonant)
            continue

        index += 1
    return units


def _render_units(units: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    index = 0
    while index < len(units):
        kind, value = units[index]
        if kind == "vowel":
            parts.append(INDEPENDENT_VOWELS[value])
            index += 1
            continue

        consonant = CONSONANTS[value]
        next_kind = units[index + 1][0] if index + 1 < len(units) else None
        next_value = units[index + 1][1] if index + 1 < len(units) else None
        if next_kind == "vowel" and next_value is not None:
            parts.append(consonant + DEPENDENT_VOWELS[next_value])
            index += 2
            continue
        if next_kind == "consonant":
            parts.append(consonant + "्")
        else:
            parts.append(consonant)
        index += 1
    return "".join(parts)


def _transliterate_acronym(token: str) -> tuple[str, ...]:
    letter_names = [LETTER_NAMES.get(letter, "") for letter in token]
    letter_names = [value for value in letter_names if value]
    if not letter_names:
        return tuple()
    return _dedupe(["".join(letter_names), " ".join(letter_names)])


def _transliterate_token_variants(token: str) -> tuple[str, ...]:
    if not token:
        return tuple()
    if UPPERCASE_ACRONYM_RE.fullmatch(token):
        return _transliterate_acronym(token)

    lowered = token.lower()
    units = _tokenize(lowered)
    if not units:
        return tuple()

    rendered = _render_units(units)
    variants = [rendered]

    if lowered.startswith("a") and len(lowered) >= 5 and rendered.startswith("अ"):
        variants.append(f"आ{rendered[1:]}")
    if lowered.endswith("iya") and rendered.endswith("य"):
        variants.append(f"{rendered}ा")
    if lowered.endswith("umar") and rendered.endswith("मर"):
        variants.append(f"{rendered[:-2]}मार")
    if lowered.endswith("rma") and rendered.endswith("र्म"):
        variants.append(f"{rendered}ा")

    return _dedupe(variants)


def generate_hindi_script_variants(text: str, *, max_variants: int = 4) -> tuple[str, ...]:
    if not LATIN_TOKEN_RE.search(text or ""):
        return tuple()

    parts: list[tuple[str, tuple[str, ...]]] = []
    index = 0
    for match in LATIN_TOKEN_RE.finditer(text):
        if match.start() > index:
            parts.append(("literal", (text[index : match.start()],)))
        parts.append(("token", _transliterate_token_variants(match.group(0)) or (match.group(0),)))
        index = match.end()
    if index < len(text):
        parts.append(("literal", (text[index:],)))

    variants = [""]
    for _kind, segment_values in parts:
        next_variants: list[str] = []
        for prefix in variants:
            for segment_value in segment_values:
                candidate = f"{prefix}{segment_value}"
                if candidate not in next_variants:
                    next_variants.append(candidate)
                if len(next_variants) >= max_variants:
                    break
            if len(next_variants) >= max_variants:
                break
        variants = next_variants
        if not variants:
            break

    return tuple(candidate for candidate in _dedupe(variants) if candidate != (text or "").strip())


def to_devanagari_digits(text: str) -> str:
    return (text or "").translate(DEVANAGARI_DIGIT_TRANSLATION)


def get_hindi_month_variants(month: int) -> tuple[str, ...]:
    return HINDI_MONTH_VARIANTS.get(month, tuple())
