"""Hindi ID grammar for PAN and Aadhaar surfaces.

This module handles the transcription step only:

* PAN: five letters, four digits, one letter.
* Aadhaar: twelve digits, emitted in ``XXXX XXXX XXXX`` grouping.

Structural validation that is easier and safer in Python, such as Aadhaar's
Verhoeff checksum, remains in ``runtime.formatters.id_pan_aadhaar_ifsc``.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import DEVANAGARI_DIGITS, LATIN_DIGITS
from itn_service.grammars.hi.cardinal import _NUM_0_99, _SP


def _concat(parts: list[pynini.Fst]) -> pynini.Fst:
    if not parts:
        raise ValueError("cannot concatenate an empty FST list")
    out = parts[0]
    for part in parts[1:]:
        out = out + part
    return out.optimize()


_LETTER_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "A": ("ए",),
    "B": ("बी",),
    "C": ("सी",),
    "D": ("डी",),
    "E": ("ई",),
    "F": ("एफ",),
    "G": ("जी",),
    "H": ("एच",),
    "I": ("आई",),
    "J": ("जे",),
    "K": ("के",),
    "L": ("एल",),
    "M": ("एम",),
    "N": ("एन",),
    "O": ("ओ",),
    "P": ("पी",),
    "Q": ("क्यू",),
    "R": ("आर",),
    "S": ("एस",),
    "T": ("टी",),
    "U": ("यू",),
    "V": ("वी",),
    "W": ("डब्ल्यू", "डब्लू"),
    "X": ("एक्स",),
    "Y": ("वाई",),
    "Z": ("ज़ेड", "जेड"),
}


def _letter_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for codepoint in range(ord("A"), ord("Z") + 1):
        letter = chr(codepoint)
        pairs.append((letter, letter))
        pairs.append((letter.lower(), letter))
        for spoken in _LETTER_NAMES[letter]:
            pairs.append((spoken, letter))
    return pairs


_LETTER: Final[pynini.Fst] = pynini.string_map(_letter_pairs()).optimize()

_DIGIT: Final[pynini.Fst] = pynini.union(LATIN_DIGITS, DEVANAGARI_DIGITS).optimize()
_AADHAAR_FIRST_DIGIT: Final[pynini.Fst] = pynini.string_map(
    [
        *[(str(n), str(n)) for n in range(2, 10)],
        *[(chr(0x0966 + n), str(n)) for n in range(2, 10)],
    ]
).optimize()

_SPOKEN_DIGIT: Final[pynini.Fst] = pynini.string_map(
    [(word, str(n)) for n in range(10) for word in _NUM_0_99[n]]
    + [("जीरो", "0"), ("ज़ीरो", "0")]
).optimize()
_SPOKEN_AADHAAR_FIRST_DIGIT: Final[pynini.Fst] = pynini.string_map(
    [(word, str(n)) for n in range(2, 10) for word in _NUM_0_99[n]]
).optimize()

_VISUAL_SEP: Final[pynini.Fst] = pynini.union(" ", "-").closure(1).optimize()
_OPT_DEL_VISUAL_SEP: Final[pynini.Fst] = pynutil.delete(_VISUAL_SEP).ques


def _written_aadhaar_digits() -> pynini.Fst:
    parts: list[pynini.Fst] = [_AADHAAR_FIRST_DIGIT]
    for _ in range(11):
        parts.extend([_OPT_DEL_VISUAL_SEP, _DIGIT])
    return _concat(parts)


def _spoken_aadhaar_digits() -> pynini.Fst:
    parts: list[pynini.Fst] = [_SPOKEN_AADHAAR_FIRST_DIGIT]
    for _ in range(11):
        parts.extend([pynutil.delete(_SP), _SPOKEN_DIGIT])
    return _concat(parts)


AADHAAR_DIGITS: Final[pynini.Fst] = pynini.union(
    _written_aadhaar_digits(),
    _spoken_aadhaar_digits(),
).optimize()

_GROUPED_AADHAAR_DIGITS: Final[pynini.Fst] = _concat(
    [
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        pynutil.insert(" "),
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        pynutil.insert(" "),
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
    ]
)

AADHAAR: Final[pynini.Fst] = (AADHAAR_DIGITS @ _GROUPED_AADHAAR_DIGITS).optimize()


def _written_pan() -> pynini.Fst:
    return _concat(
        [_LETTER, _LETTER, _LETTER, _LETTER, _LETTER, _DIGIT, _DIGIT, _DIGIT, _DIGIT, _LETTER]
    )


def _spoken_pan() -> pynini.Fst:
    parts: list[pynini.Fst] = [_LETTER]
    for _ in range(4):
        parts.extend([pynutil.delete(_SP), _LETTER])
    for _ in range(4):
        parts.extend([pynutil.delete(_SP), _SPOKEN_DIGIT])
    parts.extend([pynutil.delete(_SP), _LETTER])
    return _concat(parts)


PAN: Final[pynini.Fst] = pynini.union(_written_pan(), _spoken_pan()).optimize()

ID: Final[pynini.Fst] = pynini.union(PAN, AADHAAR).optimize()

PAN_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('pan { value: "') + PAN + pynutil.insert('" }')
).optimize()
AADHAAR_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('aadhaar { value: "') + AADHAAR + pynutil.insert('" }')
).optimize()
ID_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('id { value: "') + ID + pynutil.insert('" }')
).optimize()


__all__ = [
    "AADHAAR",
    "AADHAAR_CLASSIFIER",
    "AADHAAR_DIGITS",
    "ID",
    "ID_CLASSIFIER",
    "PAN",
    "PAN_CLASSIFIER",
]
