"""Conservative Hindi spoken-form prefilter.

The regex prefilter locates already-written semiotic forms.  This module adds
the complementary Hindi-only bridge for *spoken* surfaces that the compiled
WFST grammars already understand:

    एक सौ पच्चीस रुपये
    बारह मई दो हजार छब्बीस
    शाम पाँच बजे
    बारह दशमलव पाँच प्रतिशत

It deliberately does **not** normalise anything.  Every returned span keeps
``canonical == raw`` and carries offsets into the input text; downstream WFSTs
plus the confidence gate remain the only rewrite authority.

Design constraints:

* specific cue-bearing classes win over generic number spans;
* phone candidates require a nearby phone/mobile/number cue **and** exactly
  ten spoken digit words;
* generic cardinal / decimal spans are bounded so arbitrary Hindi text is not
  swallowed into a candidate;
* request-path imports stay lightweight — this module does not import grammar
  source files or build FST graphs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .contract import Span


# ---------------------------------------------------------------------------
# Lightweight lexical inventory.
#
# Kept local on purpose: importing ``grammars.hi.cardinal`` here would build
# Pynini graphs in a request-path module, violating the runtime contract.
# These are the spellings already supported by the Hindi grammars plus a few
# cue-only romanised words whose unsupported combinations must safely fall back
# through the existing WFST path.
# ---------------------------------------------------------------------------

_NUMBER_WORDS: frozenset[str] = frozenset(
    {
        "शून्य",
        "एक",
        "दो",
        "तीन",
        "चार",
        "पाँच",
        "पांच",
        "छह",
        "छः",
        "सात",
        "आठ",
        "नौ",
        "दस",
        "ग्यारह",
        "बारह",
        "तेरह",
        "चौदह",
        "पंद्रह",
        "पन्द्रह",
        "सोलह",
        "सत्रह",
        "अट्ठारह",
        "अठारह",
        "उन्नीस",
        "बीस",
        "इक्कीस",
        "बाईस",
        "तेईस",
        "चौबीस",
        "पच्चीस",
        "छब्बीस",
        "सत्ताईस",
        "अट्ठाईस",
        "उनतीस",
        "तीस",
        "इकतीस",
        "बत्तीस",
        "तैंतीस",
        "चौंतीस",
        "पैंतीस",
        "छत्तीस",
        "सैंतीस",
        "अड़तीस",
        "उनतालीस",
        "चालीस",
        "इकतालीस",
        "बयालीस",
        "तैंतालीस",
        "चौंतालीस",
        "पैंतालीस",
        "छियालीस",
        "सैंतालीस",
        "अड़तालीस",
        "उनचास",
        "पचास",
        "इक्यावन",
        "बावन",
        "तिरेपन",
        "चौवन",
        "पचपन",
        "छप्पन",
        "सत्तावन",
        "अट्ठावन",
        "उनसठ",
        "साठ",
        "इकसठ",
        "बासठ",
        "तिरेसठ",
        "चौंसठ",
        "पैंसठ",
        "छयासठ",
        "सरसठ",
        "अड़सठ",
        "उनहत्तर",
        "सत्तर",
        "इकहत्तर",
        "बहत्तर",
        "तिहत्तर",
        "चौहत्तर",
        "पचहत्तर",
        "छिहत्तर",
        "सतहत्तर",
        "अठहत्तर",
        "उन्यासी",
        "अस्सी",
        "इक्यासी",
        "बयासी",
        "तिरासी",
        "चौरासी",
        "पचासी",
        "छियासी",
        "सत्तासी",
        "अठासी",
        "नवासी",
        "नब्बे",
        "इक्यानवे",
        "बानवे",
        "तिरानवे",
        "चौरानवे",
        "पचानवे",
        "छियानवे",
        "सत्तानवे",
        "अट्ठानवे",
        "निन्यानवे",
    }
)

_SCALE_WORDS: frozenset[str] = frozenset(
    {"सौ", "हज़ार", "हजार", "लाख", "करोड़", "करोड", "अरब"}
)
_CODE_SWITCH_NUMBER_WORDS: frozenset[str] = frozenset(
    {
        "वन",
        "वान",
        "टू",
        "टु",
        "थ्री",
        "फोर",
        "फॉर",
        "फाइव",
        "फाईव",
        "सिक्स",
        "सेवन",
        "एट",
        "ऐट",
        "नाइन",
        "टेन",
        "ट्वेंटी",
        "ट्वेन्टी",
        "थर्टी",
        "थर्टि",
        "फोर्टी",
        "फॉर्टी",
        "फिफ्टी",
        "फिफ्टि",
        "फिफटि",
        "सिक्स्टी",
        "सिक्स्टि",
        "सिक्सटि",
        "सेवेंटी",
        "सेवन्टी",
        "सेवंटी",
        "सेवनटी",
        "एटी",
        "ऐटी",
        "एइटी",
        "नाइंटी",
        "नाइन्टी",
        "नाइंटि",
        "नाइन्टि",
    }
)
_CODE_SWITCH_TENS_WORDS: frozenset[str] = frozenset(
    {
        "ट्वेंटी",
        "ट्वेन्टी",
        "थर्टी",
        "थर्टि",
        "फोर्टी",
        "फॉर्टी",
        "फिफ्टी",
        "फिफ्टि",
        "फिफटि",
        "सिक्स्टी",
        "सिक्स्टि",
        "सिक्सटि",
        "सेवेंटी",
        "सेवन्टी",
        "सेवंटी",
        "सेवनटी",
        "एटी",
        "ऐटी",
        "एइटी",
        "नाइंटी",
        "नाइन्टी",
        "नाइंटि",
        "नाइन्टि",
    }
)
_CODE_SWITCH_SCALE_WORDS: frozenset[str] = frozenset(
    {
        "हंड्रेड",
        "हन्ड्रेड",
        "थाउजेंड",
        "थाउजंड",
        "थाउज़ेंड",
        "थाउज़ंड",
    }
)
_COMPOUND_WORDS: frozenset[str] = frozenset(
    {"सवा", "डेढ़", "डेढ", "ढाई", "साढ़े", "साढे", "पौने"}
)
_DECIMAL_MARKERS: frozenset[str] = frozenset({"दशमलव"})
_DIGIT_WORDS: frozenset[str] = frozenset(
    {
        "शून्य",
        "जीरो",
        "ज़ीरो",
        "एक",
        "दो",
        "तीन",
        "चार",
        "पाँच",
        "पांच",
        "छह",
        "छः",
        "सात",
        "आठ",
        "नौ",
    }
)

_RUPEE_CUES: frozenset[str] = frozenset(
    {"रुपये", "रुपया", "रू", "rupee", "rupees", "₹"}
)
_PAISE_CUES: frozenset[str] = frozenset({"पैसे", "पैसा"})
_PERCENT_CUES: frozenset[str] = frozenset(
    {"प्रतिशत", "फीसदी", "फ़ीसदी", "percent", "%"}
)
_TIME_OF_DAY_CUES: frozenset[str] = frozenset(
    {"सुबह", "दोपहर", "शाम", "रात", "am", "pm", "a.m.", "p.m."}
)
_PHONE_CUES: frozenset[str] = frozenset(
    {
        "phone",
        "mobile",
        "number",
        "otp",
        "फ़ोन",
        "फोन",
        "मोबाइल",
        "नंबर",
        "नम्बर",
        "ओटीपी",
    }
)
_AADHAAR_CUES: frozenset[str] = frozenset(
    {
        "aadhaar",
        "aadhar",
        "uid",
        "uidai",
        "आधार",
        "आधारकार्ड",
        "यूआईडी",
    }
)
_PAN_CUES: frozenset[str] = frozenset(
    {"pan", "पैन", "पेन"}
)
_ID_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "card",
        "number",
        "no",
        "is",
        "कार्ड",
        "नंबर",
        "नम्बर",
        "संख्या",
        "है",
    }
)
_LETTER_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "ए",
        "बी",
        "सी",
        "डी",
        "ई",
        "एफ",
        "जी",
        "एच",
        "आई",
        "जे",
        "के",
        "एल",
        "एम",
        "एन",
        "ओ",
        "पी",
        "क्यू",
        "आर",
        "एस",
        "टी",
        "यू",
        "वी",
        "डब्ल्यू",
        "डब्लू",
        "एक्स",
        "वाई",
        "ज़ेड",
        "जेड",
    }
)
_MONTH_WORDS: frozenset[str] = frozenset(
    {
        "जनवरी",
        "फरवरी",
        "फ़रवरी",
        "मार्च",
        "अप्रैल",
        "मई",
        "जून",
        "जुलाई",
        "अगस्त",
        "सितंबर",
        "सितम्बर",
        "अक्टूबर",
        "नवंबर",
        "नवम्बर",
        "दिसंबर",
        "दिसम्बर",
    }
)
_STRONG_DATE_CUES: frozenset[str] = frozenset(
    {"तारीख", "तारीख़", "दिनांक", "date"}
)

_TOKEN_RE = re.compile(r"[^\s,.;:!?()\[\]\"'।]+")

_MAX_NUMBER_TOKENS = 8
_MAX_DECIMAL_TOKENS = 8
_PHONE_CUE_WINDOW = 4
_ID_CUE_WINDOW = 4


@dataclass(frozen=True)
class _Token:
    raw: str
    norm: str
    start: int
    end: int


def prefilter(text: str, lang: str) -> list[Span]:
    """Return conservative Hindi spoken candidates.

    ``lang`` is explicit so callers can preserve the current stub-language
    boundary.  Non-Hindi text returns no spoken spans at all.
    """
    if lang != "hi" or not text:
        return []

    tokens = _tokenise(text)
    if not tokens:
        return []

    candidates: list[Span] = []
    candidates.extend(_detect_money(text, tokens))
    candidates.extend(_detect_percent(text, tokens))
    candidates.extend(_detect_time(text, tokens))
    candidates.extend(_detect_date(text, tokens))
    candidates.extend(_detect_ids(text, tokens))
    candidates.extend(_detect_phone(text, tokens))
    candidates.extend(_detect_generic_numbers(text, tokens))
    return _dedupe(candidates)


def _tokenise(text: str) -> list[_Token]:
    return [
        _Token(raw=m.group(0), norm=m.group(0).casefold(), start=m.start(), end=m.end())
        for m in _TOKEN_RE.finditer(text)
    ]


def _is_number_word(tok: _Token) -> bool:
    return tok.norm in _NUMBER_WORDS


def _is_numberish(tok: _Token) -> bool:
    return tok.norm in (
        _NUMBER_WORDS
        | _SCALE_WORDS
        | _CODE_SWITCH_NUMBER_WORDS
        | _CODE_SWITCH_SCALE_WORDS
        | _COMPOUND_WORDS
        | _DECIMAL_MARKERS
    )


def _is_native_numberish(tok: _Token) -> bool:
    return tok.norm in (
        _NUMBER_WORDS | _SCALE_WORDS | _COMPOUND_WORDS | _DECIMAL_MARKERS
    )


def _is_code_switch_number_word(tok: _Token) -> bool:
    return tok.norm in _CODE_SWITCH_NUMBER_WORDS


def _is_digit_word(tok: _Token) -> bool:
    return tok.norm in _DIGIT_WORDS


def _is_letter_word(tok: _Token) -> bool:
    return tok.norm in _LETTER_WORDS


def _date_day_start(tokens: list[_Token], month_idx: int) -> int | None:
    """Return the first token of a plausible day phrase before a month word."""
    if month_idx == 0:
        return None

    prev = tokens[month_idx - 1]
    if not (_is_number_word(prev) or _is_code_switch_number_word(prev)):
        return None

    # Hindi 21..31 are single tokens, but Devanagari-English forms often
    # arrive as ``थर्टि थ्री``. Include one preceding tens token so the date
    # candidate owns the whole phrase and generic cardinal cannot rewrite only
    # the trailing unit.
    if (
        month_idx >= 2
        and tokens[month_idx - 2].norm in _CODE_SWITCH_TENS_WORDS
        and prev.norm not in _CODE_SWITCH_TENS_WORDS
    ):
        return month_idx - 2

    return month_idx - 1


def _span(text: str, tokens: list[_Token], start: int, end: int, cls: str) -> Span:
    raw_start = tokens[start].start
    raw_end = tokens[end - 1].end
    raw = text[raw_start:raw_end]
    return Span(
        cls=cls,
        raw=raw,
        canonical=raw,
        rule_id=f"spoken.{cls}.v1",
        conf=1.0,
        ambiguous=False,
        start=raw_start,
        end=raw_end,
    )


def _scan_left_numberish(tokens: list[_Token], idx: int, *, limit: int) -> int:
    start = idx
    while start > 0 and idx - (start - 1) <= limit and _is_numberish(tokens[start - 1]):
        start -= 1
    return start


def _scan_right_numberish(tokens: list[_Token], idx: int, *, limit: int) -> int:
    end = idx
    while end < len(tokens) and end - idx < limit and _is_numberish(tokens[end]):
        end += 1
    return end


def _scan_left_native_numberish(tokens: list[_Token], idx: int, *, limit: int) -> int:
    start = idx
    while (
        start > 0
        and idx - (start - 1) <= limit
        and _is_native_numberish(tokens[start - 1])
    ):
        start -= 1
    return start


def _scan_right_native_numberish(tokens: list[_Token], idx: int, *, limit: int) -> int:
    end = idx
    while (
        end < len(tokens)
        and end - idx < limit
        and _is_native_numberish(tokens[end])
    ):
        end += 1
    return end


def _detect_money(text: str, tokens: list[_Token]) -> list[Span]:
    out: list[Span] = []
    for i, tok in enumerate(tokens):
        if tok.norm not in _RUPEE_CUES:
            continue

        left_start = _scan_left_native_numberish(tokens, i, limit=_MAX_NUMBER_TOKENS)
        if left_start < i:
            end = i + 1
            # Preserve the money grammar's optional "... रुपये <N> पैसे" tail
            # when present; unsupported variants still safely fall back later.
            paise_end = _scan_right_native_numberish(tokens, end, limit=2)
            if (
                paise_end > end
                and paise_end < len(tokens)
                and tokens[paise_end].norm in _PAISE_CUES
            ):
                end = paise_end + 1
            out.append(_span(text, tokens, left_start, end, "amount"))

        right_end = _scan_right_native_numberish(tokens, i + 1, limit=_MAX_NUMBER_TOKENS)
        if right_end > i + 1:
            out.append(_span(text, tokens, i, right_end, "amount"))
    return out


def _detect_percent(text: str, tokens: list[_Token]) -> list[Span]:
    out: list[Span] = []
    for i, tok in enumerate(tokens):
        if tok.norm not in _PERCENT_CUES:
            continue
        start = _scan_left_numberish(tokens, i, limit=_MAX_DECIMAL_TOKENS)
        if start < i:
            out.append(_span(text, tokens, start, i + 1, "percent"))
    return out


def _detect_time(text: str, tokens: list[_Token]) -> list[Span]:
    out: list[Span] = []
    for i, tok in enumerate(tokens):
        if tok.norm == "बजे":
            start = _scan_left_numberish(tokens, i, limit=4)
            if start == i:
                continue
            if start > 0 and tokens[start - 1].norm in _TIME_OF_DAY_CUES:
                start -= 1
            end = i + 1
            if end < len(tokens) and tokens[end].norm in _TIME_OF_DAY_CUES:
                end += 1
            out.append(_span(text, tokens, start, end, "time"))
            continue

        if tok.norm != "बजकर":
            continue
        start = _scan_left_numberish(tokens, i, limit=2)
        if start == i:
            continue
        if start > 0 and tokens[start - 1].norm in _TIME_OF_DAY_CUES:
            start -= 1
        minute_end = _scan_right_numberish(tokens, i + 1, limit=2)
        if minute_end == i + 1 or minute_end >= len(tokens):
            continue
        if tokens[minute_end].norm != "मिनट":
            continue
        end = minute_end + 1
        if end < len(tokens) and tokens[end].norm in _TIME_OF_DAY_CUES:
            end += 1
        out.append(_span(text, tokens, start, end, "time"))
    return out


def _detect_date(text: str, tokens: list[_Token]) -> list[Span]:
    out: list[Span] = []
    for i, tok in enumerate(tokens):
        if tok.norm not in _MONTH_WORDS:
            continue
        start = _date_day_start(tokens, i)
        if start is None:
            continue
        end = _scan_right_numberish(tokens, i + 1, limit=4)
        out.append(_span(text, tokens, start, end, "date"))

    # Strong cue without a month word: emit only a tightly bounded candidate.
    # The current Hindi WFST will usually reject it, which is desirable; the
    # caller still gets structured fallback provenance instead of a guess.
    for i, tok in enumerate(tokens):
        if tok.norm not in _STRONG_DATE_CUES:
            continue
        end = _scan_right_numberish(tokens, i + 1, limit=4)
        if end > i + 1:
            out.append(_span(text, tokens, i + 1, end, "date"))
    return out


def _skip_id_fillers(tokens: list[_Token], idx: int) -> int:
    while idx < len(tokens) and tokens[idx].norm in _ID_FILLER_WORDS:
        idx += 1
    return idx


def _nearby_id_cue(
    tokens: list[_Token],
    start: int,
    end: int,
    cues: frozenset[str],
) -> bool:
    cue_start = max(0, start - _ID_CUE_WINDOW)
    cue_end = min(len(tokens), end + _ID_CUE_WINDOW)
    return any(tokens[j].norm in cues for j in range(cue_start, cue_end))


def _pan_shape(tokens: list[_Token], start: int, end: int) -> bool:
    if end - start != 10:
        return False
    return (
        all(_is_letter_word(tok) for tok in tokens[start : start + 5])
        and all(_is_digit_word(tok) for tok in tokens[start + 5 : start + 9])
        and _is_letter_word(tokens[start + 9])
    )


def _detect_ids(text: str, tokens: list[_Token]) -> list[Span]:
    out: list[Span] = []
    for i, tok in enumerate(tokens):
        if tok.norm in _AADHAAR_CUES:
            start = _skip_id_fillers(tokens, i + 1)
            end = start
            while end < len(tokens) and _is_digit_word(tokens[end]):
                end += 1
            if end - start == 12:
                out.append(_span(text, tokens, start, end, "aadhaar"))

        if tok.norm in _PAN_CUES:
            start = _skip_id_fillers(tokens, i + 1)
            end = start
            while end < len(tokens) and (
                _is_letter_word(tokens[end]) or _is_digit_word(tokens[end])
            ):
                end += 1
            if _pan_shape(tokens, start, end):
                out.append(_span(text, tokens, start, end, "pan"))

    i = 0
    while i < len(tokens):
        if not (_is_digit_word(tokens[i]) or _is_letter_word(tokens[i])):
            i += 1
            continue
        end = i
        while end < len(tokens) and (_is_digit_word(tokens[end]) or _is_letter_word(tokens[end])):
            end += 1
        if end - i == 12 and all(_is_digit_word(tok) for tok in tokens[i:end]):
            if _nearby_id_cue(tokens, i, end, _AADHAAR_CUES):
                out.append(_span(text, tokens, i, end, "aadhaar"))
        elif _pan_shape(tokens, i, end) and _nearby_id_cue(tokens, i, end, _PAN_CUES):
            out.append(_span(text, tokens, i, end, "pan"))
        i = end
    return out


def _detect_phone(text: str, tokens: list[_Token]) -> list[Span]:
    out: list[Span] = []
    i = 0
    while i < len(tokens):
        if not _is_digit_word(tokens[i]):
            i += 1
            continue
        end = i
        while end < len(tokens) and _is_digit_word(tokens[end]):
            end += 1
        run_len = end - i
        cue_start = max(0, i - _PHONE_CUE_WINDOW)
        cue_end = min(len(tokens), end + _PHONE_CUE_WINDOW)
        has_cue = any(tokens[j].norm in _PHONE_CUES for j in range(cue_start, cue_end))
        if run_len == 10 and has_cue:
            out.append(_span(text, tokens, i, end, "phone"))
        i = end
    return out


def _detect_generic_numbers(text: str, tokens: list[_Token]) -> list[Span]:
    out: list[Span] = []
    i = 0
    while i < len(tokens):
        if not _is_numberish(tokens[i]):
            i += 1
            continue
        end = i
        while end < len(tokens) and _is_numberish(tokens[end]):
            end += 1
        run = tokens[i:end]
        if len(run) > _MAX_NUMBER_TOKENS:
            i = end
            continue
        if any(tok.norm in _DECIMAL_MARKERS for tok in run):
            marker = next(idx for idx, tok in enumerate(run) if tok.norm in _DECIMAL_MARKERS)
            if 0 < marker < len(run) - 1:
                out.append(_span(text, tokens, i, end, "decimal"))
        else:
            # Long digit-by-digit runs are usually phone/ID-like speech, not a
            # Hindi cardinal phrase.  Leave them alone unless the dedicated
            # phone detector found a cue-bearing ten-digit candidate.
            digit_only = all(_is_digit_word(tok) for tok in run)
            if not (digit_only and len(run) >= 5):
                out.append(_span(text, tokens, i, end, "cardinal"))
        i = end
    return out


def _dedupe(spans: list[Span]) -> list[Span]:
    seen: set[tuple[str, int | None, int | None]] = set()
    out: list[Span] = []
    for span in spans:
        key = (span.cls, span.start, span.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out


__all__ = ["prefilter"]
