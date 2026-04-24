from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import re
import unicodedata

from .hindi_transliteration import (
    generate_hindi_script_variants,
    get_hindi_month_variants,
    to_devanagari_digits,
)

SPACE_RE = re.compile(r"\s+")
CURRENCY_HINT_RE = re.compile(r"(?:₹|rs\.?|inr|rupees?)", re.IGNORECASE)
DATE_PARSE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %m %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%b %d %Y",
)

_UNDER_TWENTY = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def normalize_display_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("_", " ")
    normalized = SPACE_RE.sub(" ", normalized).strip()
    return normalized


def normalize_variant_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("_", " ")
    normalized = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in normalized)
    normalized = SPACE_RE.sub(" ", normalized).strip().lower()
    return normalized


def _dedupe_variants(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    variants: list[str] = []
    for value in values:
        display = normalize_display_text(value)
        key = normalize_variant_key(display)
        if not key or key in seen:
            continue
        seen.add(key)
        variants.append(display)
    return tuple(variants)


def _integer_to_words_under_thousand(value: int) -> list[str]:
    words: list[str] = []
    if value >= 100:
        words.append(_UNDER_TWENTY[value // 100])
        words.append("hundred")
        value %= 100
    if value >= 20:
        words.append(_TENS[value // 10])
        value %= 10
    if value > 0:
        words.append(_UNDER_TWENTY[value])
    return words


def integer_to_indian_words(value: int) -> str:
    if value <= 0:
        return _UNDER_TWENTY[0]

    parts: list[str] = []
    remaining = value
    for scale_value, scale_name in ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand")):
        if remaining >= scale_value:
            chunk = remaining // scale_value
            remaining %= scale_value
            parts.extend(_integer_to_words_under_thousand(chunk))
            parts.append(scale_name)

    parts.extend(_integer_to_words_under_thousand(remaining))
    return " ".join(part for part in parts if part)


def generate_general_variants(text: str, *, language: str | None = None) -> tuple[str, ...]:
    display = normalize_display_text(text)
    if not display:
        return tuple()

    variants = [display]
    lowered = display.lower()
    if lowered != display:
        variants.append(lowered)

    alpha_num_tokens = re.findall(r"[0-9A-Za-z]+", display)
    if len(alpha_num_tokens) > 1:
        joined = "".join(alpha_num_tokens)
        if 3 <= len(joined) <= 32:
            variants.append(joined)
            variants.append(joined.lower())

    if re.fullmatch(r"[A-Z0-9]{2,6}", display):
        variants.append(display.lower())
        variants.append(" ".join(display.lower()))

    if re.fullmatch(r"(?:[A-Za-z0-9]\s+){1,5}[A-Za-z0-9]", display):
        variants.append(display.replace(" ", "").lower())

    if (language or "").strip().lower() == "hi":
        variants.extend(generate_hindi_script_variants(display))

    return _dedupe_variants(variants)


def _extract_numeric_amount(text: str) -> tuple[str | None, bool]:
    display = normalize_display_text(text)
    if not display:
        return None, False

    currency_hint = bool(CURRENCY_HINT_RE.search(display))
    numeric = display
    for marker in ("₹", "rs.", "rs", "inr", "rupees", "rupee"):
        numeric = re.sub(re.escape(marker), "", numeric, flags=re.IGNORECASE)
    numeric = numeric.replace(",", "")
    numeric = SPACE_RE.sub("", numeric)
    if re.fullmatch(r"\d+(?:\.\d+)?", numeric):
        return numeric, currency_hint
    return None, currency_hint


def _extend_hindi_amount_variants(
    variants: list[str],
    *,
    numeric_forms: list[str],
    currency_hint: bool,
) -> None:
    for form in numeric_forms:
        devanagari_form = to_devanagari_digits(form)
        if devanagari_form and devanagari_form != form:
            variants.append(devanagari_form)

        if not currency_hint:
            continue

        for currency_word in ("रुपये", "रुपए"):
            variants.append(f"{form} {currency_word}")
            variants.append(f"{currency_word} {form}")
            if devanagari_form and devanagari_form != form:
                variants.append(f"{devanagari_form} {currency_word}")
                variants.append(f"{currency_word} {devanagari_form}")


def generate_amount_variants(text: str, *, language: str | None = None) -> tuple[str, ...]:
    variants = list(generate_general_variants(text))
    numeric_value, currency_hint = _extract_numeric_amount(text)
    if not numeric_value:
        return _dedupe_variants(variants)

    numeric_forms = [numeric_value]
    variants.append(numeric_value)
    try:
        decimal_value = Decimal(numeric_value)
    except InvalidOperation:
        if (language or "").strip().lower() == "hi":
            _extend_hindi_amount_variants(variants, numeric_forms=numeric_forms, currency_hint=currency_hint)
        return _dedupe_variants(variants)

    if decimal_value == decimal_value.to_integral_value():
        integer_value = int(decimal_value)
        comma_form = f"{integer_value:,}"
        numeric_forms.append(comma_form)
        variants.append(comma_form)
        if 0 < integer_value < 10**12:
            words = integer_to_indian_words(integer_value)
            variants.append(words)
            if currency_hint:
                variants.append(f"{integer_value} rupees")
                variants.append(f"{comma_form} rupees")
                variants.append(f"{words} rupees")
    elif currency_hint:
        variants.append(f"{numeric_value} rupees")

    if (language or "").strip().lower() == "hi":
        _extend_hindi_amount_variants(variants, numeric_forms=numeric_forms, currency_hint=currency_hint)

    return _dedupe_variants(variants)


def _parse_date(text: str) -> datetime | None:
    display = normalize_display_text(text)
    if not display:
        return None

    cleaned = display.replace(",", "")
    for date_format in DATE_PARSE_FORMATS:
        try:
            return datetime.strptime(cleaned, date_format)
        except ValueError:
            continue
    return None


def _extend_hindi_date_variants(variants: list[str], parsed: datetime) -> None:
    numeric_forms = [
        parsed.strftime("%Y-%m-%d"),
        parsed.strftime("%d/%m/%Y"),
        parsed.strftime("%d-%m-%Y"),
        f"{parsed.day}/{parsed.month}/{parsed.year}",
        f"{parsed.day}-{parsed.month}-{parsed.year}",
    ]
    for form in numeric_forms:
        devanagari_form = to_devanagari_digits(form)
        if devanagari_form and devanagari_form != form:
            variants.append(devanagari_form)

    day = str(parsed.day)
    padded_day = f"{parsed.day:02d}"
    year = str(parsed.year)
    day_devanagari = to_devanagari_digits(day)
    padded_day_devanagari = to_devanagari_digits(padded_day)
    year_devanagari = to_devanagari_digits(year)

    for month in get_hindi_month_variants(parsed.month):
        variants.extend(
            [
                f"{day} {month} {year}",
                f"{padded_day} {month} {year}",
                f"{month} {day} {year}",
                f"{month} {padded_day} {year}",
                f"{day_devanagari} {month} {year_devanagari}",
                f"{padded_day_devanagari} {month} {year_devanagari}",
                f"{month} {day_devanagari} {year_devanagari}",
                f"{month} {padded_day_devanagari} {year_devanagari}",
            ]
        )


def generate_date_variants(text: str, *, language: str | None = None) -> tuple[str, ...]:
    variants = list(generate_general_variants(text))
    parsed = _parse_date(text)
    if parsed is None:
        return _dedupe_variants(variants)

    variants.extend(
        [
            parsed.strftime("%Y-%m-%d"),
            parsed.strftime("%d/%m/%Y"),
            parsed.strftime("%d-%m-%Y"),
            parsed.strftime("%d %B %Y"),
            parsed.strftime("%B %d %Y"),
        ]
    )
    if (language or "").strip().lower() == "hi":
        _extend_hindi_date_variants(variants, parsed)
    return _dedupe_variants(variants)
