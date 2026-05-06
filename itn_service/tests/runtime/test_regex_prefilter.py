"""Unit tests for `runtime.regex_prefilter.prefilter`.

Coverage targets the ten classes plus a block of adversarial cases:
phone digits embedded in URLs, IFSC look-alikes, PAN look-alikes,
Aadhaar vs phone disambiguation, and so on.
"""

from __future__ import annotations

import pytest

from itn_service.runtime.contract import Span
from itn_service.runtime.regex_prefilter import prefilter


def _by_cls(spans: list[Span]) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for s in spans:
        out.setdefault(s.cls, []).append(s)
    return out


# --- empty / no-match --------------------------------------------------------


def test_empty_string_returns_empty_list() -> None:
    assert prefilter("") == []


def test_plain_words_no_matches() -> None:
    assert prefilter("the quick brown fox") == []


def test_devanagari_only_no_latin_matches() -> None:
    assert prefilter("नमस्ते दुनिया") == []


# --- contract invariants -----------------------------------------------------


def test_canonical_equals_raw_for_every_span() -> None:
    spans = prefilter("Call 9876543210, mail x@y.com, IFSC HDFC0001234")
    assert spans
    for s in spans:
        assert s.canonical == s.raw
        assert s.conf == 1.0
        assert s.ambiguous is False
        assert s.rule_id.startswith("prefilter.")
        assert s.rule_id.endswith(".v1")


def test_offsets_are_consistent_with_text_slice() -> None:
    text = "Phone 9876543210 today"
    spans = prefilter(text)
    assert len(spans) == 1
    s = spans[0]
    assert s.start is not None and s.end is not None
    assert text[s.start : s.end] == s.raw


# --- PHONE_LATN --------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        # Bare 10-digit, all valid leading digits (6, 7, 8, 9).
        "9876543210",
        "8123456789",
        "7000000000",
        "6543210987",
        # With country code in disambiguating forms (the literal '+'
        # is what makes these unambiguous vs Aadhaar shape).
        "+919876543210",
        "+91 9876543210",
        "+91-9876543210",
        "+91.9876543210",
        # Leading '0' for STD-style dial.
        "09876543210",
    ],
)
def test_phone_variants_match(raw: str) -> None:
    spans = prefilter(f"call {raw} please")
    assert any(s.cls == "phone" and raw in s.raw for s in spans), spans


def test_phone_first_digit_below_six_does_not_match() -> None:
    # Indian mobiles begin with 6, 7, 8 or 9.
    assert _by_cls(prefilter("call 5876543210 please")).get("phone") is None


def test_phone_too_short_does_not_match() -> None:
    # Nine digits, not ten.
    assert _by_cls(prefilter("call 987654321 please")).get("phone") is None


def test_phone_too_long_does_not_match_as_phone() -> None:
    # Eleven digit run -> not a phone (and not Aadhaar shape either).
    assert "phone" not in _by_cls(prefilter("call 98765432109 please"))


# --- PAN ---------------------------------------------------------------------


def test_pan_basic() -> None:
    spans = prefilter("PAN: ABCDE1234F filed.")
    pans = _by_cls(spans)["pan"]
    assert len(pans) == 1
    assert pans[0].raw == "ABCDE1234F"


def test_pan_lowercase_also_matches() -> None:
    spans = prefilter("pan abcde1234f")
    assert any(s.cls == "pan" for s in spans)


def test_pan_too_short_no_match() -> None:
    assert "pan" not in _by_cls(prefilter("ABCDE1234"))


def test_pan_too_many_digits_no_match() -> None:
    assert "pan" not in _by_cls(prefilter("ABCDE12345F"))


# --- AADHAAR -----------------------------------------------------------------


def test_aadhaar_no_separator() -> None:
    spans = prefilter("Aadhaar 234567890123 verified")
    aadhaar = _by_cls(spans)["aadhaar"]
    assert aadhaar[0].raw == "234567890123"


def test_aadhaar_with_spaces() -> None:
    spans = prefilter("Aadhaar 2345 6789 0123 verified")
    assert any(s.cls == "aadhaar" for s in spans)


def test_aadhaar_with_dashes() -> None:
    spans = prefilter("Aadhaar 2345-6789-0123 verified")
    assert any(s.cls == "aadhaar" for s in spans)


def test_aadhaar_first_digit_low_does_not_match() -> None:
    # First digit must be 2-9 per UIDAI rules.
    assert "aadhaar" not in _by_cls(prefilter("ID 0234567890123 issued"))
    assert "aadhaar" not in _by_cls(prefilter("ID 1234567890123 issued"))


# --- IFSC --------------------------------------------------------------------


def test_ifsc_basic() -> None:
    spans = prefilter("IFSC HDFC0001234 main branch")
    assert any(s.cls == "ifsc" for s in spans)


def test_ifsc_lowercase() -> None:
    assert any(s.cls == "ifsc" for s in prefilter("ifsc hdfc0001234"))


def test_ifsc_fifth_char_must_be_zero() -> None:
    # Five-letter prefix without the mandatory '0' fails IFSC.
    assert "ifsc" not in _by_cls(prefilter("HDFCX001234 wrong"))


def test_ifsc_wrong_length_no_match() -> None:
    assert "ifsc" not in _by_cls(prefilter("HDFC000123"))   # 9 chars


# --- AMOUNT_LATN -------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "₹1250",
        "₹1,250",
        "₹1,25,000",
        "₹1,25,000.50",
        "Rs1250",
        "Rs.1,250",
        "INR 5000",
        "$10",
        "$1,000.00",
        "£99",
        "€20",
    ],
)
def test_amount_variants(raw: str) -> None:
    spans = prefilter(f"pay {raw} now")
    assert any(s.cls == "amount" for s in spans), (raw, spans)


def test_amount_without_currency_symbol_does_not_match() -> None:
    assert "amount" not in _by_cls(prefilter("pay 1250 now"))


# --- PERCENT_LATN ------------------------------------------------------------


def test_percent_integer() -> None:
    spans = prefilter("growth 12% YoY")
    assert any(s.cls == "percent" and s.raw == "12%" for s in spans)


def test_percent_decimal() -> None:
    spans = prefilter("rate 12.5%")
    pcts = _by_cls(spans)["percent"]
    assert pcts[0].raw == "12.5%"


def test_percent_with_space() -> None:
    spans = prefilter("rate 12 % per annum")
    assert any(s.cls == "percent" for s in spans)


def test_percent_without_digits_no_match() -> None:
    assert prefilter("just a % sign") == []


# --- DATE_NUMERIC ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "12/05/2026",
        "12-05-2026",
        "12.05.2026",
        "1/5/26",
        "31/12/1999",
        "01/01/2000",
    ],
)
def test_date_numeric_variants(raw: str) -> None:
    spans = prefilter(f"on {raw} we met")
    assert any(s.cls == "date" and s.raw == raw for s in spans)


def test_date_invalid_day_no_match() -> None:
    assert "date" not in _by_cls(prefilter("32/01/2024"))


def test_date_invalid_month_no_match() -> None:
    assert "date" not in _by_cls(prefilter("12/13/2024"))


# --- TIME_NUMERIC ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["5:30", "05:30", "17:30", "5:30 PM", "5:30 AM", "5:30:45", "23:59"],
)
def test_time_numeric_variants(raw: str) -> None:
    spans = prefilter(f"at {raw} sharp")
    assert any(s.cls == "time" for s in spans), (raw, spans)


def test_time_invalid_hour_no_match() -> None:
    assert "time" not in _by_cls(prefilter("at 25:30 sharp"))


def test_time_invalid_minute_no_match() -> None:
    assert "time" not in _by_cls(prefilter("at 12:60 sharp"))


# --- EMAIL -------------------------------------------------------------------


def test_email_basic() -> None:
    spans = prefilter("contact me at user@example.com please")
    assert any(s.cls == "email" and s.raw == "user@example.com" for s in spans)


def test_email_with_subdomain_and_plus() -> None:
    spans = prefilter("send to first.last+tag@mail.example.co.in today")
    assert any(s.cls == "email" for s in spans)


def test_email_without_at_no_match() -> None:
    assert "email" not in _by_cls(prefilter("user.example.com"))


# --- URL ---------------------------------------------------------------------


def test_url_https() -> None:
    spans = prefilter("see https://example.com/path?x=1 today")
    urls = _by_cls(spans)["url"]
    assert urls[0].raw.startswith("https://example.com")


def test_url_bare_www() -> None:
    spans = prefilter("see www.example.com today")
    assert any(s.cls == "url" for s in spans)


def test_url_http() -> None:
    assert any(s.cls == "url" for s in prefilter("http://x.io/y"))


# --- adversarial (10 cases) --------------------------------------------------


def test_adversarial_phone_inside_url_is_swallowed_by_url() -> None:
    text = "Visit https://example.com/9876543210/path"
    spans = prefilter(text)
    classes = {s.cls for s in spans}
    assert "url" in classes
    assert "phone" not in classes


def test_adversarial_phone_inside_email_is_swallowed_by_email() -> None:
    text = "Mail 9876543210@telco.in for info"
    spans = prefilter(text)
    assert "email" in {s.cls for s in spans}
    assert "phone" not in {s.cls for s in spans}


def test_adversarial_aadhaar_overlapping_phone_prefers_aadhaar() -> None:
    # 12-digit run starting with 9 would also satisfy a 10-digit phone
    # at offsets 0..10 or 2..12. The overlap resolver must prefer
    # AADHAAR (priority 5) over PHONE (priority 6).
    text = "ID 987654321023"
    spans = prefilter(text)
    classes = {s.cls for s in spans}
    assert "aadhaar" in classes
    assert "phone" not in classes


def test_adversarial_pan_lookalike_with_extra_letter_does_not_match() -> None:
    text = "Code ABCDEF1234G filed"
    classes = {s.cls for s in prefilter(text)}
    assert "pan" not in classes


def test_adversarial_ifsc_lookalike_missing_zero_does_not_match() -> None:
    text = "Code HDFC1001234 filed"
    classes = {s.cls for s in prefilter(text)}
    assert "ifsc" not in classes


def test_adversarial_phone_with_embedded_text_is_separate_match() -> None:
    text = "Numbers: 9876543210 and 8000000000 both."
    spans = prefilter(text)
    phones = [s for s in spans if s.cls == "phone"]
    assert len(phones) == 2
    assert {p.raw for p in phones} == {"9876543210", "8000000000"}


def test_adversarial_url_with_amount_inside_is_swallowed_by_url() -> None:
    text = "Visit https://example.com/$1000/path"
    classes = {s.cls for s in prefilter(text)}
    assert "url" in classes
    assert "amount" not in classes


def test_adversarial_alnum_id_ab1234_is_not_pan_or_phone() -> None:
    spans = prefilter("Code AB1234 filed")
    # Must not be classified as PAN, phone, or anything else.
    assert spans == []


def test_adversarial_phone_adjacent_to_alpha_no_false_match() -> None:
    # The phone has a letter immediately attached -> word boundary
    # fails -> not a phone.
    classes = {s.cls for s in prefilter("ABC9876543210XYZ")}
    assert "phone" not in classes


def test_adversarial_long_digit_run_classified_only_as_aadhaar_when_valid() -> None:
    text = "ref 234567890123456 filed"  # 15 digits — neither phone nor aadhaar
    classes = {s.cls for s in prefilter(text)}
    assert "aadhaar" not in classes
    assert "phone" not in classes


# --- multi-span integration --------------------------------------------------


def test_multi_class_in_single_text_all_distinct_offsets() -> None:
    text = (
        "Call 9876543210 today, pay ₹1,250 by 12/05/2026 at 5:30 PM. "
        "Mail user@example.com or visit https://example.com. "
        "PAN ABCDE1234F, IFSC HDFC0001234, growth 12.5%"
    )
    spans = prefilter(text)
    classes = {s.cls for s in spans}
    expected = {"phone", "amount", "date", "time", "email", "url",
                "pan", "ifsc", "percent"}
    missing = expected - classes
    assert not missing, f"missing classes: {missing}"

    # Spans are sorted by start and non-overlapping.
    starts = [s.start for s in spans]
    assert starts == sorted(starts)
    for a, b in zip(spans, spans[1:]):
        assert a.end is not None and b.start is not None
        assert a.end <= b.start, (a, b)


def test_no_overlap_invariant_under_aggressive_input() -> None:
    text = "https://x.com/9876543210 ABCDE1234F HDFC0001234 ₹1,000 12.5%"
    spans = prefilter(text)
    for a, b in zip(spans, spans[1:]):
        assert a.end is not None and b.start is not None
        assert a.end <= b.start
