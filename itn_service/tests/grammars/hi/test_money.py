"""Tests for the Hindi money grammar.

Coverage targets from the stage-2 deliverable:

* >= 150 gold cases (positive, multi-form: spoken + Latin + symbol).
* >= 20 adversarial cases (must fail to normalise; cue absent, malformed
  amounts, mid-utterance corrections, etc.).

All cases run through the FAR-loaded pipeline (``WFSTPipeline("hi")``)
so they verify the production graph, not the source ``MONEY`` FST
directly. Compile the FAR first::

    python -m itn_service.compile --lang hi
"""

from __future__ import annotations

import pytest

from itn_service.runtime.wfst_pipeline import WFSTPipeline

from ._hi_spell import indian_group, spell_hi


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("hi")


# ---------------------------------------------------------------------------
# Hand-curated spoken / symbol gold (covers every surface form).
# ---------------------------------------------------------------------------

_HAND_GOLD: list[tuple[str, str]] = [
    # Form 1: spoken int + post-cue (rupee word).
    ("एक रुपये", "₹1"),
    ("एक रुपया", "₹1"),
    ("दस रुपये", "₹10"),
    ("पच्चीस रुपये", "₹25"),
    ("एक सौ रुपये", "₹100"),
    ("एक सौ पच्चीस रुपये", "₹125"),
    ("पाँच सौ रुपये", "₹500"),
    ("एक हज़ार रुपये", "₹1,000"),
    ("एक हज़ार पाँच सौ रुपये", "₹1,500"),
    ("दस हज़ार रुपये", "₹10,000"),
    ("एक लाख रुपये", "₹1,00,000"),
    ("एक लाख पच्चीस हज़ार रुपये", "₹1,25,000"),
    ("दस लाख रुपये", "₹10,00,000"),
    ("एक करोड़ रुपये", "₹1,00,00,000"),
    # Half / quarter compounds with scale (integer).
    ("सवा सौ रुपये", "₹125"),
    ("डेढ़ सौ रुपये", "₹150"),
    ("ढाई सौ रुपये", "₹250"),
    ("सवा हज़ार रुपये", "₹1,250"),
    ("डेढ़ हज़ार रुपये", "₹1,500"),
    ("ढाई हज़ार रुपये", "₹2,500"),
    ("ढाई लाख रुपये", "₹2,50,000"),
    ("साढ़े पाँच हज़ार रुपये", "₹5,500"),
    ("साढ़े दस हज़ार रुपये", "₹10,500"),
    ("पौने पाँच हज़ार रुपये", "₹4,750"),
    # Form 2: paise tail.
    ("एक हज़ार रुपये पच्चीस पैसे", "₹1,000.25"),
    ("एक रुपया पाँच पैसे", "₹1.05"),       # single-digit paise -> 2-digit
    ("सौ रुपये पचास पैसे", "₹100.50"),
    ("एक सौ रुपये एक पैसा", "₹100.01"),
    ("पाँच सौ रुपये निन्यानवे पैसे", "₹500.99"),
    # Form 3: spoken decimal + post-cue.
    ("एक दशमलव पाँच रुपये", "₹1.5"),
    ("शून्य दशमलव पाँच रुपये", "₹0.5"),
    ("बारह दशमलव पाँच रुपये", "₹12.5"),
    # Bare half/quarter compounds (fractional rupees).
    ("डेढ़ रुपये", "₹1.5"),
    ("ढाई रुपये", "₹2.5"),
    ("सवा दो रुपये", "₹2.25"),
    ("साढ़े तीन रुपये", "₹3.5"),
    ("पौने चार रुपये", "₹3.75"),
    # Form 4: pre-cue (rupee word first).
    ("रुपये एक हज़ार", "₹1,000"),
    ("रुपये एक लाख", "₹1,00,000"),
    # Form 5: pre-cue + decimal.
    ("रुपये एक दशमलव पाँच", "₹1.5"),
    # Form 6: symbol prefix.
    ("₹1000", "₹1,000"),
    ("₹125", "₹125"),
    ("₹1,25,000", "₹1,25,000"),
    ("₹10000000", "₹1,00,00,000"),
    ("Rs. 250", "₹250"),
    ("Rs 1500", "₹1,500"),
    ("rs. 1500", "₹1,500"),
    ("INR 12345", "₹12,345"),
    # Form 7: symbol + decimal.
    ("Rs 1500.50", "₹1,500.50"),
    ("₹1500.5", "₹1,500.5"),
    ("Rs. 99.99", "₹99.99"),
    # Latin int with rupee word post-cue.
    ("1000 रुपये", "₹1,000"),
    ("125 रुपये", "₹125"),
]


@pytest.mark.parametrize("raw,expected", _HAND_GOLD)
def test_hand_gold(pipeline: WFSTPipeline, raw: str, expected: str) -> None:
    out = pipeline.normalize_span(raw, "money")
    assert out == expected, (raw, expected, out)


# ---------------------------------------------------------------------------
# Programmatic gold via spell_hi: 100 spoken + 100 Rs-prefix Latin cases
# across diverse magnitudes.
# ---------------------------------------------------------------------------

def _gen_money_spoken(n_per_band: int = 25) -> list[tuple[str, str]]:
    """Generate (spoken Hindi + रुपये -> ₹<grouped>) cases across the
    integer-magnitude bands the production grammar handles."""
    bands: list[range] = [
        range(1, 100),                            # 1..99
        range(100, 1000, 7),                      # hundreds, every 7th
        range(1000, 100_000, 833),                # thousands
        range(100_000, 10_000_000, 95_321),       # lakhs
    ]
    out: list[tuple[str, str]] = []
    for band in bands:
        seen = 0
        for n in band:
            if seen >= n_per_band:
                break
            spoken = spell_hi(n)
            out.append((f"{spoken} रुपये", f"₹{indian_group(n)}"))
            seen += 1
    return out


def _gen_money_latin(n_cases: int = 100) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    samples = [
        1, 5, 25, 99, 100, 125, 500, 999, 1000, 1500, 2500, 9999,
        10_000, 12_500, 50_000, 99_999, 100_000, 125_000, 500_000,
        999_999, 1_000_000, 1_250_000, 9_999_999, 10_000_000,
        12_345_678, 99_999_999, 100_000_000,
    ]
    while len(out) < n_cases and samples:
        for n in samples:
            if len(out) >= n_cases:
                break
            grouped = indian_group(n)
            out.append((f"Rs {n}", f"₹{grouped}"))
            if len(out) >= n_cases:
                break
            out.append((f"Rs. {n}", f"₹{grouped}"))
            if len(out) >= n_cases:
                break
            out.append((f"₹{n}", f"₹{grouped}"))
            if len(out) >= n_cases:
                break
            out.append((f"INR {n}", f"₹{grouped}"))
        if len(out) < n_cases:
            samples = samples + [s + 1 for s in samples]  # diversify
    return out[:n_cases]


_PROG_SPOKEN = _gen_money_spoken()
_PROG_LATIN = _gen_money_latin()


@pytest.mark.parametrize("raw,expected", _PROG_SPOKEN, ids=[r for r, _ in _PROG_SPOKEN])
def test_programmatic_spoken(pipeline: WFSTPipeline, raw: str, expected: str) -> None:
    assert pipeline.normalize_span(raw, "money") == expected


@pytest.mark.parametrize("raw,expected", _PROG_LATIN, ids=[r for r, _ in _PROG_LATIN])
def test_programmatic_latin(pipeline: WFSTPipeline, raw: str, expected: str) -> None:
    assert pipeline.normalize_span(raw, "money") == expected


# ---------------------------------------------------------------------------
# Adversarial — every entry must yield None (no cue / malformed / corrected).
# ---------------------------------------------------------------------------

_ADVERSARIAL: list[str] = [
    # No cue at all — must NOT fire as money.
    "एक हज़ार",
    "एक लाख पच्चीस हज़ार",
    "1000",
    "1,25,000",
    # Cue but garbage amount.
    "नमस्ते रुपये",
    "रुपये xyz",
    "Rs. abcdef",
    "₹",
    "₹ ",
    # Empty / whitespace.
    "",
    "   ",
    # Numbers that look like phone or ID (10 digits) — without cue, no money.
    "9876543210",
    # Wrong scale composition.
    "एक हज़ार सौ रुपये पचास पैसे पैसे",   # double पैसे
    "एक रुपये पाँच",                       # paise word missing
    "एक रुपये 5 पैसे",                      # 1-digit Latin paise rejected
    "एक रुपये 555 पैसे",                    # 3-digit paise
    # Mid-utterance corrections — handled by self_correction, not money.
    "एक हज़ार रुपये नहीं दो हज़ार रुपये",
    # Cross-cue contamination (rupee word + percent cue).
    "एक हज़ार रुपये प्रतिशत",
    # Stray currency hint without amount.
    "रुपये",
    "Rs.",
]


@pytest.mark.parametrize("raw", _ADVERSARIAL)
def test_adversarial_no_match(pipeline: WFSTPipeline, raw: str) -> None:
    assert pipeline.normalize_span(raw, "money") is None


# ---------------------------------------------------------------------------
# Coverage / classifier wrapper.
# ---------------------------------------------------------------------------

def test_classifier_emits_nemo_tag(pipeline: WFSTPipeline) -> None:
    out = pipeline.classify_span("एक हज़ार रुपये", "money")
    assert out == 'money { currency: "INR" amount: "1,000" }'


def test_total_case_count_meets_minimum() -> None:
    total = len(_HAND_GOLD) + len(_PROG_SPOKEN) + len(_PROG_LATIN)
    assert total >= 150, total
    assert len(_ADVERSARIAL) >= 20, len(_ADVERSARIAL)
