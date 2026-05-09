"""Tests for the Hindi date grammar.

The grammar is described in ``itn_service/grammars/hi/date.py``. Two
surface families with different safety properties are exercised:

* ``DATE_MONTHWORD`` — always safe; month is named, so DD/MM is
  unambiguous. Tested via ``pipeline.normalize_span(raw, "date_monthword")``
  and via ``pipeline.normalize_date(raw, date_order=...)`` (which
  always tries the month-word branch first).

* ``DATE_NUMERIC`` — DMY-only; ``12/05/2026`` is meaningful only when
  the tenant policy is day-first. Tested via
  ``pipeline.normalize_date(raw, date_order="DMY")`` for positive
  cases, and via ``date_order="MDY"`` to verify that the same input
  is *deferred* with ``fallback_reason="ambiguous_numeric_date"``.

Coverage targets:

* >= 100 gold date cases (combined month-word + numeric DMY).
* Ambiguous numeric forms must NOT be auto-resolved when policy != DMY.

Compile the FAR first::

    python -m itn_service.compile --lang hi
"""

from __future__ import annotations

import pytest

from itn_service.runtime.wfst_pipeline import WFSTPipeline


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("hi")


# ---------------------------------------------------------------------------
# Month-word gold cases.
#
# These should normalise regardless of tenant locale_policy: the month
# name is the disambiguator. We test them through both surfaces:
#  * ``normalize_span(raw, "date_monthword")`` — direct branch hit.
#  * ``normalize_date(raw, date_order="DMY")`` — full pipeline.
#
# Output is canonical "DD/MM[/YYYY]" with leading zeros.
# ---------------------------------------------------------------------------

_MONTHWORD_GOLD: list[tuple[str, str]] = [
    # Hindi day + Hindi month + spoken year (canonical form per blueprint).
    ("बारह मई दो हज़ार छब्बीस", "12/05/2026"),
    ("एक जनवरी दो हज़ार छब्बीस", "01/01/2026"),
    ("इकतीस दिसंबर दो हज़ार छब्बीस", "31/12/2026"),
    ("इकतीस दिसम्बर दो हज़ार छब्बीस", "31/12/2026"),
    ("पंद्रह अगस्त एक हज़ार नौ सौ सैंतालीस", "15/08/1947"),
    ("दो अक्टूबर एक हज़ार नौ सौ उनहत्तर", "02/10/1969"),
    ("पच्चीस फरवरी दो हज़ार बीस", "25/02/2020"),
    ("पच्चीस फ़रवरी दो हज़ार बीस", "25/02/2020"),
    ("चौदह नवंबर एक हज़ार नौ सौ नवासी", "14/11/1989"),
    ("चौदह नवम्बर एक हज़ार नौ सौ नवासी", "14/11/1989"),
    ("तीस अप्रैल दो हज़ार छब्बीस", "30/04/2026"),
    ("बीस मार्च दो हज़ार पच्चीस", "20/03/2025"),
    ("नौ जुलाई दो हज़ार चौबीस", "09/07/2024"),
    ("तेईस सितंबर दो हज़ार तेईस", "23/09/2023"),
    ("सात जून दो हज़ार बाईस", "07/06/2022"),
    ("सत्रह अगस्त दो हज़ार इक्कीस", "17/08/2021"),
    # Hindi day + Hindi month, no year (year never invented).
    ("बारह मई", "12/05"),
    ("एक जनवरी", "01/01"),
    ("इकतीस दिसंबर", "31/12"),
    ("दो जुलाई", "02/07"),
    ("नौ नवंबर", "09/11"),
    ("पंद्रह अगस्त", "15/08"),
    # Hindi day + English month (Hinglish).
    ("बारह May दो हज़ार छब्बीस", "12/05/2026"),
    ("एक January", "01/01"),
    ("इकतीस December", "31/12"),
    ("बीस July", "20/07"),
    ("दस October दो हज़ार बीस", "10/10/2020"),
    # Latin day + Hindi month.
    ("12 मई 2026", "12/05/2026"),
    ("12 मई", "12/05"),
    ("31 दिसंबर 1999", "31/12/1999"),
    ("1 जनवरी 2000", "01/01/2000"),
    ("5 जून", "05/06"),
    ("15 अगस्त 1947", "15/08/1947"),
    # Latin day + English month (Hinglish).
    ("12 May 2026", "12/05/2026"),
    ("12 May", "12/05"),
    ("1 January 2000", "01/01/2000"),
    ("31 December 1999", "31/12/1999"),
    ("4 July 1776", "04/07/1776"),
    ("15 August 1947", "15/08/1947"),
    ("2 October 1969", "02/10/1969"),
    ("25 December 2024", "25/12/2024"),
    # Abbreviated English months (Hinglish abbreviations).
    ("12 Jan 2026", "12/01/2026"),
    ("31 Dec 1999", "31/12/1999"),
    ("4 Feb 2024", "04/02/2024"),
    ("9 Mar 2023", "09/03/2023"),
    ("18 Apr 2022", "18/04/2022"),
    ("7 Jun 2021", "07/06/2021"),
    ("19 Jul 2020", "19/07/2020"),
    ("28 Aug 2019", "28/08/2019"),
    ("11 Sep 2018", "11/09/2018"),
    ("3 Sept 2018", "03/09/2018"),
    ("22 Oct 2017", "22/10/2017"),
    ("14 Nov 2016", "14/11/2016"),
    # Lower-case English month variants.
    ("12 may 2026", "12/05/2026"),
    ("12 january 2000", "12/01/2000"),
    ("12 dec 1999", "12/12/1999"),
    # Already-Latin-padded day inputs.
    ("05 जून 2024", "05/06/2024"),
    ("09 अगस्त 2024", "09/08/2024"),
    ("01 January", "01/01"),
    ("09 December", "09/12"),
    # Latin day with Latin year.
    ("4 जुलाई 1776", "04/07/1776"),
    ("12 मई 1857", "12/05/1857"),
]


@pytest.mark.parametrize("raw,expected", _MONTHWORD_GOLD)
def test_monthword_gold_via_branch(
    pipeline: WFSTPipeline, raw: str, expected: str,
) -> None:
    """Direct hit on ``DATE_MONTHWORD`` FAR entry."""
    assert pipeline.normalize_span(raw, "date_monthword") == expected, raw


@pytest.mark.parametrize("raw,expected", _MONTHWORD_GOLD)
def test_monthword_gold_via_pipeline_dmy(
    pipeline: WFSTPipeline, raw: str, expected: str,
) -> None:
    """Same cases through the policy-aware ``normalize_date`` (DMY tenant)."""
    res = pipeline.normalize_date(raw, date_order="DMY")
    assert res.canonical == expected, raw
    assert res.fallback_reason is None


@pytest.mark.parametrize("raw,expected", _MONTHWORD_GOLD)
def test_monthword_gold_via_pipeline_mdy(
    pipeline: WFSTPipeline, raw: str, expected: str,
) -> None:
    """Month-word dates work for *every* tenant — the month name is
    the disambiguator, so locale policy doesn't matter."""
    res = pipeline.normalize_date(raw, date_order="MDY")
    assert res.canonical == expected, raw
    assert res.fallback_reason is None


# ---------------------------------------------------------------------------
# Numeric DMY gold cases.
#
# These succeed only when the tenant policy is DMY. Tested here against
# DMY only; the MDY-defer behaviour is exercised by the ambiguous-numeric
# tests further down and by ``test_locale_policy.py``.
# ---------------------------------------------------------------------------

_NUMERIC_DMY_GOLD: list[tuple[str, str]] = [
    # 4-digit year, slash separator (canonical input shape).
    ("12/05/2026", "12/05/2026"),
    ("01/01/2026", "01/01/2026"),
    ("31/12/1999", "31/12/1999"),
    ("15/08/1947", "15/08/1947"),
    ("02/10/1969", "02/10/1969"),
    ("04/07/1776", "04/07/1776"),
    ("25/12/2024", "25/12/2024"),
    ("28/02/2024", "28/02/2024"),
    # Single-digit day or month — padded on output.
    ("5/6/2024", "05/06/2024"),
    ("1/1/2000", "01/01/2000"),
    ("9/9/2009", "09/09/2009"),
    ("5/12/2024", "05/12/2024"),
    ("12/5/2024", "12/05/2024"),
    ("3/4/2026", "03/04/2026"),
    # Hyphen separator.
    ("12-05-2026", "12/05/2026"),
    ("01-01-2026", "01/01/2026"),
    ("31-12-1999", "31/12/1999"),
    ("15-08-1947", "15/08/1947"),
    # Dot separator.
    ("12.05.2026", "12/05/2026"),
    ("31.12.1999", "31/12/1999"),
    ("15.08.1947", "15/08/1947"),
    # 2-digit year — expanded to 20YY.
    ("12-05-26", "12/05/2026"),
    ("12/05/26", "12/05/2026"),
    ("12.05.26", "12/05/2026"),
    ("01-01-00", "01/01/2000"),
    ("31-12-99", "31/12/2099"),  # 2-digit year always 20YY (documented choice)
    ("15-08-47", "15/08/2047"),
    ("28-02-24", "28/02/2024"),
    # Mixed separators are still consumed (each is independent).
    ("12/05-2026", "12/05/2026"),
    ("12-05/2026", "12/05/2026"),
    ("12.05-26", "12/05/2026"),
    # Leading-zero variants.
    ("05/06/2024", "05/06/2024"),
    ("09/09/2009", "09/09/2009"),
    ("01/12/2026", "01/12/2026"),
]


@pytest.mark.parametrize("raw,expected", _NUMERIC_DMY_GOLD)
def test_numeric_dmy_gold_via_branch(
    pipeline: WFSTPipeline, raw: str, expected: str,
) -> None:
    """Direct hit on ``DATE_NUMERIC`` FAR entry (DMY semantics baked in)."""
    assert pipeline.normalize_span(raw, "date_numeric") == expected, raw


@pytest.mark.parametrize("raw,expected", _NUMERIC_DMY_GOLD)
def test_numeric_dmy_gold_via_pipeline(
    pipeline: WFSTPipeline, raw: str, expected: str,
) -> None:
    """Same cases through ``normalize_date`` with a DMY tenant."""
    res = pipeline.normalize_date(raw, date_order="DMY")
    assert res.canonical == expected, raw
    assert res.fallback_reason is None


# ---------------------------------------------------------------------------
# Critical safety: ambiguous numeric dates MUST NOT auto-resolve under
# non-DMY tenants. They must be deferred with the documented reason.
# ---------------------------------------------------------------------------

_AMBIGUOUS_NUMERIC_INPUTS: list[str] = [
    "12/05/2026",
    "5/6/2024",
    "01/01/2000",
    "31-12-1999",
    "12-05-26",
    "12.05.2026",
    "1/1/00",
    "9/9/9",
    "11/12/2025",
    "07/04/2024",
    "01/02/2024",
    "10/11/2024",
]


@pytest.mark.parametrize("raw", _AMBIGUOUS_NUMERIC_INPUTS)
def test_numeric_deferred_for_mdy_tenant(
    pipeline: WFSTPipeline, raw: str,
) -> None:
    res = pipeline.normalize_date(raw, date_order="MDY")
    assert res.canonical is None, raw
    assert res.fallback_reason == "ambiguous_numeric_date", raw


@pytest.mark.parametrize("raw", _AMBIGUOUS_NUMERIC_INPUTS)
def test_numeric_deferred_for_ymd_tenant(
    pipeline: WFSTPipeline, raw: str,
) -> None:
    res = pipeline.normalize_date(raw, date_order="YMD")
    assert res.canonical is None, raw
    assert res.fallback_reason == "ambiguous_numeric_date", raw


# ---------------------------------------------------------------------------
# Adversarial: cue absent / malformed dates must not normalise.
# ---------------------------------------------------------------------------

_ADVERSARIAL_DATE: list[str] = [
    # Bare numbers (no separator-shaped date).
    "बारह",
    "12",
    "2026",
    # Wrong shapes.
    "12/05",                 # day/month only — could be either order
    "12-05",                 # ditto
    "12 / 05 / 2026",        # spaced separators around digits not in spec
    # Out-of-range day or month.
    "32 मई 2026",            # day 32 — invalid
    "12/13/2026",            # month 13 — invalid
    "0 जनवरी 2024",          # day 0 — invalid (only 1-31 in lexicon)
    # No month name and no separators.
    "12 2026",
    "twelve May",            # English day word (not a Hindi cardinal)
    # Garbage / empty.
    "",
    "   ",
    "नमस्ते",
    "abc",
    "12/abc/2026",
    "12/मई/2026",            # spoken month between numeric separators
    # Mid-utterance correction — multiple month words.
    "बारह मई नहीं तेरह मई",
    # Time-of-day / unrelated cue.
    "बारह बजे",
    # Empty year placeholder.
    "12/05/",
    "/05/2026",
]


@pytest.mark.parametrize("raw", _ADVERSARIAL_DATE)
def test_adversarial_no_match_via_branch(
    pipeline: WFSTPipeline, raw: str,
) -> None:
    """Neither month-word nor numeric branch may match these."""
    assert pipeline.normalize_span(raw, "date_monthword") is None
    assert pipeline.normalize_span(raw, "date_numeric") is None


@pytest.mark.parametrize("raw", _ADVERSARIAL_DATE)
def test_adversarial_no_match_via_pipeline(
    pipeline: WFSTPipeline, raw: str,
) -> None:
    """Neither tenant policy may auto-rewrite these."""
    for order in ("DMY", "MDY", "YMD"):
        res = pipeline.normalize_date(raw, date_order=order)
        assert res.canonical is None, (raw, order)


# ---------------------------------------------------------------------------
# Classifier wrapper — sanity check on the NeMo-tagged surface.
# ---------------------------------------------------------------------------

def test_classifier_emits_nemo_tag(pipeline: WFSTPipeline) -> None:
    out = pipeline.classify_span("बारह मई दो हज़ार छब्बीस", "date")
    assert out == 'date { value: "12/05/2026" }'


# ---------------------------------------------------------------------------
# Coverage budget — keep the test count honest.
# ---------------------------------------------------------------------------

def test_total_case_count_meets_minimum() -> None:
    total = len(_MONTHWORD_GOLD) + len(_NUMERIC_DMY_GOLD)
    assert total >= 100, total
    assert len(_AMBIGUOUS_NUMERIC_INPUTS) >= 10
    assert len(_ADVERSARIAL_DATE) >= 15
