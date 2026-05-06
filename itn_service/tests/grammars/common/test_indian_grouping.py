"""Unit + property tests for ``grammars.common.indian_grouping``.

The function is the deterministic post-cardinal reformatter that
inserts the Indian thousands separator (``1,25,000``); per
``CONTRIBUTING.md`` invariant 4 it is the only path between a Latin
integer string and the canonical separator surface.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from itn_service.grammars.common.indian_grouping import indian_grouping


# --- known mappings ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", "0"),
        ("1", "1"),
        ("9", "9"),
        ("99", "99"),
        ("125", "125"),
        ("999", "999"),
        ("1000", "1,000"),
        ("1500", "1,500"),
        ("9999", "9,999"),
        ("12500", "12,500"),
        ("99999", "99,999"),
        ("100000", "1,00,000"),
        ("125000", "1,25,000"),
        ("999999", "9,99,999"),
        ("1000000", "10,00,000"),
        ("12500000", "1,25,00,000"),
        ("100000000", "10,00,00,000"),
        ("999999999", "99,99,99,999"),
        ("1000000000", "1,00,00,00,000"),
        ("99999999999", "99,99,99,99,999"),
    ],
)
def test_known_groupings(raw: str, expected: str) -> None:
    assert indian_grouping(raw) == expected


# --- error contract ----------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",            # empty
        "012",         # leading zero (semantically ambiguous)
        "00",          # leading zero
        "abc",         # non-digit
        "12.5",        # decimal point
        "-12",         # sign
        "1 000",       # whitespace
        "१२५",        # native digits — must be normalised before reaching us
    ],
)
def test_rejects_invalid_input(bad: str) -> None:
    with pytest.raises(ValueError):
        indian_grouping(bad)


# --- separator override ------------------------------------------------------


def test_custom_separator() -> None:
    # Display layer might want non-canonical separators (e.g. for an
    # accessibility skin); the function honours `sep`.
    assert indian_grouping("125000", sep=" ") == "1 25 000"


# --- property test (hypothesis) ---------------------------------------------


@given(n=st.integers(min_value=0, max_value=10**11))
@settings(max_examples=500, deadline=None)
def test_roundtrip_strip_commas(n: int) -> None:
    """Removing the inserted separators recovers the original integer."""
    s = str(n)
    grouped = indian_grouping(s)
    assert grouped.replace(",", "") == s


@given(n=st.integers(min_value=0, max_value=10**11))
@settings(max_examples=500, deadline=None)
def test_first_group_is_one_or_two_digits(n: int) -> None:
    """For numbers ≥ 1,000 (i.e. those that get grouped), every group
    *except* the rightmost must be 1 or 2 digits, and the rightmost
    must be exactly 3 digits."""
    s = str(n)
    grouped = indian_grouping(s)
    if "," not in grouped:
        # No grouping → the whole string is short (< 4 digits).
        assert len(s) <= 3
        return
    parts = grouped.split(",")
    assert len(parts[-1]) == 3
    for p in parts[:-1]:
        assert 1 <= len(p) <= 2
