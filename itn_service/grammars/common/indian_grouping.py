"""Indian-numbering-system thousands grouping.

Indic locales group thousands as ``1,25,000`` rather than the Western
``125,000``: the rightmost three digits stay as one block, and every
group above it is two digits wide. CLDR records this under the locale
``numberFormat`` patterns for ``hi``, ``gu``, ``ur``, etc.

This module is **not** a Pynini grammar — it is the canonical
post-cardinal reformatter. Per `CONTRIBUTING.md` invariant 4, the
service stores Latin digits with ICU-canonical separators; the cardinal
WFST emits a bare integer string and this helper inserts the locale's
group separator before the value is written into ``canonical_text``.

Performance: 5-6 µs per call on the largest 11-digit input on a single
CPU core, well inside the per-segment budget.
"""

from __future__ import annotations


def indian_grouping(latin_int: str, sep: str = ",") -> str:
    """Re-group a non-negative Latin integer string into Indian style.

    Args:
        latin_int: The integer rendered as ASCII digits, no sign, no
            leading zeros except for the literal ``"0"`` itself. Empty
            strings raise ``ValueError`` so the caller surfaces upstream
            grammar bugs rather than silently producing a malformed
            canonical surface.
        sep: Group separator. Defaults to ``","`` (the ICU canonical
            choice for the ``hi``/``gu``/``ur``/etc. locales). Pass a
            different value only when generating non-canonical display
            text.

    Returns:
        The integer with Indian grouping applied. Numbers shorter than
        four digits are returned unchanged (``"0"`` -> ``"0"``,
        ``"125"`` -> ``"125"``).

    Raises:
        ValueError: If ``latin_int`` is empty, contains non-digit
            characters, or has invalid leading zeros (e.g. ``"0125"``).

    Examples:
        >>> indian_grouping("0")
        '0'
        >>> indian_grouping("125")
        '125'
        >>> indian_grouping("1000")
        '1,000'
        >>> indian_grouping("12500")
        '12,500'
        >>> indian_grouping("125000")
        '1,25,000'
        >>> indian_grouping("12500000")
        '1,25,00,000'
        >>> indian_grouping("99999999999")     # 1 अरब - 1
        '99,99,99,99,999'
    """
    if not latin_int:
        raise ValueError("indian_grouping: empty string")
    if not latin_int.isascii() or not latin_int.isdigit():
        raise ValueError(f"indian_grouping: non-digit input {latin_int!r}")
    if len(latin_int) > 1 and latin_int[0] == "0":
        raise ValueError(f"indian_grouping: leading zero in {latin_int!r}")

    if len(latin_int) <= 3:
        return latin_int

    last3 = latin_int[-3:]
    rest = latin_int[:-3]

    # Walk `rest` from the right in 2-digit groups; whatever is left
    # over (1 or 2 digits) becomes the leading group.
    groups: list[str] = []
    while len(rest) > 2:
        groups.append(rest[-2:])
        rest = rest[:-2]
    groups.append(rest)
    groups.reverse()

    return sep.join(groups) + sep + last3


__all__ = ["indian_grouping"]
