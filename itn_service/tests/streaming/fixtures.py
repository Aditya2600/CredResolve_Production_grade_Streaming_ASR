"""Twenty hand-authored partial -> final streams.

Each :class:`Stream` is a list of :class:`Hyp` (one hypothesis) plus a
short ``name`` for failure messages. The hypothesis sequence within a
``Stream`` represents what an ASR gateway emits for **one segment**:
the raw text grows monotonically (each partial text is a prefix of the
next), and the last hypothesis is ``is_final=True``.

That monotonicity matters for the no-flicker invariant: once a span is
emitted at offset ``[a, b)``, the same substring is still present at
the same offset in every later partial — so any disappearance is a
flicker bug, not a legitimate revision. Real streaming ASR systems
*do* revise partials; we deliberately fix that out here so the
invariant we test is precise. Revision behaviour belongs in a separate
test (the ``stable_count`` mechanism inside ``StreamState`` is what
handles it).

Coverage targets the regex prefilter classes that ``normalize_segment``
runs by default:

* AMOUNT_LATN  (5 streams across Hindi, Marathi, English, Bengali, Tamil)
* PHONE_LATN   (4 streams: bare 10-digit, +91-prefixed, with separators, OTP framing)
* PERCENT_LATN (2 streams)
* DATE_NUMERIC (2 streams)
* TIME_NUMERIC (2 streams)
* IFSC / PAN / AADHAAR (3 streams)
* EMAIL / URL  (2 streams)

Twenty total. The text is realistic-shaped but not lifted from any
particular call; numbers / dates / IDs are synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from itn_service.runtime.contract import Token


@dataclass(frozen=True)
class Hyp:
    """One ASR hypothesis at a point in the call."""

    text: str
    is_final: bool
    tokens: tuple[Token, ...] = ()
    lang_hint: str | None = None


@dataclass(frozen=True)
class Stream:
    """A named partial -> final sequence."""

    name: str
    hyps: tuple[Hyp, ...]
    lang_hint: str | None = None


def _grow(prefixes: Sequence[str], final: str, lang: str | None = None) -> Stream:
    """Build a monotonically-growing stream from prefixes + a final text.

    Each entry in ``prefixes`` becomes one partial; ``final`` is the
    last (is_final=True) hypothesis. ``lang_hint`` is attached to
    every hypothesis (none-by-default).
    """
    hyps = tuple(
        Hyp(text=t, is_final=False, lang_hint=lang) for t in prefixes
    ) + (Hyp(text=final, is_final=True, lang_hint=lang),)
    return Stream(name="", hyps=hyps, lang_hint=lang)


def _named(stream: Stream, name: str) -> Stream:
    return Stream(name=name, hyps=stream.hyps, lang_hint=stream.lang_hint)


STREAMS: tuple[Stream, ...] = (
    # --- AMOUNT_LATN (5) ---------------------------------------------------
    _named(_grow(
        ["please pay", "please pay ₹", "please pay ₹1,", "please pay ₹1,250"],
        "please pay ₹1,250 today",
        lang="en",
    ), "amount_en_simple"),

    _named(_grow(
        ["aap ko", "aap ko ₹", "aap ko ₹2,50,000", "aap ko ₹2,50,000 jama"],
        "aap ko ₹2,50,000 jama karne hain",
        lang="hi",
    ), "amount_hi_lakh"),

    _named(_grow(
        ["amount is Rs.", "amount is Rs.1500", "amount is Rs.1500."],
        "amount is Rs.1500.00",
        lang="en",
    ), "amount_en_rs_decimal"),

    _named(_grow(
        ["total ₹", "total ₹99", "total ₹999.99"],
        "total ₹999.99 only",
        lang="en",
    ), "amount_en_decimal"),

    _named(_grow(
        ["आपका बिल", "आपका बिल ₹", "आपका बिल ₹1,200"],
        "आपका बिल ₹1,200 है",
        lang="hi",
    ), "amount_hi_devanagari"),

    # --- PHONE_LATN (4) ----------------------------------------------------
    _named(_grow(
        ["call me on", "call me on 9876543210"],
        "call me on 9876543210 please",
        lang="en",
    ), "phone_bare10"),

    _named(_grow(
        ["my number is", "my number is +91", "my number is +91 9876543210"],
        "my number is +91 9876543210",
        lang="en",
    ), "phone_plus91"),

    _named(_grow(
        ["OTP sent to", "OTP sent to 09876543210"],
        "OTP sent to 09876543210 now",
        lang="en",
    ), "phone_leading_zero"),

    _named(_grow(
        ["dial 8765432109", "dial 8765432109 for"],
        "dial 8765432109 for support",
        lang="en",
    ), "phone_alt"),

    # --- PERCENT_LATN (2) --------------------------------------------------
    _named(_grow(
        ["interest is", "interest is 12.5", "interest is 12.5%"],
        "interest is 12.5% per annum",
        lang="en",
    ), "percent_decimal"),

    _named(_grow(
        ["discount of", "discount of 25%"],
        "discount of 25% on EMI",
        lang="en",
    ), "percent_int"),

    # --- DATE_NUMERIC (2) --------------------------------------------------
    _named(_grow(
        ["due on", "due on 12/05/2026"],
        "due on 12/05/2026 sharp",
        lang="en",
    ), "date_dmy_slash"),

    _named(_grow(
        ["dated", "dated 01-04-2025"],
        "dated 01-04-2025 vide",
        lang="en",
    ), "date_dmy_dash"),

    # --- TIME_NUMERIC (2) --------------------------------------------------
    _named(_grow(
        ["meeting at", "meeting at 17:30"],
        "meeting at 17:30 today",
        lang="en",
    ), "time_24h"),

    _named(_grow(
        ["call at", "call at 5:30 PM"],
        "call at 5:30 PM sharp",
        lang="en",
    ), "time_12h_pm"),

    # --- IFSC / PAN / AADHAAR (3) ------------------------------------------
    _named(_grow(
        ["IFSC code", "IFSC code SBIN0001234"],
        "IFSC code SBIN0001234 verified",
        lang="en",
    ), "ifsc_simple"),

    _named(_grow(
        ["PAN is", "PAN is ABCDE1234F"],
        "PAN is ABCDE1234F filed",
        lang="en",
    ), "pan_simple"),

    _named(_grow(
        ["Aadhaar number", "Aadhaar number 2345 6789 0123"],
        "Aadhaar number 2345 6789 0123 confirmed",
        lang="en",
    ), "aadhaar_spaced"),

    # --- EMAIL / URL (2) ---------------------------------------------------
    _named(_grow(
        ["email me at", "email me at customer@bank.in"],
        "email me at customer@bank.in for the receipt",
        lang="en",
    ), "email_simple"),

    _named(_grow(
        ["visit", "visit https://bank.in/pay", "visit https://bank.in/pay/9876"],
        "visit https://bank.in/pay/9876 to settle",
        lang="en",
    ), "url_with_path"),
)


assert len(STREAMS) == 20, f"expected 20 fixture streams, got {len(STREAMS)}"
