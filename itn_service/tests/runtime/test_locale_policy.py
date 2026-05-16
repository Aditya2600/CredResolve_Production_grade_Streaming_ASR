"""Tests for ``runtime.locale_policy``.

Covers two surfaces:

* ``for_tenant`` — date-order + currency fallback (existing behaviour).
* ``currency_cues_for`` — the bn-IN / bn-BD policy split added with
  Bengali / Gujarati / Punjabi language support. The bn case is the
  motivating one: Bengali speakers in India use either টাকা or রুপি
  with ₹ canonical output; Bengali speakers in Bangladesh use টাকা
  (or ৳ / BDT / Tk) with ৳ canonical output. Resolution must walk
  ``{lang}-{tenant.region}`` first, then ``{lang}``.
"""

from __future__ import annotations

import pytest

from itn_service.runtime.locale_policy import (
    CurrencyCues,
    load_locale_policy,
)


@pytest.fixture(scope="module")
def policy():  # type: ignore[no-untyped-def]
    return load_locale_policy()


# ---------------------------------------------------------------------------
# Tenant resolution.
# ---------------------------------------------------------------------------


def test_default_tenant_resolves_to_default_when_none(policy) -> None:  # type: ignore[no-untyped-def]
    p = policy.for_tenant(None)
    assert p.tenant_id == policy.default_tenant_id


def test_unknown_tenant_falls_back_to_default(policy) -> None:  # type: ignore[no-untyped-def]
    p = policy.for_tenant("does-not-exist")
    assert p.tenant_id == policy.default_tenant_id


# ---------------------------------------------------------------------------
# Currency cue resolution — the bn-IN / bn-BD motivator.
# ---------------------------------------------------------------------------


def test_bn_base_locale_accepts_taka_and_rupi_with_rupee_canonical(
    policy,  # type: ignore[no-untyped-def]
) -> None:
    """For Indian tenants the Bengali money grammar must accept either
    টাকা (colloquial rupee word) or রুপি (literal rupee) and emit ₹
    as canonical. This is the bn-IN default in the locales.yaml block
    and the test that pins it in place."""
    cues = policy.currency_cues_for("bn")
    assert cues is not None
    assert "টাকা" in cues.accept
    assert "রুপি" in cues.accept
    assert cues.canonical == "₹"


def test_bn_bd_overrides_canonical_to_taka_symbol(
    policy,  # type: ignore[no-untyped-def]
) -> None:
    """The Bangladesh variant canonicalises to ৳, not ₹, even though
    টাকা is accepted in both — the canonical follows the tenant's
    ISO currency, not the spoken cue."""
    cues = policy.currency_cues_for("bn-BD")
    assert cues is not None
    assert cues.canonical == "৳"
    assert "টাকা" in cues.accept
    # bn-BD should NOT inherit the rupee-only forms from bn — the
    # accept list is a hard override, not a merge.
    assert "₹" not in cues.accept
    assert "রুপি" not in cues.accept


def test_currency_cues_resolution_prefers_region_qualified_lookup(
    policy,  # type: ignore[no-untyped-def]
) -> None:
    """An Indian tenant looking up Bengali cues must resolve to the
    base ``bn`` block (₹ canonical). A US tenant likewise — there is
    no ``bn-US`` override, so we fall back to base ``bn``. Only an
    explicit ``bn-BD`` lookup (or, in the future, a tenant whose
    region is ``BD``) gets the ৳ canonical."""
    in_cues = policy.currency_cues_for("bn", tenant_id="acme_in")
    assert in_cues is not None and in_cues.canonical == "₹"

    us_cues = policy.currency_cues_for("bn", tenant_id="acme_us")
    assert us_cues is not None and us_cues.canonical == "₹"


def test_languages_without_cue_block_resolve_to_none(
    policy,  # type: ignore[no-untyped-def]
) -> None:
    """Languages that have not yet had cue policy written (ta, te,
    kn, ml, ur as of this stage) must resolve to ``None`` so the
    money grammar can fall back to its runtime default rather than
    crashing."""
    for lang in ("ta", "te", "kn", "ml", "ur"):
        assert policy.currency_cues_for(lang) is None


@pytest.mark.parametrize(
    "lang,canonical_symbol",
    [
        ("hi", "₹"),
        ("mr", "₹"),
        ("gu", "₹"),
        ("pa", "₹"),
    ],
)
def test_indic_inr_locales_canonicalise_to_rupee(
    policy, lang: str, canonical_symbol: str,  # type: ignore[no-untyped-def]
) -> None:
    """All INR-tenant Indic locales (hi, mr, gu, pa) canonicalise to
    ₹. This is the invariant the money grammar depends on; if a
    future locale lands with a different canonical symbol it must
    update its money tests in lockstep."""
    cues = policy.currency_cues_for(lang)
    assert cues is not None
    assert cues.canonical == canonical_symbol


def test_currency_cues_block_validates_required_fields(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A ``currency_cues`` block without ``accept`` or ``canonical``
    must fail at load time, not silently — the money grammar relies
    on both being present."""
    bad = tmp_path / "bad_locales.yaml"
    bad.write_text(
        "defaults:\n"
        "  tenant: t1\n"
        "tenants:\n"
        "  t1:\n"
        "    region: IN\n"
        "    date_order: DMY\n"
        "    currency: INR\n"
        "locales:\n"
        "  hi:\n"
        "    name: Hindi\n"
        "    currency_cues:\n"
        "      accept: []\n"  # empty
        "      canonical: ₹\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="currency_cues"):
        load_locale_policy(bad)


def test_currency_cues_dataclass_is_frozen() -> None:
    """``CurrencyCues`` must be immutable so callers can cache it
    without defensive copies."""
    cues = CurrencyCues(accept=("₹",), canonical="₹", paise_accept=())
    with pytest.raises(Exception):
        cues.canonical = "$"  # type: ignore[misc]
