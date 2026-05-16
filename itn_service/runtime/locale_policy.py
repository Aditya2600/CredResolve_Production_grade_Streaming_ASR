"""Per-tenant locale policy loader.

Reads the ``tenants`` section of ``configs/locales.yaml`` and exposes a
small read-only table that the date / currency runtime consults to
decide whether ambiguous spans (e.g. ``12/05/2026``) are safe to
auto-normalise.

The module deliberately does **no** parsing work itself — it just
answers "what is this tenant's policy". The numeric-date guard lives in
:mod:`itn_service.runtime.normalizer` (and tests under
``tests/runtime/test_locale_policy.py``).

Per ``CONTRIBUTING.md``: ``configs/thresholds.yaml`` is the single
source for confidence gates; ``configs/locales.yaml`` is the single
source for locale and tenant policy. This module reads the latter only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

import yaml


# ---------------------------------------------------------------------------
# Parsed policy table.
# ---------------------------------------------------------------------------

# Allowed values for `date_order`. Keep tight — ICU/CLDR records other
# orders ("YDM" etc.) but they aren't used by any tenant we serve, so
# we reject them at load time rather than silently passing them through.
_ALLOWED_DATE_ORDERS: Final[frozenset[str]] = frozenset({"DMY", "MDY", "YMD"})


@dataclass(frozen=True)
class TenantPolicy:
    """Effective locale policy for one tenant."""

    tenant_id: str
    region: str
    date_order: str             # "DMY" | "MDY" | "YMD"
    currency: str               # ISO 4217


@dataclass(frozen=True)
class CurrencyCues:
    """Per-locale money-cue policy.

    ``accept`` is the recognition vocabulary the money WFST consumes
    (any surface form here flips the segment to the money branch).
    ``canonical`` is the single output prefix the canonical text
    carries regardless of which cue was recognised. ``paise_accept``
    is the subunit vocabulary (paise / poysha / paisa / etc.).

    All lists are stored already folded for the language's working-
    copy transformations (NFC for all; Gurmukhi bindi-fold for ``pa``).
    The caller is the money grammar, not the regex prefilter — the
    prefilter still pattern-matches on the canonical symbol directly.
    """

    accept: tuple[str, ...]
    canonical: str
    paise_accept: tuple[str, ...] = ()


@dataclass(frozen=True)
class TenantPolicyTable:
    """Parsed ``configs/locales.yaml`` tenant section."""

    default_tenant_id: str
    tenants: Mapping[str, TenantPolicy]
    # Per-locale currency-cue table. Keyed by the locale id as it
    # appears under ``locales:`` (e.g. ``hi``, ``mr``, ``bn``,
    # ``bn-IN``, ``bn-BD``, ``gu``, ``pa``). Built from the
    # ``currency_cues`` block in ``locales.yaml``.
    locale_currency_cues: Mapping[str, CurrencyCues] = field(
        default_factory=dict
    )

    def for_tenant(self, tenant_id: str | None) -> TenantPolicy:
        """Resolve a tenant id to its effective policy.

        ``None`` and unknown ids both fall back to the default tenant.
        We deliberately do not raise on unknown ids — production calls
        often arrive with stale or misspelled tenant routing keys, and
        the safest fallback is "use the default policy" rather than
        crash the call.
        """
        if tenant_id is not None and tenant_id in self.tenants:
            return self.tenants[tenant_id]
        return self.tenants[self.default_tenant_id]

    def currency_cues_for(
        self,
        lang: str,
        tenant_id: str | None = None,
    ) -> CurrencyCues | None:
        """Resolve the currency-cue policy for a (lang, tenant) pair.

        Resolution order, first hit wins:

        1. ``f"{lang}-{tenant.region}"`` — regional override (e.g.
           ``bn-IN`` / ``bn-BD``). Lets a Bangladesh tenant pick the
           ৳ canonical even though the script-routed lang is ``bn``.
        2. ``lang`` — the base locale entry.
        3. ``None`` — the lang has no cue table; the money grammar
           is expected to fall back to a generic ``₹ + Rs + INR``
           policy (the runtime default for INR-tenant Indic locales).

        Returns ``None`` when neither key is present. Callers are
        expected to be defensive: a missing entry means "use the
        runtime default", not "money is disabled".
        """
        # 1. region-qualified lookup, only if the tenant resolves to
        # a non-empty region.
        if tenant_id is not None:
            tenant = self.for_tenant(tenant_id)
            if tenant.region:
                regional = f"{lang}-{tenant.region}"
                hit = self.locale_currency_cues.get(regional)
                if hit is not None:
                    return hit
        # 2. base locale.
        return self.locale_currency_cues.get(lang)


# ---------------------------------------------------------------------------
# YAML loader.
# ---------------------------------------------------------------------------

def _default_locales_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "locales.yaml"


def load_locale_policy(path: Path | None = None) -> TenantPolicyTable:
    """Load and parse the tenant section of ``locales.yaml``."""
    p = path if path is not None else _default_locales_path()
    with p.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    defaults: dict[str, Any] = data.get("defaults") or {}
    tenants_raw: dict[str, Any] = data.get("tenants") or {}
    locales_raw: dict[str, Any] = data.get("locales") or {}

    default_tenant_id = str(defaults.get("tenant", "default"))
    if default_tenant_id not in tenants_raw:
        raise ValueError(
            f"defaults.tenant={default_tenant_id!r} not found in tenants: "
            f"{sorted(tenants_raw)}"
        )

    fallback_currency = str(defaults.get("currency", "INR"))
    fallback_date_order = str(defaults.get("date_order", "DMY"))

    parsed: dict[str, TenantPolicy] = {}
    for tid, cfg in tenants_raw.items():
        cfg = cfg or {}
        date_order = str(cfg.get("date_order", fallback_date_order))
        if date_order not in _ALLOWED_DATE_ORDERS:
            raise ValueError(
                f"tenant {tid!r}: unsupported date_order={date_order!r}; "
                f"expected one of {sorted(_ALLOWED_DATE_ORDERS)}"
            )
        parsed[str(tid)] = TenantPolicy(
            tenant_id=str(tid),
            region=str(cfg.get("region", "")),
            date_order=date_order,
            currency=str(cfg.get("currency", fallback_currency)),
        )

    cues: dict[str, CurrencyCues] = {}
    for loc_id, cfg in locales_raw.items():
        cfg = cfg or {}
        block = cfg.get("currency_cues")
        if not block:
            continue
        accept_raw = block.get("accept") or []
        canonical = block.get("canonical")
        if not canonical or not accept_raw:
            raise ValueError(
                f"locale {loc_id!r}: currency_cues requires both "
                f"`accept` (non-empty) and `canonical`"
            )
        paise_raw = block.get("paise_accept") or []
        cues[str(loc_id)] = CurrencyCues(
            accept=tuple(str(a) for a in accept_raw),
            canonical=str(canonical),
            paise_accept=tuple(str(p) for p in paise_raw),
        )

    return TenantPolicyTable(
        default_tenant_id=default_tenant_id,
        tenants=parsed,
        locale_currency_cues=cues,
    )


__all__ = [
    "CurrencyCues",
    "TenantPolicy",
    "TenantPolicyTable",
    "load_locale_policy",
]
