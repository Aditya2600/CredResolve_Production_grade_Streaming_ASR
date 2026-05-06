# Contributing to itn_service

This service is a deterministic, WFST-first inverse text normalisation
layer for live Indic telephony ASR. The full design lives in
[../docs/implementation_bluprint_INR.md](../docs/implementation_bluprint_INR.md).
Read it before opening a non-trivial PR.

## Invariants

These are non-negotiable. A change that breaks any of them must be
explicitly justified in the PR description and signed off by a
maintainer; otherwise it will be reverted.

1. **`raw_asr_text` is never modified.**
   The decoder's verbatim output is preserved on `SegmentResult.raw_text`
   for audit, debugging, and legal review. Every rewrite produces a new
   surface (`canonical_text`, `display_text`); `raw_text` is read-only
   for every stage downstream of the ASR gateway.

2. **No LLM calls in the live path.**
   No seq2seq, small-LLM, or hosted-API call may execute inside the
   request-handling code path. LLMs are off-path tooling for grammar
   authoring, error analysis, and offline candidate generation only.
   Hallucinated formatting errors are unrecoverable in finance and
   healthcare contexts.

3. **No FAR compilation in the request path.**
   Pynini grammars are compiled to FAR archives at build / deploy time
   and loaded once at process start (`runtime/far_cache.py`). Per-request
   `pynini.compile`, `pynini.string_map(...)` against runtime data, or
   any other graph-construction call inside a hot path is a regression.

4. **Storage is always Latin digits + ICU-canonical separators.**
   `canonical_text` and persisted span fields use Latin digits and
   ICU/CLDR-canonical separators, regardless of the call's locale.
   Native-digit shaping (Devanagari, Bengali, Tamil, Arabic-Indic, etc.)
   happens only in `display_text` via the locale renderer. This keeps
   downstream search, NER, analytics, and CRM ingestion stable across
   locales.

## What goes where

- `configs/thresholds.yaml` is the **single source of truth** for
  confidence gating. Don't sprinkle thresholds across grammar code or
  runtime helpers — read them from this file.
- `configs/policy.yaml` holds output-policy invariants and the
  endpoint-normalisation default.
- `configs/locales.yaml` holds per-language script, numbering systems,
  date order, currency, and bidi flags.
- `runtime/contract.py` is the only public contract. Callers depend on
  `Token`, `Span`, and `SegmentResult`; everything else in `runtime/` is
  internal.
- `grammars/<lang>/` follows the `classify -> parse -> generate
  permutations -> verbalise` pattern. Shared lexica live under
  `grammars/common/`.

## Lint, type, test

```bash
pip install -e .[dev]
ruff check .
mypy
pytest
```

`mypy` runs in strict mode on `runtime/` only — that is the type-checked
core. Grammar files and tools are excluded for now.

## Adding a language

Copy a shared class skeleton, swap only language-specific lexica
(connectors, digit maps, month names, ambiguity rules), and add gold
tests under `tests/gold/<lang>/`. Do **not** rewrite each language from
scratch.
