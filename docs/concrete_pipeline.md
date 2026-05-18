# Concrete ITN pipeline for live ASR

**Status:** executable target shape, with current-state notes  
**As of:** 2026-05-18  
**Companion docs:** [implementation_blueprint_INR.md](implementation_blueprint_INR.md), [itn_live_path_gap_analysis.md](itn_live_path_gap_analysis.md), [flowchart_ITN.mmd](flowchart_ITN.mmd)

This is the smallest useful description of the live inverse text normalisation path: what enters it, what is allowed to change, when a rewrite is safe, and what must be emitted for downstream systems to trust it.

The service contract is intentionally three-surfaced:

```text
raw_text        = exactly what ASR emitted; immutable audit surface
canonical_text  = stable machine-facing text after accepted rewrites
display_text    = UI-facing rendering of canonical_text
spans[]         = provenance surface for located / accepted rewrites
```

---

## At a glance

```text
ASR hypothesis
  text + tokens + is_final + lang_hint + locale_policy
                         |
                         v
                update_stability(text)
                         |
             +-----------+-----------+
             |                       |
   unstable partial              final or stable partial
             |                       |
             v                       v
   emit raw on all surfaces     working_copy(raw_text)
                                     |
                                     v
                           route_language(text, hint)
                                     |
                                     v
                           classify / rewrite spans
                                     |
                                     v
                              confidence_gate
                                     |
                                     v
                             apply accepted spans
                                     |
                                     v
                           render display surface
                                     |
                                     v
      emit { raw_text, canonical_text, display_text, spans, deferred, lang, script }
```

### Current repo truth

Today, the default request path is still deliberately conservative:

```text
working_copy
   -> route_language
   -> regex_prefilter only
   -> confidence_gate
   -> apply_spans(canonical == raw)
```

So the scaffolding is live, but the default classifier does **not** yet perform user-visible WFST rewrites. Until the WFST-backed classifier is wired into the request path, `canonical_text` and `display_text` remain equal to the incoming text for normal requests. See [itn_live_path_gap_analysis.md](itn_live_path_gap_analysis.md) for the gap and staging plan.

---

## Segment policy

| Input state | What may happen | Why |
|---|---|---|
| Unstable partial | Return raw text on every surface, `deferred=true` | Avoid UI churn while the decoder is still revising itself |
| Stable partial | Only low-risk classes may render: `cardinal`, `money`, `percent` | Prefix-safe updates are useful; risky rewrites are not |
| Final segment | Full classification, rewrite, gate, splice, display rendering | Final hypotheses have the context needed for conservative ITN |

The organising principle is simple: **partials may improve readability; finals may change meaning only when the evidence is strong enough.**

---

## Final-segment pipeline

```text
1. Preserve
   raw_text is copied verbatim from the decoder and never mutated.

2. Prepare
   working = working_copy(raw_text)
   - Unicode-safe internal form for matching and splicing
   - separate from the immutable audit surface

3. Route
   route = route_language(working, asr_hint=lang_hint)
   - ASR hint first
   - script fallback when hints are absent

4. Locate and propose
   spans = classifier(working, route.lang)
   - current default: regex_prefilter only
   - intended live path: prefilter -> class-specific WFST / formatter rewrite

5. Gate
   for each span:
       reject if ambiguous
       combine classifier confidence, ASR confidence, cue policy, and partial/final policy
       if unsafe: canonical = raw and record fallback_reason

6. Assemble
   canonical_text = apply_spans(working, safe_spans)

7. Render
   display_text = display_renderer(canonical_text, route.lang)
   - currently a passthrough
   - later: locale-aware digits / punctuation without mutating canonical storage

8. Emit
   SegmentResult(raw_text, canonical_text, display_text, spans, deferred, lang, script, itn_version)
```

---

## Rewrite loop

Once the WFST-backed path is enabled, the core decision loop should behave like this:

```python
for span in candidate_spans:
    candidate = rewrite(span.raw, cls=span.cls, lang=lang, policy=locale_policy)

    if candidate is None:
        emit_span(
            raw=span.raw,
            canonical=span.raw,
            ambiguous=True,
            fallback_reason="no_parse",
        )
        continue

    confidence = score(
        candidate=candidate,
        raw=span.raw,
        cls=span.cls,
        context=context,
        asr_confidence=asr_confidence,
    )

    if confidence < threshold_for(span.cls) or requires_missing_cue(span):
        emit_span(
            raw=span.raw,
            canonical=span.raw,
            fallback_reason="gate_reject",
        )
    else:
        emit_span(
            raw=span.raw,
            canonical=candidate,
            conf=confidence,
        )
```

The important asymmetry is intentional: **a missed rewrite is tolerable; a wrong rewrite is expensive.**

---

## Output contract

| Field | Meaning | Mutation policy |
|---|---|---|
| `raw_text` | Verbatim ASR hypothesis | Never rewritten |
| `canonical_text` | Stable machine-facing transcript | Only accepted spans are spliced in |
| `display_text` | UI-facing rendering | Derived from canonical text |
| `spans[]` | Rewrite provenance | Includes class, raw, canonical, rule id, confidence, offsets, ambiguity, fallback reason |
| `deferred` | Whether a partial was intentionally not normalised | `true` for unstable partials and request-level fallback |
| `lang`, `script` | Routed language metadata | Derived inside the service |

This contract is what lets product, analytics, and review workflows each get the surface they need without collapsing them into one lossy transcript.

---

## Failure semantics

The pipeline should fail locally before it fails globally:

```text
one unsafe span        -> keep that span raw, preserve telemetry
one formatter failure  -> keep that span raw, continue the segment
one pipeline exception -> emit raw_text on every surface, deferred=true
ITN outage             -> transcription still survives
```

The gRPC service already enforces the request-level guarantee: ITN failure must never break transcription delivery. The next production step is to make the same conservatism true span-by-span once live rewrites are enabled.

---

## Concrete example

```text
ASR final:
  "kal 12-05-2026 ko ₹500 bhejna"

Candidate spans:
  [date: "12-05-2026", money: "₹500"]

If tenant policy confirms DMY:
  canonical_text = "kal 12/05/2026 ko ₹500 bhejna"

If date order is unresolved:
  canonical_text = "kal 12-05-2026 ko ₹500 bhejna"
  spans[date].fallback_reason = "ambiguous_numeric_date"
```

The amount may still be accepted while the date is refused. That selectivity is the whole point of span-level provenance.

---

## Non-negotiable invariants

1. `raw_text` is immutable.
2. No generative model sits in the hot path.
3. No FAR compilation happens on requests.
4. Canonical storage stays stable; locale shaping belongs in `display_text`.
5. Ambiguity falls back to raw, not to guesswork.
6. ITN must degrade transcript quality gracefully, never service availability.

If a future implementation preserves these six properties, the machinery underneath can evolve without changing the trust model above it.
