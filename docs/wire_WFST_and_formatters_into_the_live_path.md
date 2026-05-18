# Stage 1 execution note — wire WFST + formatters into the live path

**Status:** partially implemented, not yet live by default  
**As of:** 2026-05-18  
**Companion docs:** [implementation_blueprint_INR.md](implementation_blueprint_INR.md), [itn_live_path_gap_analysis.md](itn_live_path_gap_analysis.md), [concrete_pipeline.md](concrete_pipeline.md)

Stage 1 is the first user-visible ITN boundary. Before it, the service locates spans and preserves provenance but the default request path still returns effectively raw text. After it, selected classes are actually rewritten on final segments while preserving the same conservative fallback model.

The codebase is now **partway through Stage 1**:

- the additive WFST classifier surface exists,
- lazy per-language WFST construction exists,
- focused classifier tests exist,
- but the gRPC request path still defaults to `default_classifier`,
- and the orchestration work that makes the new classifier safe enough to flip live is still incomplete.

In other words: the engine has been built on the bench; it has not yet been bolted into the vehicle.

---

## Target request path

```text
final ASR text
    |
    v
working_copy(raw_text)
    |
    v
route_language(working_text, lang_hint)
    |
    v
make_wfst_classifier(tenant_policy)
    |
    +--> regex_prefilter(working_text)
    |
    +--> dispatch located spans
    |       phone            -> deterministic formatter
    |       amount           -> WFST money
    |       percent / time   -> WFST class rewrite
    |       date             -> policy-aware WFST date
    |                            \-> guarded dateparser fallback only when allowed
    |
    v
detect_self_corrections(working_text, spans)
    |
    v
confidence_gate
    |
    v
apply_spans -> canonical_text -> display renderer
```

The classifier remains a **per-tenant closure**:

```python
classifier = make_wfst_classifier(tenant_policy)
spans = classifier(working_text, lang)
```

That keeps the public classifier protocol stable — `(working_text, lang) -> list[Span]` — while still allowing date policy and later tenant-specific rewrite policy to participate.

---

## Current implementation state

| Area | State on 2026-05-18 | Evidence |
|---|---|---|
| Lazy WFST factory | Done | `runtime/wfst_factory.py` |
| Additive WFST classifier | Done | `runtime/wfst_classifier.py` |
| Amount → money bridge | Done inside classifier | `_PREFILTER_TO_WFST_CLASS` |
| Phone formatter integration | Done | `fmt.phone` branch |
| Policy-aware date branch | Done | `normalize_date(..., date_order=...)` |
| Guarded dateparser fallback | Done | only after `has_date_cue(...)` |
| Focused unit tests | Done | `tests/runtime/test_wfst_classifier.py`, `test_wfst_factory.py` |
| Live request-path flip | Not done | `grpc_server.py` still injects `default_classifier` |
| Self-correction orchestration | Not done | `normalizer.py` does not call `detect_self_corrections` |
| Cue semantics for rewritten spans | Not done | `_span_has_lex_cue` only recognises `prefilter.*` |
| Threshold taxonomy cleanup | Not done | runtime emits `money`; thresholds still configure `currency` |
| Feature flag / rollback switch | Not done | `configs/policy.yaml` has no WFST classifier flag |
| CI with real WFST deps | Not done | workflow still documents “scaffolding-stage CI” |
| Request-path regression gate | Not done | current gold tests exercise `WFSTPipeline` directly, not the default service path |

### What is already true

`runtime/wfst_classifier.py` now rewrites the subset intended for the first rollout:

```text
phone  -> formatter
amount -> money
percent / time -> WFST
date -> WFST, then guarded dateparser fallback
```

It also records failed attempts as raw spans with explicit reasons such as:

```text
fmt_no_parse
wfst_unavailable
wfst_no_parse
ambiguous_numeric_date
```

That is the right local failure shape: attempted rewrites remain observable without forcing a guess into the transcript.

### What is still not true

The live service still behaves as:

```text
grpc_server
   -> default_classifier
   -> regex_prefilter only
   -> canonical == raw
```

So Stage 1 has not crossed the product boundary yet. The user-visible flip only happens when the gRPC layer starts selecting the WFST classifier and the remaining safety work lands around it.

---

## Remaining work, in the order it should land

| # | Change | Why it belongs before the flip |
|---|---|---|
| 1 | Unify the live class taxonomy: choose `money` or `currency`, then use it consistently across classifier, thresholds, tests, and docs | Today rewritten amount spans are `money`, but `thresholds.yaml` configures `currency`; that silently bypasses the intended gate |
| 2 | Replace blanket lexical-cue handling with rule-aware semantics | Rewritten spans currently lose cue recognition because `_span_has_lex_cue` only accepts `prefilter.*`; risky classes must not be accepted or rejected accidentally |
| 3 | Insert `detect_self_corrections(...)` between classification and gating | Cross-span correction safety belongs in the orchestrator, not in the classifier |
| 4 | Decide identifier scope for this stage | `id_pan_aadhaar_ifsc.py` exists, but the present classifier only wires `phone`; either wire PAN / Aadhaar / IFSC now or state that Stage 1 is phone-only for identifiers |
| 5 | Add request-path integration tests | Unit tests prove the parts; Stage 1 needs tests proving the default service path rewrites and still defers unsafe spans |
| 6 | Upgrade CI to execute real grammar-dependent tests | A skipped regression bar is a painting of a guardrail, not a guardrail |
| 7 | Add a config-level rollback switch | The first live flip needs a reversible knob |
| 8 | Flip gRPC to the WFST classifier per tenant / stream | This is the actual product-visible activation |

---

## The two important hidden hazards

### 1. `money` vs `currency` is not cosmetic

The classifier already maps prefilter `amount` spans to runtime class `money`:

```text
amount -> money
```

But `thresholds.yaml` still defines the gated class as `currency`. Since the gate treats unknown classes as pass-through unless ambiguous, a successful `money` rewrite can bypass the class-specific threshold entirely.

This should be fixed before any live flip. The safest choice is likely:

```text
prefilter amount -> runtime money -> threshold money
```

because the grammar namespace is already `money`, and the live path should have one name at every downstream boundary.

### 2. The current cue function is correct for the old path, not the new one

`normalizer._span_has_lex_cue(...)` currently returns `True` only for `prefilter.*` rule ids. That was harmless while prefilter spans kept `canonical == raw`. Once the classifier emits `wfst.money`, `wfst.date`, `wfst.time`, `fmt.phone`, or `dateparser.date`, cue handling becomes materially important.

The fix should not be “all WFST spans have cues.” It should be class-aware:

```text
money     -> symbol / lexical currency cue
percent   -> literal % or explicit percent word
time      -> AM/PM or strong lexical cue
date      -> month word, trusted policy, or explicit date cue
phone     -> phone/mobile/OTP context or accepted structural rule
```

Prefilter location is evidence; it is not always sufficient evidence.

---

## Proposed commit sequence

### Commit A — close the safety gaps

- normalize the class taxonomy,
- make cue detection rule-aware,
- wire self-correction into `normalize_segment`,
- add tests for those interactions.

This commit changes behavior only when a WFST classifier is explicitly injected; it should still leave the default request path untouched.

### Commit B — make the tests tell the truth

- add service-path integration tests,
- move at least one regression harness onto the same classifier path intended for production,
- update CI to install the native WFST stack or otherwise guarantee the relevant tests truly run.

The core principle: once Stage 1 is live, the green path in CI must exercise real rewrites, not only importable scaffolding.

### Commit C — perform the reversible live flip

- add `wfst_classifier_enabled` under runtime policy,
- resolve tenant policy at stream open,
- construct or cache the tenant classifier,
- pass it into `normalize_segment`,
- deploy initially with the flag off,
- smoke-test,
- then enable for the first rollout cohort.

That split keeps the risky semantic work separate from the operational switch.

---

## Rollout contract

```text
flag off:
  request path remains regex-only passthrough

flag on, WFST available:
  selected spans rewrite under gate control

flag on, WFST missing for one language:
  that language falls back locally to raw spans;
  other languages continue to work

pipeline exception:
  gRPC emits raw_text on every surface, deferred=true;
  transcription delivery survives
```

Do not silently compile FARs in the request path. Do not make missing FARs indistinguishable from a healthy rollout. A raw fallback is acceptable; invisible broken deployment is not.

---

## Decisions already settled

| Question | Decision |
|---|---|
| Where does tenant policy live? | In a closure returned by `make_wfst_classifier(tenant_policy)` |
| What happens on parser / formatter uncertainty? | Emit raw span with telemetry, not a silent drop |
| Should dateparser guess after `ambiguous_numeric_date`? | No; preserve the policy rejection |
| Should one language’s missing FAR poison the process? | No; `get_pipeline(lang)` falls back per language |

---

## Decisions still worth making before the flip

| Question | Recommended default |
|---|---|
| Runtime taxonomy: `money` or `currency`? | Use `money` everywhere after prefiltering |
| Identifier scope in Stage 1? | Either wire PAN / Aadhaar / IFSC now, or explicitly rename the rollout “WFST + phone formatter” |
| CI strategy? | Install the real WFST dependencies in CI; do not let the rewrite path skip on every PR |
| Classifier lifetime? | Resolve per stream, cache by effective tenant policy if profiling proves it matters |
| First rollout surface? | Final segments only; partial-path richness can wait until the gateway genuinely emits partials |

---

## Out of scope for Stage 1

- `display_renderer.py` locale shaping
- gateway / worker integration
- IndicLID promotion onto the hot path
- C++ serving runtime
- FAR cache redesign beyond the current lazy factory
- broad language expansion
- partial transcript UX beyond the existing conservative policy

Stage 1 should stay narrow. Its job is not to finish ITN; its job is to make the first real deterministic rewrites happen safely on the path users will actually hit.

---

## Acceptance bar

Stage 1 is complete only when all of the following are true:

1. A final request through the default service path can produce a real rewrite.
2. Unsafe spans remain raw with explicit fallback reasons.
3. `money` / `currency` class naming is no longer split across layers.
4. Cue policy and self-correction both run before a rewrite becomes visible.
5. CI exercises the grammar-backed path rather than merely importing it.
6. Operators have a rollback switch.
7. A missing FAR or ITN exception degrades the segment, not the transcription service.

That is the moment the architecture stops being latent capability and becomes a trustworthy product behavior.
