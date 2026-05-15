# ITN live-path gap analysis and near-term staging

**Status:** current-state note  
**As of:** 2026-05-15  
**Companion docs:** [implementation_bluprint_INR.md](implementation_bluprint_INR.md), [flowchart_ITN.mmd](flowchart_ITN.mmd)

The blueprint describes a deterministic WFST-first ITN system. The codebase already contains many of the right pieces, but the default production path does **not** yet connect them. This note records what is live today, what that means for planning, and which decisions Stage 1 must close before the rest of the roadmap can create user-visible value.

---

## Executive finding

The default ITN service currently behaves as a **passthrough normalizer with provenance spans**, not as a live text rewriter.

On the default gRPC server path:

- [`runtime.normalizer.default_classifier`](../itn_service/runtime/normalizer.py) calls only [`regex_prefilter.prefilter`](../itn_service/runtime/regex_prefilter.py).
- The prefilter intentionally emits spans with `canonical == raw` and `conf == 1.0`.
- [`WFSTPipeline`](../itn_service/runtime/wfst_pipeline.py), [`dateparser_fallback`](../itn_service/runtime/dateparser_fallback.py), and [`self_correction`](../itn_service/runtime/self_correction.py) are implemented but are not called from [`normalize_segment`](../itn_service/runtime/normalizer.py).
- [`service.grpc_server`](../itn_service/service/grpc_server.py) uses `default_classifier` unless a different classifier is explicitly injected at startup.

Therefore, under the default server configuration, accepted spans are located and logged, but `canonical_text` and `display_text` remain equal to the input text for every normal request. The Hindi grammars, tenant date-order policy, and PAN / Aadhaar / IFSC / phone formatters exist in the tree but do not yet affect a live request.

There is also no gateway or worker integration yet: searches across [`gateway/`](../gateway) and [`worker/`](../worker) currently show no ITN or inverse-normalization call site.

---

## What actually runs today

```text
ASR text
   |
   v
working_copy
   |
   v
route_language
   |
   v
regex_prefilter only
   |
   v
confidence_gate
   |
   v
apply_spans(canonical == raw)
   |
   v
canonical_text == display_text == raw_text
```

The service is still useful as scaffolding:

- It preserves the raw/canonical/display contract.
- It exercises the streaming gate, span offsets, proto translation, and passthrough failure policy.
- It proves the intended service boundary before the heavier normalizer is wired in.

But it is not yet delivering the central product promise of ITN: spoken forms are not being rewritten into canonical written forms on the default live path.

---

## Adjacent truths that change the plan

### 1. Language metadata already exists upstream

The worker resolves a language for each transcription in [`worker/app/model.py`](../worker/app/model.py) and returns it as `language`. The gateway preserves that value on final results via [`gateway/app/worker_client.py`](../gateway/app/worker_client.py) and [`gateway/app/main.py`](../gateway/app/main.py).

Implication: if gateway integration forwards that value into `NormalizeRequest.lang_hint`, ITN-side IndicLID is not required for the current live path. [`script_router.indiclid_predict`](../itn_service/runtime/script_router.py) remains relevant only when upstream language metadata is absent or intentionally set to auto / unknown.

### 2. The gateway is effectively final-only today

[`BufferedWorkerRNNTStream.get_partial`](../gateway/app/main.py) currently returns `None`, and non-final events in the WebSocket path are logged but not emitted to the client. The current deployment path is therefore final-result oriented even though the ITN service itself has streaming machinery.

Implication: the shortest truthful Stage 2 integration is probably:

```text
final ASR result -> gateway ITN RPC -> UI
```

with raw fallback on ITN failure. A richer dual-stream partial architecture can follow once the ASR/gateway path genuinely emits partial transcript updates to users.

### 3. Stage 1 has a hidden class-vocabulary problem

Three names are currently used for closely related amount classes:

| Surface | Current name |
|---|---|
| Regex prefilter | `amount` |
| WFST pipeline | `money` |
| Threshold config | `currency` |

Without choosing one canonical live vocabulary, the confidence gate will become inconsistent exactly where the system most needs to be conservative. This must be normalized as part of Stage 1, not deferred as cleanup.

### 4. The current lexical-cue shortcut is only safe because nothing rewrites yet

[`normalizer._span_has_lex_cue`](../itn_service/runtime/normalizer.py) currently returns `True` for every prefilter span. That means a bare prefilter match such as `17:30` or `12/05/2026` is treated as cue-bearing by the confidence gate even when surrounding text contains no lexical cue.

That shortcut is benign while the prefilter emits `canonical == raw`. It becomes unsafe once Stage 1 starts attaching real rewrites to those spans, because the threshold policy for `date`, `time`, `phone`, and similar classes explicitly expects cue-aware conservatism. Stage 1 must replace the blanket shortcut with class-specific cue semantics before live rewrites are enabled.

### 5. Stage 1 needs an artifact story, not only Python wiring

[`WFSTPipeline`](../itn_service/runtime/wfst_pipeline.py) expects precompiled FAR files under `itn_service/compiled_grammars/`, but that directory currently has no committed artifacts. [`compile.py`](../itn_service/compile.py) can build Hindi and Marathi FARs, yet the default server cannot safely switch to WFST-backed normalization until the build / deploy path guarantees those artifacts exist before startup.

### 6. Some “existing pieces” are optional by dependency, not only by call graph

[`dateparser_fallback`](../itn_service/runtime/dateparser_fallback.py) is intentionally lazy-imported, and `dateparser` is not currently declared in [`itn_service/pyproject.toml`](../itn_service/pyproject.toml). If the fallback is meant to be live in Stage 1, that becomes an explicit runtime dependency decision rather than a no-op integration task.

---

## Recommended staging

| # | Stage | Why it belongs here |
|---|---|---|
| 1 | Wire WFST + deterministic formatters into the live normalizer | Until this lands, the service does not visibly normalize anything. This is the first user-visible value boundary. |
| 2 | Gateway integration | Once Stage 1 rewrites correctly, expose it on the product path. Start with final-only gateway integration because that matches the system that exists today. |
| 3 | Marathi native-review unblock | Marathi is one of the two MVP languages, but the risk is reviewer availability rather than code structure. |
| 4 | Display rendering policy | Needed only if product wants native digits or locale-specific display shaping. The current `canonical == display` default is safe. |
| 5 | Hindi gold expansion | Hindi is not honestly “covered” while the regression bar measures only cardinal gold. |
| 6 | FAR cache hardening | Important for future multi-tenant / multi-worker layouts, low ROI for the current single-process server. |
| 7 | Remaining language expansion | Order by traffic share, not alphabetically. |
| 8 | IndicLID | Pull forward only if upstream language hints are not reliable in production. |
| 9 | C++ runtime / Sparrowhawk | Defer until Python latency is a real bottleneck. Current tests already target live-call budgets. |
| 10 | Tooling | Build alongside gold expansion and language rollout where it compounds development speed. |

---

## Stage 1: what the next PR actually needs to decide

Stage 1 should remain a small-surface integration PR, but it is more than “call the existing classes.” It needs to make the following decisions explicit:

1. **Choose one canonical span taxonomy.**  
   Resolve `amount` vs `money` vs `currency`, and make prefilter, classifier, thresholds, tests, and docs speak the same language.

2. **Widen the classifier contract or move policy-aware rewriting outside it.**  
   The current `Classifier` protocol receives only `(working_text, lang)`. That is not enough to honor tenant-specific `date_order`, because `normalize_segment` receives `locale_policy` separately. Either:
   - widen the classifier interface to receive policy/context, or
   - keep classification pure and add a dedicated rewrite stage after classification.

3. **Define the rewrite order.**  
   The intended safe order is:

   ```text
   prefilter locate
      -> WFST normalize supported spoken classes
      -> deterministic formatters for phone / PAN / Aadhaar / IFSC
      -> policy-aware date normalization
      -> strict dateparser fallback only when allowed
      -> self-correction marking
      -> confidence gate
      -> splice accepted spans
   ```

4. **Replace blanket lexical-cue inference with class-specific semantics.**  
   Prefilter location alone must not satisfy `require_lex_cue` for risky classes. Currency symbols and literal `%` signs may be self-cuing; bare numeric dates, times, and phone-shaped runs should require stronger evidence.

5. **Specify failure behavior per span, not only per request.**  
   The service-level fallback in `grpc_server.py` already protects transcription when the whole pipeline raises. Stage 1 should also preserve local safety: when one formatter or grammar declines a span, that span should fall through to raw rather than poisoning unrelated rewrites in the same segment.

6. **Guarantee FAR availability before request handling.**  
   Either build FARs in the image / release pipeline or make server startup fail loudly with an operator-friendly error if required FARs are absent. Silent fallback to passthrough would hide a broken deploy.

7. **Add end-to-end tests against the default classifier path.**  
   The repo already has strong unit tests for grammars and formatters. Stage 1 needs tests proving the actual service path rewrites representative examples and still defers unsafe spans.

---

## Suggested next PR slice

The highest-leverage next PR is still Stage 1 only:

- introduce a WFST-backed live classifier / rewrite stage,
- unify the amount-class taxonomy,
- replace blanket lexical-cue inference with class-specific cue handling,
- connect tenant date policy,
- wire deterministic identifier / phone formatters,
- run self-correction before the final gate,
- add default-path integration tests,
- and make FAR presence an explicit startup requirement.

Everything else compounds on top of that. Before this PR, the service architecture exists. After it, the service begins doing ITN.

---

## Open questions worth resolving before later stages

1. **How reliable is upstream language resolution in production?**  
   The code path exists; the production question is whether `language` is always present and trustworthy enough to keep IndicLID off the critical path.

2. **What is the actual tenant `date_order` distribution?**  
   The config supports DMY / MDY / YMD, but if production is effectively all-DMY today, numeric-date ambiguity is a colder branch than the blueprint suggests.

3. **What is the real traffic mix across the eight stub languages?**  
   That should determine rollout order after Hindi / Marathi, not the directory listing.

4. **Does product want native-digit display at all?**  
   If not, [`display_renderer.py`](../itn_service/runtime/display_renderer.py) can remain safely boring for longer.

5. **Who owns Marathi native review, and by when?**  
   [`tests/gold/mr/REVIEW_REQUIRED.md`](../itn_service/tests/gold/mr/REVIEW_REQUIRED.md) is the largest non-code blocker to a two-language MVP.

---

## Bottom line

The current architecture is promising, but the present live path is still a dry canal: all the channel work is there, yet the water has not been connected. Stage 1 is the sluice gate. Until it lands, downstream work can improve readiness, but it cannot change what users actually see.
