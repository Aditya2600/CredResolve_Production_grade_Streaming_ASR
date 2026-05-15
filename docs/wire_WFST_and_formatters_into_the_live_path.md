Stage 1 plan — wire WFST + formatters into the live path
Architecture, one diagram

default_classifier(working_text, lang, *, tenant_policy)
  │
  ├──► prefilter(working_text)                    # locates spans, sets canonical=raw, conf=1.0
  │
  ├──► for each prefilter span, dispatch by class:
  │      PHONE                       → formatters.parse_phone_in(raw)
  │      AMOUNT/PERCENT/TIME         → WFSTPipeline(lang).normalize_span(raw, cls)
  │      DATE                        → WFSTPipeline(lang).normalize_date(raw, date_order=…)
  │                                     └─ if no parse + has_date_cue → dateparser_fallback
  │
  ├──► detect_self_correction(working_text, spans)  # pairwise pass; marks unsafe → ambiguous=True
  │
  └──► return spans  (gate runs downstream as today)
The classifier becomes a per-tenant closure so the existing Classifier protocol (text, lang) -> list[Span] is unchanged; tenant policy is captured at construction time.

Tasks, in order
#	File	Change	Why
1	new runtime/wfst_factory.py	@functools.cache-d get_pipeline(lang) -> WFSTPipeline | None. Catches ImportError/FileNotFoundError, returns None if FAR or pynini unavailable.	Defers the full far_cache.py design. Lazy import keeps Python unit tests runnable without pynini.
2	new runtime/wfst_classifier.py	make_wfst_classifier(tenant_policy) -> Classifier. Internally: prefilter → per-class dispatch → returns rewritten spans. Each rewrite uses rule_id wfst.<cls> / fmt.<cls> / dateparser.<cls>; fallback to canonical=raw, ambiguous=True, fallback_reason=<reason> on no-parse.	New classifier; doesn't touch the protocol or normalize_segment.
3	runtime/wfst_classifier.py	Class-name map: PHONE → phone; AMOUNT → money; PERCENT → percent; DATE → date; TIME → time.	Bridges prefilter labels (UPPERCASE, locating) and threshold/WFST labels (lowercase, rewriting).
4	runtime/normalizer.py:71-111	Keep default_classifier as the regex-only safe default. Add _span_has_lex_cue recognition for wfst.*, fmt.*, dateparser.* rule ids.	Doesn't break the existing "safe before FARs build" guarantee.
5	runtime/normalizer.py:174-…	Add a self-correction pass between classifier and gate: unsafe = detect_self_correction(working, spans); spans = [s.model_copy(update={"ambiguous": True, "fallback_reason": "self_correction"}) if i in unsafe else s for i, s in enumerate(spans)].	Cross-span; belongs in the orchestrator, not the classifier. Gate then reverts canonical→raw on those.
6	service/grpc_server.py:169-302	At stream open, resolve tenant_policy = locale_policy.for_tenant(req.locale_policy); build classifier = make_wfst_classifier(tenant_policy); pass to normalize_segment. Cache per (tenant_id,) via functools.lru_cache.	Single change to make the WFST classifier the request-path default.
7	configs/policy.yaml	Add a feature flag wfst_classifier_enabled: true (default true) read by grpc_server to pick between default_classifier and make_wfst_classifier(…).	Lets us roll back without code change if a regression slips. Deletable later.
8	new tests/runtime/test_wfst_classifier.py	Per-class tests: phone uses formatter, AMOUNT routes to WFST money, DATE honours date_order and falls back to dateparser on cue+no-parse.	New unit surface.
9	tests/regression/test_hi_gold_no_regression.py	Switch the regression harness from default_classifier to the WFST classifier (with default tenant policy).	Today the ≥98% bar is measured on the no-op path; this makes it meaningful.
10	.github/workflows/itn_service.yml	Add libicu-dev, libopenfst-dev + pip install pynini PyICU to the CI step, or mark grammar-dependent tests with pytest.importorskip("pynini") and continue skipping.	Pick one based on Q2 below.
Sequencing
Three commits, mergeable independently:

Commit A — Tasks 1, 2, 3, 8. Adds the new classifier module and its tests. Nothing in the request path changes yet; this is pure addition. Safe to merge before CI installs pynini, because tests skip if pynini is missing.
Commit B — Tasks 4, 5, 9. Wires self-correction into normalize_segment, adjusts _span_has_lex_cue, and points the regression harness at the new classifier. The harness will catch real regressions here.
Commit C — Tasks 6, 7. Flips grpc_server to the WFST classifier behind the policy flag. This is the user-visible flip. Roll out with wfst_classifier_enabled: false first, smoke-test, then flip to true.
Task 10 (CI) lands before Commit B since the regression test needs real FARs.

Design decisions worth confirming before I start
Q	Decision	Default if you don't push back
Q1	Does WFST-no-parse + no cue mean "emit raw with wfst_no_parse" (telemetry) or just "leave the prefilter span untouched" (silent)?	Telemetry. Record the attempt; lets us measure WFST coverage in production.
Q2	CI: install pynini/PyICU vs. skip-if-missing?	Install in CI. Otherwise the regression test silently no-ops on PR builds.
Q3	Tenant resolution lifetime: per-stream (cached on StreamState) or per-process?	Per-stream. Tenants rarely change but reload-on-stream-open lets us pick up locales.yaml edits without bouncing the service.
Q4	Where does tenant_policy live in the classifier protocol — closure, third arg, or a request_ctx object?	Closure (make_wfst_classifier(tenant_policy)). Keeps the protocol stable and avoids threading state through every call.
Q5	If the phone formatter returns None on uncertainty, should we map that to canonical=raw, ambiguous=True, fallback_reason="fmt_no_parse" or drop the span entirely?	Ambiguous + raw. Same as WFST-no-parse. The prefilter already located it as a phone-shaped span; downstream consumers benefit from the span existing with a reason.
What this doesn't touch
display_renderer.py still passthrough (Stage 4).
far_cache.py still stub (the functools.cache factory in Task 1 is a placeholder).
Marathi remains review-gated.
IndicLID still stub.
Gateway integration untouched (Stage 2).
Risks
CI pynini install adds ~3–5 min build time. Acceptable; alternative is a regression bar that no-ops in CI which is worse.
Hindi gold is cardinal-only today (223 examples). Once the WFST classifier is live, the ≥ 98% regression bar applies to cardinals only. Expanding gold (Stage 5) should happen close in time — otherwise the harness is reassuring but narrow.
WFSTPipeline.__init__ raises KeyError if a FAR entry is missing (wfst_pipeline.py:99-123). Today's FAR has cardinal/decimal/money/percent/date/time for hi+mr. The factory should try/except construction and fall back to regex-only for that language. Failed-init for one language must not poison others.
Self-correction false positives. detect_self_correction flags pairs within a 6-token window when a marker word appears. Hindi नहीं (no/negation) is a marker but is also conversational. Worth a careful look at self_correction.py:34-62 once we have real call traffic — but for Stage 1, ship as-is and tune from telemetry.
