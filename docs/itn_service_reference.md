# ITN Service — Complete Reference

**Scope:** `itn_service/`  
**Status:** Stage 1 in progress — WFST classifier built, gateway integration pending  
**Last updated:** 2026-06-06  
**Related docs:**
- [implementation_blueprint_INR.md](implementation_blueprint_INR.md) — full architecture design  
- [itn_live_path_gap_analysis.md](itn_live_path_gap_analysis.md) — current live-path gap and staging plan  
- [concrete_pipeline.md](concrete_pipeline.md) — pipeline execution reference  

---

## 1. What ITN Is

**Inverse Text Normalisation (ITN)** is the post-ASR step that converts spoken-form transcripts into their canonical written form.

| Direction | Name | Used in |
|---|---|---|
| written → spoken | Text Normalisation (TN) | TTS |
| spoken → written | Inverse Text Normalisation (ITN) | ASR post-processing |

For this system, ITN converts ASR output such as:

```
"kal pachas rupaye transfer karna hai"
```

into a normalised, machine-readable form:

```
"kal ₹50 transfer karna hai"
```

This service is **deterministic and WFST-first**. No LLM or seq2seq model sits in the live request path. Pynini grammars compiled to FAR archives, deterministic formatters, and class-specific confidence thresholds do all the work.

---

## 2. Non-Negotiable Invariants

These four invariants are enforced by [`CONTRIBUTING.md`](../itn_service/CONTRIBUTING.md), [`policy.yaml`](../itn_service/configs/policy.yaml), and the code in [`runtime/normalizer.py`](../itn_service/runtime/normalizer.py). **A change that breaks any of them requires explicit maintainer sign-off.**

| # | Invariant | Why |
|---|---|---|
| 1 | `raw_text` is **never mutated** | Audit, debugging, legal review of verbatim ASR output |
| 2 | **No LLM / seq2seq** in the live path | Hallucinated formatting errors are unrecoverable in finance/healthcare |
| 3 | **No FAR compilation** in the request path | Compilation is a build-time step; doing it per-request is a latency regression |
| 4 | **Canonical storage uses Latin digits** + ICU separators | Keeps downstream search, NER, analytics, and CRM ingestion locale-stable |

---

## 3. Output Contract

Every call produces a [`SegmentResult`](../itn_service/runtime/contract.py) with these fields:

| Field | Surface | Consumer |
|---|---|---|
| `raw_text` | Verbatim ASR hypothesis — immutable audit surface | Logging, legal, error recovery |
| `canonical_text` | Stable machine-facing form after accepted rewrites. Latin digits, ICU separators | Search, NER, analytics, CRM ingestion |
| `display_text` | Locale-rendered for the agent UI | Frontend transcript display |
| `spans[]` | Per-rewrite provenance: class, raw, canonical, rule_id, confidence, offsets, fallback reason | Debugging, review workflows, analytics |
| `deferred` | `true` when segment was not normalised (unstable partial or pipeline failure) | Caller should not persist a deferred result |
| `lang` | Routed BCP-47-ish 2-letter language code | Grammar selection |
| `script` | Detected dominant script name (e.g. `"Devanagari"`) | Routing metadata |
| `itn_version` | Snapshot of `ITN_CONTRACT_VERSION` at server start | Schema compatibility checks |

### Pydantic models (`runtime/contract.py`)

```python
class Token(BaseModel):       # One ASR token
    text: str
    start_ms: int             # >= 0
    end_ms: int               # >= 0
    conf: float               # [0.0, 1.0]

class Span(BaseModel):        # One normalised span
    cls: str                  # semiotic class: "money", "date", "phone", ...
    raw: str                  # verbatim text from working copy
    canonical: str            # rewritten form (== raw when gate rejects)
    rule_id: str              # e.g. "wfst.money", "fmt.phone", "prefilter.date.v1"
    conf: float               # [0.0, 1.0]
    ambiguous: bool           # True when multiple parses exist
    start: int | None         # codepoint offset into working copy
    end: int | None
    fallback_reason: str | None  # ';'-joined gate failure reasons

class SegmentResult(BaseModel):
    raw_text: str
    canonical_text: str
    display_text: str
    spans: list[Span]
    deferred: bool
    lang: str
    script: str
    itn_version: str
```

---

## 4. Repository Layout

```
itn_service/
├── configs/
│   ├── policy.yaml          # Output-policy invariants + rollout switches
│   ├── locales.yaml         # Per-language: script, numbering, date order, currency, bidi
│   └── thresholds.yaml      # Confidence gating table — single source of truth
├── grammars/
│   ├── common/              # Shared: digit_maps.tsv, separators.tsv, currency_aliases.tsv
│   ├── hi/                  # Hindi Pynini grammars: cardinal, decimal, money, date, time, ...
│   ├── mr/                  # Marathi
│   ├── bn/ gu/ ta/ te/ kn/ ml/ pa/ ur/   # Other Indic languages
├── runtime/
│   ├── contract.py          # PUBLIC: Token, Span, SegmentResult — the only caller contract
│   ├── normalizer.py        # normalize_segment() — main entry point
│   ├── confidence_gate.py   # Threshold-based gating logic
│   ├── regex_prefilter.py   # Cheap regex span location (no rewrites)
│   ├── spoken_prefilter.py  # Spoken-form number span detector
│   ├── wfst_classifier.py   # WFST-backed classifier (Stage 1)
│   ├── wfst_factory.py      # Lazy per-language FAR loader
│   ├── wfst_pipeline.py     # WFST normalization pipeline
│   ├── script_router.py     # Language and script detection + routing
│   ├── stream_state.py      # Per-call stability counter
│   ├── unicode_clean.py     # working_copy() — Unicode NFC + compatibility folding
│   ├── locale_policy.py     # Tenant policy: date order, currency
│   ├── dateparser_fallback.py  # Guarded dateparser fallback
│   ├── self_correction.py   # Cross-span correction detection
│   ├── display_renderer.py  # Locale display shaping (stub, passthrough now)
│   └── formatters/
│       └── phone_in.py      # Indian mobile number formatter
├── service/
│   ├── itn.proto            # gRPC service definition
│   ├── grpc_server.py       # Bidi-streaming gRPC server
│   └── build_protos.sh      # Generates itn_pb2 / itn_pb2_grpc stubs
├── cxx_runtime/             # C++ serving path (Sparrowhawk-style) — future
├── tests/
│   ├── gold/<lang>/         # Per-language gold test sets
│   ├── regression/          # Service-path regression tests
│   └── latency/             # Latency benchmarks
├── tools/
│   ├── export_far.sh        # Build FAR archives from Pynini grammars
│   ├── benchmark_latency.py
│   └── build_gold_from_csv.py
├── compile.py               # Build Hindi / Marathi FARs
├── CONTRIBUTING.md          # Invariants, what-goes-where, lint/type/test guide
└── README.md                # One-pager + layout + dev install
```

---

## 5. End-to-End Request Pipeline

### 5.1 High-level flow

```
ASR hypothesis (text + tokens + is_final + lang_hint + locale_policy)
        │
        ▼
 update_stability(text)
        │
   ┌────┴──────────────────────────────────┐
   │ unstable partial                      │ final or stable partial
   ▼                                       ▼
emit raw on all surfaces           working_copy(raw_text)      ← unicode_clean.py
   (deferred=true)                         │
                                   route_language(text, hint)   ← script_router.py
                                           │
                                   classify spans               ← wfst_classifier.py
                                           │                      or default_classifier
                                   _mark_self_corrections       ← self_correction.py
                                           │
                                [partial: filter to safe classes]
                                           │
                                   confidence_gate              ← confidence_gate.py
                                           │
                                   apply_spans → canonical_text ← normalizer.apply_spans
                                           │
                                   display_renderer             ← display_renderer.py
                                           │
                                   SegmentResult emitted
```

### 5.2 Segment policy

| Input state | What runs | Notes |
|---|---|---|
| Unstable partial (`stable_count < 2`) | Return raw on all surfaces, `deferred=true` | Avoids UI churn while decoder revises |
| Stable partial | Only `cardinal`, `money`, `percent` may render | Prefix-safe updates only |
| Final segment | Full classify → gate → splice → display | Full pipeline; all classes eligible |

### 5.3 Final-segment step sequence

1. **Preserve** — `raw_text` copied verbatim, never mutated  
2. **Prepare** — `working_copy(raw_text)` produces a Unicode-NFC internal copy  
3. **Route** — `route_language(working, lang_hint)` → `RouteResult(lang, script, source)`  
4. **Classify** — `classifier(working, lang)` → `list[Span]` with proposed rewrites  
5. **Self-correction** — `_mark_self_corrections(working, spans)` marks corrected spans ambiguous  
6. **Gate** — per span: check classifier conf, ASR conf, lexical cue, partial/final policy  
7. **Assemble** — `apply_spans(working, safe_spans)` → `canonical_text`  
8. **Render** — `display_renderer(canonical_text, lang)` → `display_text` (passthrough now)  
9. **Emit** — `SegmentResult(raw_text, canonical_text, display_text, spans, …)`

### 5.4 Rewrite loop (inside classifier)

```python
for span in candidate_spans:
    candidate = rewrite(span.raw, cls=span.cls, lang=lang, policy=locale_policy)

    if candidate is None:
        emit_span(raw=span.raw, canonical=span.raw,
                  ambiguous=True, fallback_reason="no_parse")
        continue

    confidence = score(candidate, raw, cls, context, asr_confidence)

    if confidence < threshold_for(span.cls) or requires_missing_cue(span):
        emit_span(raw=span.raw, canonical=span.raw, fallback_reason="gate_reject")
    else:
        emit_span(raw=span.raw, canonical=candidate, conf=confidence)
```

> **Key asymmetry**: a missed rewrite is tolerable. A wrong rewrite is expensive and potentially unrecoverable.

---

## 6. Language & Script Routing (`runtime/script_router.py`)

Language routing uses a priority cascade:

| Priority | Rule | Code path |
|---|---|---|
| (a) | ASR `lang_hint` present and in trusted set | `asr_hint` branch |
| (b) | ICU UScript histogram of working copy | `script_majority` branch |
| (b′) | Devanagari → Marathi vs Hindi keyword score | `script_majority_mr_keywords` |
| (c) | IndicLID (long/romanised spans) | **Stub — not wired yet** |

### Supported languages

`hi` Hindi · `mr` Marathi · `bn` Bengali · `ta` Tamil · `te` Telugu · `kn` Kannada · `ml` Malayalam · `gu` Gujarati · `pa` Punjabi · `ur` Urdu · `en` English/romanised

### Devanagari disambiguation (Hindi vs. Marathi)

Both languages share the Devanagari script, so a keyword-score tiebreak is used.

**Marathi wins** when its score exceeds Hindi's by > 0.5 (the `_MR_WIN_MARGIN`). High-signal Marathi cues:

| Category | Cues (weight) |
|---|---|
| Time cues | `वाजता` (2.0), `वाजून` (2.0), `मिनिटे` (1.5) |
| Half/quarter | `दीड` (2.0), `अडीच` (2.0), `पावणे` (2.0) |
| Percent | `टक्के` (2.0), `टक्का` (1.5) |
| Hundreds | `पाचशे` (2.0), `दोनशे` (2.0), `तीनशे` (2.0), … |
| Numerals | `दोन`, `सहा`, `नऊ`, `दहा`, `अकरा` (1.0 each) |
| Month names | `जानेवारी` (2.0), `जुलै` (2.0), `ऑगस्ट` (2.0), … |

**Hindi wins** (blocks Marathi) via cues like `बजे` (2.0), `बजकर` (2.0), `डेढ़` (2.0), `करोड़` (1.5), `जनवरी` (2.0), etc.

---

## 7. Regex Prefilter (`runtime/regex_prefilter.py`)

The prefilter is a **locator, not a normaliser**. Every span it emits has `canonical == raw` and `conf == 1.0`. Downstream stages own the rewrite.

| Class | Rule ID | Pattern |
|---|---|---|
| `url` | `prefilter.url.v1` | `http(s)://…` or `www.<host>…` |
| `email` | `prefilter.email.v1` | Practical RFC subset |
| `ifsc` | `prefilter.ifsc.v1` | `[A-Z]{4}0[A-Z0-9]{6}` |
| `pan` | `prefilter.pan.v1` | `[A-Z]{5}\d{4}[A-Z]` |
| `aadhaar` | `prefilter.aadhaar.v1` | 12-digit UID, optional space/dash separators |
| `phone` | `prefilter.phone.v1` | Indian mobile: optional `+91`/`0`, then 10 digits starting `6–9` |
| `date` | `prefilter.date.v1` | `dd/mm/yy(yy)` with `/`, `-`, or `.` separators |
| `time` | `prefilter.time.v1` | `hh:mm[:ss][ AM\|PM]` |
| `amount` | `prefilter.amount.v1` | Currency symbol (`₹`, `Rs.`, `INR`, `$`, …) + digits |
| `percent` | `prefilter.percent.v1` | `\d+(\.\d+)? %` |

**Overlap resolution:** `url > email > ifsc > pan > aadhaar > phone > date > time > amount > percent` (lower priority number wins).

---

## 8. WFST Classifier (`runtime/wfst_classifier.py`)

The WFST classifier is the Stage 1 upgrade over the default regex-only classifier. It uses `make_wfst_classifier(tenant_policy)` to build a **per-tenant closure** that keeps the stable `(working_text, lang) -> list[Span]` protocol.

### Span merge priority (written + spoken prefilter combined)

| Priority | Class |
|---|---|
| 1 | `url` |
| 2 | `email` |
| 3 | `ifsc` |
| 4 | `pan` |
| 5 | `aadhaar` |
| 6 | `phone` |
| 7 | `date` |
| 8 | `time` |
| 9 | `amount` |
| 10 | `percent` |
| 11 | `decimal` |
| 12 | `cardinal` |

### Per-class rewrite branches

| Prefilter class | WFST class | Handler |
|---|---|---|
| `phone` | `phone` | `parse_indian_mobile()` (deterministic formatter) |
| `amount` | `money` | `wfst_pipeline.normalize_span(raw, "money")` |
| `percent` | `percent` | WFST |
| `time` | `time` | WFST |
| `cardinal` | `cardinal` | WFST |
| `decimal` | `decimal` | WFST |
| `date` | `date` | WFST → dateparser fallback (if cue present) |

### Class taxonomy note

The regex prefilter emits `amount`; the WFST classifier maps this to `money` at rewrite time. `thresholds.yaml` defines gates for both `currency` (legacy) and `money`. The threshold for `money` and `currency` are intentionally kept at the same level (`classifier_min: 0.90`, `asr_min: 0.80`) until taxonomy consolidation lands.

### Failure isolation

Each span is wrapped in a `try/except`. A formatter or WFST exception for one span emits a raw fallback for that span only — it never blocks rewrites of other spans in the same segment.

---

## 9. Confidence Gate (`runtime/confidence_gate.py`)

The gate is the **single decision point** for whether a proposed rewrite is accepted. It reads from `configs/thresholds.yaml` — the single source of truth.

### Gate inputs (per span)

| Input | Source |
|---|---|
| `span.conf` | Classifier / rule confidence |
| `asr_conf` | Min token confidence over the segment |
| `has_lex_cue` | Class-specific lexical evidence (see below) |
| `is_partial` | Whether the segment is not final |
| `span.ambiguous` | Set by upstream classifiers on parse conflict |

### Threshold table (`configs/thresholds.yaml`)

| Class | `classifier_min` | `asr_min` | `require_lex_cue` | `defer_on_partial` |
|---|---|---|---|---|
| `cardinal` | 0.85 | 0.70 | false | false |
| `decimal` | 0.85 | 0.75 | **true** | false |
| `percent` | 0.85 | 0.75 | **true** | false |
| `currency` | 0.90 | 0.80 | **true** | false |
| `money` | 0.90 | 0.80 | **true** | false |
| `time` | 0.90 | 0.80 | **true** | **true** |
| `date` | 0.90 | 0.80 | **true** | **true** |
| `phone` | 0.95 | 0.85 | **true** | **true** |
| `id` | 0.95 | 0.90 | **true** | **true** |
| `health_dose` | 0.95 | 0.90 | **true** | **true** |

Streaming policy: `partial_stable_min: 2`, `partial_safe_classes: [cardinal, money, percent]`

### Lexical cue semantics (`normalizer._span_has_lex_cue`)

| Class | Cue evidence required |
|---|---|
| `amount` / `currency` / `money` | Explicit currency symbol or word (`₹`, `Rs`, `INR`, `रुपये`, …) |
| `percent` | Literal `%` or `percent` / `प्रतिशत` / `टक्के` |
| `decimal` | Explicit `point` / `dot` / `decimal` / `दशमलव` |
| `time` | AM/PM or strong Hindi/Marathi time cue (`बजे`, `बजकर`, `सुबह`, …) |
| `date` | Month word, year cue, trusted locale prior, **or** `rule_id` starts with `wfst.` / `dateparser.` |
| `phone` | `rule_id` starts with `fmt.` **or** phone/mobile/OTP context word present |

Prefilter location alone is **not** sufficient for risky classes. Bare `17:30` is not self-cuing for `time`; bare `12/05/2026` is not self-cuing for `date`.

### Fallback behaviour

When the gate rejects a span:
- `canonical` is reset to `raw` (no rewrite)
- `fallback_reason` is populated with `;`-joined failure tokens (e.g. `"asr_conf<0.80;missing_lex_cue"`)
- The span is still emitted in `spans[]` so provenance is preserved

---

## 10. Streaming Stability (`runtime/stream_state.py`)

One `StreamState` is created per ASR call/session. It counts consecutive identical partial hypotheses.

```python
state = StreamState(stability_threshold=2)  # default threshold from thresholds.yaml

count = state.update_stability(partial_text)
# Returns 1 on first call, increments on repeated identical text,
# resets to 1 on any text change.

state.reset()   # Called after is_final=True; clears count for next segment
```

Callers branch on `state.stable_count >= state.stability_threshold` before allowing partial rewrites.

---

## 11. gRPC API (`service/itn.proto`)

### Service

```protobuf
service ItnService {
  rpc StreamNormalize(stream NormalizeRequest)
      returns (stream NormalizeResponse);
}
```

One gRPC stream = one ASR call session. The server holds one `StreamState` per stream. Requests and responses are ordered (one response per request, in order).

### Request (`NormalizeRequest`)

| Field | Type | Description |
|---|---|---|
| `text` | `string` | Verbatim ASR hypothesis |
| `tokens` | `repeated Token` | Per-token timing + confidence. Empty → `asr_conf = 1.0` |
| `is_final` | `bool` | `true` = segment final; runs full pipeline |
| `lang_hint` | `string` | BCP-47-ish 2-letter code (`"hi"`, `"mr"`, …). Empty = no hint |
| `locale_policy` | `string` | Tenant ID for date order + currency. Empty = default tenant |

### Response (`NormalizeResponse`)

| Field | Type | Description |
|---|---|---|
| `raw_text` | `string` | Verbatim copy of request text |
| `canonical_text` | `string` | Normalised machine-facing form |
| `display_text` | `string` | Locale-rendered UI form |
| `spans` | `repeated Span` | Rewrite provenance |
| `deferred` | `bool` | `true` when segment was not normalised |
| `lang` | `string` | Routed 2-letter language code |
| `script` | `string` | Detected script (e.g. `"Devanagari"`) |
| `itn_version` | `string` | `ITN_CONTRACT_VERSION` snapshot |

### Span proto fields

| Field | Notes |
|---|---|
| `cls`, `raw`, `canonical`, `rule_id`, `conf`, `ambiguous` | Mirror `runtime.contract.Span` |
| `has_position` | `true` iff `start` / `end` are meaningful (proto3 int cannot distinguish 0 from absent) |
| `start`, `end` | Codepoint offsets into the **working copy** (not raw_text) |
| `fallback_reason` | Empty string = no fallback applied |

---

## 12. gRPC Server (`service/grpc_server.py`)

### Startup sequence

1. Load `thresholds.yaml` once → `ThresholdTable`
2. Load `policy.yaml` → `ServicePolicy` (reads `wfst_classifier_enabled`)
3. Load `locales.yaml` → `TenantPolicyTable` (all tenant configs)
4. If `wfst_classifier_enabled`, build `make_wfst_classifier(tenant_policy)` closures per tenant
5. Start gRPC server, bind, install `SIGTERM`/`SIGINT` handler (10s graceful drain)

### Per-request path

```
StreamNormalize(request_iterator, context)
    for req in request_iterator:
        tenant_policy = locale_policies.for_tenant(req.locale_policy)
        tokens = request_to_tokens(req.tokens)
        result = normalize_segment(
            raw_text=req.text,
            tokens=tokens,
            is_final=req.is_final,
            state=state,          # per-stream StreamState
            lang_hint=req.lang_hint,
            locale_policy=tenant_policy.tenant_id,
            thresholds=shared_thresholds,
            classifier=tenant_classifier,
        )
        if req.is_final:
            state.reset()
        yield result_to_response(result)
```

### Failure semantics

| Failure scope | Behaviour |
|---|---|
| One span formatter/WFST error | Span falls back to raw; other spans in segment continue |
| `normalize_segment` raises | `passthrough_result(raw_text)` emitted → `deferred=true`, empty spans |
| Proto translation raises | Minimal `NormalizeResponse(raw_text=raw_text, deferred=true)` emitted |
| Request iterator fails | Stream closes; final passthrough attempted for last request |

**ITN failures never break transcription delivery.** The stream stays open after per-message failures.

### Run locally

```bash
# 1. Build proto stubs
bash itn_service/service/build_protos.sh

# 2. Start server
python -m itn_service.service.grpc_server --bind '[::]:50051'
```

---

## 13. Configuration Files

### `configs/policy.yaml`

Controls output policy invariants and the WFST rollout switch.

```yaml
invariants:
  preserve_raw_asr_text: true
  no_llm_in_live_path: true
  no_far_compile_in_request_path: true
  canonical_storage_latin_digits: true

endpoint_normalisation_default: true
wfst_classifier_enabled: true          # flip to false for regex-only fallback

unsafe_rewrite_classes_blocklist:      # never auto-rewrite; emit suggestion only
  - account
  - policy_id
  - claim_id
  - medicine_dose

rewrite_cache:
  enabled: true
  max_entries: 4096

logging:
  log_spans: true
  span_fields: [raw_span, canonical_span, class, language, script, rule_id,
                alternatives_count, asr_confidence, classifier_confidence,
                fallback_reason, ui_display_span]
```

### `configs/thresholds.yaml`

Single source of truth for all confidence thresholds. Never hard-code thresholds elsewhere. See §9 for the full table.

### `configs/locales.yaml`

Per-language metadata: script name, numbering systems (`latn` / `deva` / `arabext` / …), date order (`DMY` / `MDY` / `YMD`), default currency, and bidi flags (needed for Urdu).

---

## 14. Grammar Authoring

Grammars follow the NeMo/Kestrel style: **classify → parse → generate permutations → verbalise**.

```
grammars/<lang>/
    cardinal.py       # Spoken integers → Latin digit string
    decimal.py        # Spoken decimals
    money.py          # Currency amounts
    date.py           # Date rewriting with locale policy
    time.py           # Clock time
    percent.py        # Percentage
    phone.py          # Phone (defer to fmt.phone_in for formatting)
    id.py             # Identifier separator cleanup
    verbalize_final.py
grammars/common/
    digit_maps.tsv
    separators.tsv
    currency_aliases.tsv
    class_labels.tsv
    id_prefixes.tsv
```

### Adding a new language

1. Copy a shared class skeleton (e.g., `hi/`).
2. Swap only language-specific lexica: connectors, digit maps, month names, ambiguity rules.
3. Add gold tests under `tests/gold/<lang>/`.
4. Do **not** rewrite each language from scratch.
5. Build FARs with `compile.py` or `tools/export_far.sh`.

### Pynini grammar sketch (Hindi money)

```python
import pynini
from pynini.lib import pynutil

DEVANAGARI_TO_LATN = pynini.string_map([
    ("०","0"), ("१","1"), ("२","2"), ("३","3"), ("४","4"),
    ("५","5"), ("६","6"), ("७","7"), ("८","8"), ("९","9"),
])

COMMON_CURRENCY = pynini.union("₹", "Rs", "रुपये", "रुपया")

digit          = pynini.union(*"0123456789")
native_or_latn = DEVANAGARI_TO_LATN | digit
integer        = pynini.closure(native_or_latn, 1)

money = (
    pynutil.insert('money { currency: "INR" amount: "') +
    (integer + pynini.closure("," + integer)) +
    pynutil.insert('" }')
)

CLASSIFY = money.optimize()
```

---

## 15. Supported Semiotic Classes

| Class | Examples | Auto-rewrite on partials? |
|---|---|---|
| `cardinal` | "पाँच सौ" → `500` | ✅ |
| `money` | "₹1,250", "पचास रुपये" → `₹50` | ✅ |
| `percent` | "12.5%" | ✅ |
| `decimal` | "बारह दशमलव पाँच" → `12.5` | ❌ (defer) |
| `time` | "5:30 PM", "शाम पाँच बजकर तीस" → `17:30` | ❌ (defer) |
| `date` | "12/05/2026" (DMY policy) → `12 May 2026` | ❌ (defer) |
| `phone` | `9876543210` → `+91-9876543210` | ❌ (defer) |
| `pan` | `ABCDE1234F` — separator cleanup only | ❌ (defer) |
| `aadhaar` | 12-digit UID — separator cleanup | ❌ (defer) |
| `ifsc` | `HDFC0001234` — no rewrite | ❌ (defer) |
| `id` | Account/policy numbers — separator cleanup only | ❌ (defer) |
| `health_dose` | Dosage amounts — suggestion only | ❌ (defer) |

Classes in the `unsafe_rewrite_classes_blocklist` (`account`, `policy_id`, `claim_id`, `medicine_dose`) **never auto-rewrite**; they emit canonical-form suggestions for downstream review only.

---

## 16. Indic-Specific Handling

### Numbering systems

Canonical storage always uses Latin digits (`latn`). Native-digit display is the renderer's concern.

| Language | Scripts | Numbering systems |
|---|---|---|
| Hindi | Devanagari + Latin | `latn`, `deva` |
| Marathi | Devanagari + Latin | `latn`, `deva` |
| Bengali | Bengali + Latin | `latn`, `beng` |
| Tamil | Tamil + Latin | `latn`, `tamldec`, traditional `taml` |
| Urdu | Arabic-ext + Latin | `latn`, `arabext` (bidi required) |

### Date ambiguity

`12/05/2026` is only normalised to `12 May 2026` if the tenant `locale_policy` is confirmed as day-first (`DMY`). Without a confirmed date order, `ambiguous_numeric_date` is recorded and the span falls back to raw. This is a **policy rejection**, not a guess invitation.

### Urdu bidi

CLDR's numbering system for Urdu is `arabext`. Bidi control characters must **not** be injected into stored `canonical_text` — handle right-to-left display in the UI renderer only.

---

## 17. Development Setup

```bash
# Install runtime only
pip install -e .
pytest

# Install with grammar authoring + lint/type tools
pip install -e .[dev]
ruff check .
mypy          # Strict mode on runtime/ only
pytest
```

### Generate gRPC stubs

```bash
bash itn_service/service/build_protos.sh
# Or
make -C itn_service protos
```

### Build FAR archives

```bash
python itn_service/compile.py          # Hindi + Marathi
bash itn_service/tools/export_far.sh   # All languages
```

FAR files must exist under `itn_service/compiled_grammars/` before the server starts. Missing FARs cause a loud startup failure — **silent passthrough on missing FARs is not acceptable**.

---

## 18. Current Implementation Status

| Area | Status | Evidence |
|---|---|---|
| Regex prefilter | ✅ Live | `runtime/regex_prefilter.py` |
| Spoken prefilter | ✅ Built | `runtime/spoken_prefilter.py` |
| Lazy WFST factory | ✅ Built | `runtime/wfst_factory.py` |
| WFST classifier | ✅ Built | `runtime/wfst_classifier.py` |
| Phone formatter | ✅ Built | `runtime/formatters/phone_in.py` |
| Date branch (WFST + dateparser) | ✅ Built | `runtime/wfst_classifier._rewrite_date` |
| Self-correction detection | ✅ Built | `runtime/self_correction.py` |
| Lexical cue semantics | ✅ Built | `normalizer._span_has_lex_cue` |
| `StreamState` + stability | ✅ Live | `runtime/stream_state.py` |
| gRPC server | ✅ Live | `service/grpc_server.py` |
| **Live request-path WFST flip** | ⚠️ **Not yet** | `policy.yaml wfst_classifier_enabled: true` set; gateway not integrated |
| Gateway integration | ❌ Pending | See [itn_live_path_gap_analysis.md](itn_live_path_gap_analysis.md) |
| `money`/`currency` taxonomy unification | ❌ Pending | Both names still present in thresholds |
| Display renderer (native digits) | ❌ Stub | `runtime/display_renderer.py` — passthrough |
| IndicLID | ❌ Stub | `script_router.indiclid_predict` raises `NotImplementedError` |
| C++ runtime (Sparrowhawk) | ❌ Future | `cxx_runtime/` scaffolding only |
| CI with real WFST deps | ❌ Pending | Current CI skips grammar-dependent tests |

---

## 19. Rollout Contract

```
flag off  (wfst_classifier_enabled: false):
    request path → regex prefilter only → canonical == raw (passthrough)

flag on, WFST FARs available:
    selected spans rewrite under gate control

flag on, FAR missing for one language:
    that language falls back locally to raw spans;
    other languages continue to work

pipeline exception on any span:
    that span emitted as raw with fallback_reason;
    rest of segment continues

pipeline exception at segment level:
    SegmentResult with raw on all surfaces, deferred=true;
    transcription delivery survives

ITN service outage:
    gateway falls back to raw_text for all surfaces;
    ASR session continues
```

---

## 20. Key Metrics to Monitor

| Metric | What it measures |
|---|---|
| Sentence exact match | End-user transcript quality |
| Class exact match | Per-class normalisation accuracy |
| Digit accuracy (phone, IDs) | Small mistakes are catastrophic here |
| **Unsafe rewrite rate** | Most important business metric |
| Ambiguity deferral rate | System being conservative enough? |
| p50 / p95 normalisation latency | Live-call suitability |
| Partial rollback rate | Streaming UI stability |

For finance / healthcare, additionally track: **wrong decimal point**, **wrong date order**, **wrong account digit**, **wrong dosage/unit**, **lost currency sign**.

---

## 21. What Not to Do

- Do **not** put any LLM or seq2seq model on the live call path.
- Do **not** auto-resolve numeric-only dates without a confirmed locale prior.
- Do **not** store locale-rendered native-digit text as `canonical_text`.
- Do **not** silently transliterate whole utterances before class detection.
- Do **not** compile FARs in the request path.
- Do **not** let missing FARs silently degrade to passthrough — fail loudly at startup.
- Do **not** rewrite high-risk IDs, account numbers, policy numbers, claim codes, medicine dosages, or amounts without both strong context **and** high confidence.
