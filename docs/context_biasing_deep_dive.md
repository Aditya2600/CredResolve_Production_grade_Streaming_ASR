# Context Biasing / Domain Biasing — Deep Dive

## Overview

Context biasing (also called domain biasing) is the system that steers ASR decoding toward vocabulary that is known to be relevant for a specific call. In a debt-collection context this means names, lenders, loan products, amounts, dates, and domain jargon that an acoustic model trained on general speech may otherwise miss or mis-transcribe.

The implementation wraps NeMo's CTC word-synchronous (CTC-WS) beam search decoder. Before decoding begins, the decoder is told which phrases to prefer; those phrases receive a log-probability bonus so the beam explorer favors token paths that spell them out. Everything else about decoding is unchanged.

---

## Architecture at a Glance

```
Request (audio + metadata)
        │
        ▼
  Gateway: parse biasing_context + requested mode
        │
        ▼
  Worker: decide() — eligibility check
        │
        ├── ineligible → return baseline only
        │
        └── eligible
              │
              ├── build_request_scoped_phrase_pack()
              │       ├── load static phrase file (hi.txt / ...)
              │       ├── generate variants per field
              │       ├── rank + prune (top-32)
              │       └── merge into one phrase file
              │
              ├── _apply_decoding_strategy()  [NeMo change_decoding_strategy]
              │
              ├── transcribe_pcm16()          [NeMo model.transcribe]
              │
              └── should_return_active_biasing_transcript()
                      ├── active mode  → return biased if no regression
                      └── shadow mode  → return baseline, log both
```

---

## Key Source Files

| File | Role |
|---|---|
| [worker/app/context_biasing.py](../worker/app/context_biasing.py) | Runtime: model loading, eligibility, NeMo decoding config, inference |
| [worker/app/context_assembler.py](../worker/app/context_assembler.py) | Phrase pack assembly: parse context, build candidates, merge with static file |
| [worker/app/phrase_ranker.py](../worker/app/phrase_ranker.py) | Candidate scoring, deduplication, top-k selection |
| [worker/app/variant_generator.py](../worker/app/variant_generator.py) | Per-field surface form expansion (amounts, dates, names, Hindi script) |
| [worker/app/hindi_transliteration.py](../worker/app/hindi_transliteration.py) | Latin → Devanagari phoneme mapping for name/keyword variants |
| [worker/app/config.py](../worker/app/config.py) | All `ASR_CONTEXT_BIASING_*` env-var defaults |
| [context_biasing/phrases/hi.txt](../context_biasing/phrases/hi.txt) | Static Hindi domain vocabulary (underscore-delimited variant groups) |

---

## Modes

Context biasing has three operating modes, set per-deployment via `ASR_CONTEXT_BIASING_MODE` and overridable per-request.

### `disabled`
No biasing at all. The NeMo model is not loaded; `decide()` short-circuits immediately. Zero latency overhead.

### `shadow`
The biased decode runs in parallel with the baseline, but the baseline transcript is always returned to the caller. Both transcripts are logged in structured events (`transcribe_context_biasing_result`) so teams can measure phrase recall and latency impact before enabling active mode.

Shadow requests are sampled: `ASR_CONTEXT_BIASING_SHADOW_SAMPLE_RATE` (default `1.0`) controls the fraction. Sampling is deterministic — SHA-256 of `utterance_id` (or `session_id` as fallback) is compared against the rate, so a given utterance always lands in the same bucket.

### `active`
The biased transcript is returned to the caller when it passes the quality gate (see [Quality Gate](#quality-gate) below). If the gate fails (phrase regression, empty candidate, timeout), the system falls back to the baseline without surfacing an error.

---

## Eligibility Decision (`decide()`)

[context_biasing.py:390](../worker/app/context_biasing.py#L390)

Before any inference is attempted, `decide()` returns a `ContextBiasingDecision` that specifies:

- `eligible: bool` — whether to proceed
- `reason: str` — human-readable skip/proceed cause
- `phrase_file: str | None` — path to the merged phrase file to use

Ineligibility reasons (in evaluation order):

| Reason | Condition |
|---|---|
| `request_disabled` | Per-request override set mode to `disabled` |
| `disabled` | Deployment-level mode is `disabled` |
| `language_auto` | Requested language is `auto`; no language-specific phrase file can be selected |
| `not_ready` | Model failed to load at startup |
| `missing_phrase_file` | No static phrase file for the language AND no dynamic context provided |
| `shadow_unsampled` | Shadow mode but utterance/session hash falls outside the sample rate |
| `resolved_language_mismatch` | Language detected by LID differs from requested language (checked post-baseline) |

---

## Phrase Pack Assembly

### Static Phrase File

[context_biasing/phrases/hi.txt](../context_biasing/phrases/hi.txt) contains language-specific domain vocabulary. Each line is an underscore-delimited group of variant surface forms. The first field is the canonical form; subsequent fields are accepted alternative spellings or transliterations:

```
emi_e m i_ईएमआई_इएमआई
bounce charges_bounce charge_बाउंस चार्जेस_बाउंस चार्ज
cred resolve_credresolve_क्रेड रिजॉल्व_क्रेडरिजॉल्व
```

This file is loaded once at startup (via `PhraseLexicon.from_file`) and used for the quality gate. At runtime it is also loaded by `_load_phrase_groups` to serve as the base into which dynamic phrases are merged.

### Dynamic Phrase Generation

When a request includes a `biasing_context` dict, [context_assembler.py:221](../worker/app/context_assembler.py#L221) runs `build_request_scoped_phrase_pack()`:

1. **Parse** the context dict into a typed `BiasingContext` dataclass. Scalar fields (one value) and list fields (multiple values) are handled separately.

2. **Generate variants** per field type:

   - **General fields** (`debtor_name`, `lender`, `product`, `city`, `branch`, `agent_name`, `account_terms`, `prior_call_entities`, `campaign_vocabulary`): display form, lowercase, joined-alphanumeric form, space-separated form for all-caps acronyms, and Hindi Devanagari transliteration when `language=hi`.

   - **Amounts** (`amounts`): all general variants, plus numeric extraction (strip ₹/Rs/INR), comma-formatted integer (`12,500`), Indian English words (`twelve thousand five hundred`), "N rupees" form, and Hindi Devanagari digit/currency variants.

   - **Dates** (`dates`): all general variants, plus multiple date format strings (`%Y-%m-%d`, `%d/%m/%Y`, `%d-%m-%Y`, `%d %B %Y`, ...), and Hindi Devanagari date variants with month names.

3. **Rank** all `PhraseCandidate` objects via `rank_phrase_candidates()`. See [Ranking](#ranking) below.

4. **Prune** to `max_dynamic_phrases` (default 32).

5. **Merge** selected dynamic phrases into the base phrase groups loaded from the static file. Duplicates (by normalized key) are skipped.

6. **Render** to `tuple[str, ...]` of underscore-delimited lines, written to a temp file for the decoder.

### BiasingContext Fields and Their Priority

```python
# Scalar (one value per field)
debtor_name, agent_name, lender, product, city, branch

# List (zero or more values)
account_terms, prior_call_entities, campaign_vocabulary, amounts, dates
```

Comma parsing for list fields is smart: commas between digits are treated as thousand separators, not list delimiters (so `12,500` stays as one item).

---

## Ranking

[phrase_ranker.py:85](../worker/app/phrase_ranker.py#L85)

Candidates are first deduplicated by normalized key. When two candidates share a key, their variant lists are merged and the field with higher weight wins for scoring.

Each surviving candidate is scored:

```
score = FIELD_WEIGHTS[field]
      + min(token_count, 4) × 2.0   # length bonus, capped at 4 tokens
      + 3.0                          # if multi-token
      + 1.5                          # if contains any digit
      + 1.0                          # if explicit (always True for dynamic context)
      - ambiguity_penalty(canonical)
```

**Field weights** (higher = more important):

| Field | Weight |
|---|---|
| `debtor_name` | 120.0 |
| `lender` | 105.0 |
| `product` | 100.0 |
| `amounts` | 90.0 |
| `dates` | 88.0 |
| `account_terms` | 86.0 |
| `city` | 72.0 |
| `branch` | 70.0 |
| `agent_name` | 68.0 |
| `prior_call_entities` | 58.0 |
| `campaign_vocabulary` | 54.0 |

**Ambiguity penalties** (subtracted from score):

| Condition | Penalty |
|---|---|
| Joined text ≤ 2 chars | −70.0 |
| Joined text ≤ 3 chars | −40.0 |
| Single token ≤ 4 chars | −22.0 |
| Single token in generic set (`loan`, `payment`, ...) | −26.0 |
| Single digit token ≤ 2 digits | −20.0 |

Generic tokens that trigger the ambiguity penalty: `account`, `agent`, `amount`, `branch`, `call`, `city`, `date`, `loan`, `name`, `payment`, `product`.

Sorted descending by `(-score, -token_count, canonical)`.

---

## Decoder Configuration

[context_biasing.py:576](../worker/app/context_biasing.py#L576)

`_build_decoding_cfg()` constructs an OmegaConf config from the model's existing decoding config and overlays the biasing parameters:

```python
decoding_cfg.apply_context_biasing = True
decoding_cfg.context_file          = "<path to merged phrase file>"
decoding_cfg.beam_threshold        = 8.0   # ASR_CONTEXT_BIASING_BEAM_THRESHOLD
decoding_cfg.context_score         = 3.0   # ASR_CONTEXT_BIASING_CONTEXT_SCORE
decoding_cfg.ctc_ali_token_weight  = 0.6   # ASR_CONTEXT_BIASING_CTC_ALI_TOKEN_WEIGHT
```

**`context_score`** is the single most impactful tuning knob. It is a log-probability bonus added to every CTC emission that matches a context phrase token. Higher values increase recall of biased phrases but also increase substitution errors on out-of-vocabulary words that phonetically resemble a phrase. The default `3.0` is conservative; values in the range `2.0–5.0` are typical in production.

**`beam_threshold`** controls beam pruning width during the WS beam search. A larger threshold retains more hypotheses (higher recall, more compute); a smaller threshold prunes aggressively.

**`ctc_ali_token_weight`** weights the CTC alignment path relative to the context graph path. Lower values give more weight to the context graph (phrase boost); higher values trust the acoustic model more.

`_apply_decoding_strategy()` calls `model.change_decoding_strategy(decoding_cfg)` with appropriate kwargs inferred from the model's method signature.

---

## Inference Flow

[context_biasing.py:660](../worker/app/context_biasing.py#L660)

`transcribe_pcm16()`:

1. Convert raw PCM-16 bytes → float32 numpy array, normalized to `[-1, 1]`.
2. Linear resample from request sample rate to the model's native sample rate (typically 16 kHz).
3. Write a temporary WAV file.
4. Apply the decoding strategy (set phrase file + params).
5. Call `model.transcribe([wav_path], ...)`.
6. Delete the temp WAV.
7. Return `ContextBiasingResult(text, language, phrase_file, latency_ms)`.

Each request now leases one model from `ContextBiasingModelPool` before applying the decoding strategy. This is the safety boundary: `change_decoding_strategy()` mutates decoder state, so concurrent requests must use independent NeMo model instances rather than one shared mutable model. Defaults remain serialized (`pool_size=1`, `max_concurrent=1`, `executor_workers=1`).

Queue wait and inference timeouts are separate. If no model lease arrives before `ASR_CONTEXT_BIASING_QUEUE_TIMEOUT_MS`, the request falls back with `queue_timeout`. If an already-leased decode exceeds `ASR_CONTEXT_BIASING_TIMEOUT_MS`, it falls back with `inference_timeout`; the slot is marked `draining_after_timeout` and is not returned to the pool until the underlying thread actually exits.

---

## Quality Gate

[context_biasing.py:176](../worker/app/context_biasing.py#L176)

`should_return_active_biasing_transcript()` decides in active mode whether to return the biased or baseline transcript.

The gate uses a greedy longest-match phrase counter (`PhraseLexicon.count_terms`) over the merged phrase file to count how many known domain phrases appear in each transcript.

| Condition | Decision | `selection_reason` |
|---|---|---|
| Biased text is empty | Use baseline | `empty_candidate` |
| No lexicon available | Use biased | `no_lexicon` |
| Both have 0 phrase hits | Use baseline | `no_phrase_gain` |
| Biased < Baseline phrase hits | Use baseline | `phrase_regression` |
| Biased == Baseline phrase hits | Use biased | `phrase_preserved` |
| Biased > Baseline phrase hits | Use biased | `phrase_gain` |

The gate is intentionally conservative: if biasing does not increase or at least preserve phrase recall it is rejected. This prevents hallucination of domain vocabulary into utterances where the model is uncertain.

In shadow mode, `selection_reason` is always `shadow_mode` and the baseline is always returned regardless of phrase hit comparison.

---

## Observability

Every biased inference emits a structured log event (`transcribe_context_biasing_result` or `transcribe_context_biasing_skipped`) with the following fields:

```
bias_mode              shadow | active | disabled
bias_method            ctc_ws
status                 ok | empty_candidate | fallback
returned_source        baseline | biased
selection_reason       phrase_gain | phrase_preserved | phrase_regression |
                       no_phrase_gain | empty_candidate | shadow_mode | ...
baseline_phrase_hits   int
biased_phrase_hits     int
returned_phrase_hits   int
bias_latency_ms        int
phrase_source          static | dynamic_only | dynamic_merged
dynamic_context_present bool
dynamic_context_used   bool
fields_provided        list of field names present in the request
phrase_count_before_pruning  int
phrase_count_after_pruning   int
total_phrase_count     int
top_phrases            list[str] (top-8 ranked phrases)
baseline_transcript    str
biased_transcript      str
selected_transcript    str
```

Every pool attempt also logs:

```
bias_queue_wait_ms
bias_inflight_count
bias_max_concurrency
bias_timeout_reason
configured_max_concurrency
effective_max_concurrency
model_pool_size
```

Prometheus metrics:
- `CONTEXT_BIASING_REQUESTS{mode, status}` — counter
- `CONTEXT_BIASING_FALLBACKS{reason}` — counter (`queue_timeout` and `inference_timeout` are distinct)
- `CONTEXT_BIASING_LATENCY` — inference-only histogram
- `CONTEXT_BIASING_TOTAL_LATENCY` — caller-visible queue + inference histogram
- `CONTEXT_BIASING_QUEUE_WAIT_MS` — queue-wait histogram
- `CONTEXT_BIASING_INFLIGHT` — active or draining leases
- `CONTEXT_BIASING_POOL_AVAILABLE` — currently leaseable models

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `ASR_CONTEXT_BIASING_MODE` | `disabled` | `disabled` / `shadow` / `active` |
| `ASR_CONTEXT_BIASING_METHOD` | `ctc_ws` | Decoding method (only `ctc_ws` supported) |
| `ASR_CONTEXT_BIASING_NEMO_SOURCE` | `""` | NeMo model path or HuggingFace repo |
| `ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS` | `""` | E.g., `EncDecCTCModelBPE` |
| `ASR_CONTEXT_BIASING_PHRASES_DIR` | `""` | Directory containing `<lang>.txt` phrase files |
| `ASR_CONTEXT_BIASING_TIMEOUT_MS` | `4000` | Max inference time before fallback |
| `ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES` | `1` | Requested biased decodes in flight; clamped to pool size |
| `ASR_CONTEXT_BIASING_EXECUTOR_WORKERS` | max concurrent | Dedicated context-biasing worker threads |
| `ASR_CONTEXT_BIASING_QUEUE_TIMEOUT_MS` | inference timeout | Max wait for a model lease before `queue_timeout` fallback |
| `ASR_CONTEXT_BIASING_MODEL_POOL_SIZE` | `1` | Number of independent NeMo model instances |
| `ASR_CONTEXT_BIASING_MODEL_POOL_LOAD_MODE` | `eager` | `eager` loads all slots at startup; `lazy` loads on first lease |
| `ASR_CONTEXT_BIASING_DEVICE` | `cuda` | `cuda` or `cpu` |
| `ASR_CONTEXT_BIASING_SHADOW_SAMPLE_RATE` | `1.0` | Fraction of shadow requests to run (0.0–1.0) |
| `ASR_CONTEXT_BIASING_BEAM_THRESHOLD` | `8.0` | CTC-WS beam pruning threshold |
| `ASR_CONTEXT_BIASING_CONTEXT_SCORE` | `3.0` | Log-prob boost per phrase token |
| `ASR_CONTEXT_BIASING_CTC_ALI_TOKEN_WEIGHT` | `0.6` | Alignment path weight vs context graph |
| `ASR_CONTEXT_BIASING_DYNAMIC_MAX_PHRASES` | `32` | Max dynamic phrases before static merge |

---

## Request API

Biasing context is passed per-request in the session config (WebSocket or HTTP):

```json
{
  "context_biasing": {
    "mode": "active"
  },
  "biasing_context": {
    "debtor_name": "Ramesh Kumar",
    "lender": "HDFC Bank",
    "product": "personal loan",
    "amounts": ["50000", "₹12,500"],
    "dates": ["21/04/2026"],
    "account_terms": ["EMI", "bounce charge"],
    "city": "Pune"
  }
}
```

`context_biasing.mode` can override the server-side mode on a per-request basis. Setting it to `"disabled"` suppresses biasing for that request even if the server is in `active` mode.

---

## Static Phrase File Format

Files live at `{ASR_CONTEXT_BIASING_PHRASES_DIR}/{language}.txt` (e.g., `phrases/hi.txt`).

Rules:
- One phrase group per line.
- Fields are separated by `_` (underscore).
- First field is the canonical form shown in logs; remaining fields are alternate surface forms accepted during phrase-hit counting.
- Lines starting with `#` are comments.
- Unicode is supported; Devanagari and Latin scripts can be mixed within one group.

Example:
```
aadhaar_aadhar_आधार
emi_e m i_ईएमआई_इएमआई
cred resolve_credresolve_क्रेड रिजॉल्व_क्रेडरिजॉल्व
bounce charges_bounce charge_बाउंस चार्जेस_बाउंस चार्ज
```

---

## Tuning Guide

### Increasing phrase recall (missing domain terms)

1. Add missing phrases to `phrases/hi.txt` with all known surface forms.
2. Increase `ASR_CONTEXT_BIASING_CONTEXT_SCORE` (e.g., from `3.0` → `4.0`). Monitor for false-positive phrase substitutions.
3. Widen beam search by increasing `ASR_CONTEXT_BIASING_BEAM_THRESHOLD`.

### Reducing false positives (wrong words substituted)

1. Lower `ASR_CONTEXT_BIASING_CONTEXT_SCORE`.
2. Remove overly short or generic entries from the static phrase file.
3. Rely on the ambiguity penalty in ranking — very short or single-token candidates are penalized automatically.

### Latency

The biasing model runs synchronously on GPU inside a dedicated executor. Typical latency for a 30-second utterance at 16 kHz on a mid-range GPU is 200–600 ms. Keep queue wait and inference latency separate when tuning: queue pressure means pool capacity is exhausted, while inference timeout means the leased decode itself is too slow. `tools/context_biasing_loadtest.py` reports queue, inference, total latency, fallback rate, and GPU memory by pool-size group.

### Shadow → Active rollout

1. Start with default serialized behavior: pool size `1`, max concurrency `1`, executor workers `1`.
2. Deploy with `ASR_CONTEXT_BIASING_MODE=shadow` and `ASR_CONTEXT_BIASING_SHADOW_SAMPLE_RATE=1.0`.
3. Validate phrase isolation first: no request-specific phrase file should appear in another request's decode logs.
4. In staging, test `pool_size=2`, `max_concurrent=2`, `executor_workers=2`; optionally test `4/4/4` only if GPU memory allows.
5. Analyze `transcribe_context_biasing_result` events plus timeout/fallback metrics.
6. Do not promote to `active` until phrase-leakage tests and shadow logs are clean.

---

## Phrase Count Metrics Explained

| Metric | Meaning |
|---|---|
| `phrase_count_before_pruning` | Number of unique dynamic phrase candidates generated from context fields before top-k selection |
| `phrase_count_after_pruning` | Candidates retained after top-32 cut |
| `base_phrase_count` | Entries that came from the static phrase file only |
| `total_phrase_count` | Final merged count (static + dynamic, deduplicated) — this is the decoder's input |
| `top_phrases` | Top-8 ranked dynamic phrases (by score), logged for observability |

---

## Common Skip Reasons and Fixes

| `reason` in logs | Cause | Fix |
|---|---|---|
| `language_auto` | Language detection is set to auto; can't select phrase file | Send an explicit `language` in the request |
| `missing_phrase_file` | No `{lang}.txt` exists in phrases dir and no dynamic context | Add static phrase file or provide `biasing_context` |
| `shadow_unsampled` | Hash-based sampling excluded this utterance | Expected behavior; adjust `SHADOW_SAMPLE_RATE` if too few samples |
| `not_ready` | Model failed to initialize | Check `ASR_CONTEXT_BIASING_NEMO_SOURCE` path and GPU availability |
| `resolved_language_mismatch` | LID detected a different language than requested | Improve LID confidence or narrow language scope |
| `phrase_regression` | Biased transcript lost phrases vs baseline | Inspect biased transcript; may need higher `context_score` or better phrase variants |
| `no_phrase_gain` | Neither transcript contains any known phrase | Audio may not contain domain vocabulary; verify phrase file coverage |
