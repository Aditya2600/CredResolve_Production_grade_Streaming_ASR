# Dynamic Context Biasing Demo

This repo now supports additive, request-scoped context biasing on top of the existing static language phrase files.

## Request shape

The existing WebSocket handshake stays unchanged. For demo use, send an optional control message before audio for a session:

```json
{
  "type": "session_config",
  "context_biasing": {
    "enabled": true,
    "mode": "shadow"
  },
  "biasing_context": {
    "debtor_name": "Ravi Kumar",
    "agent_name": "Priya Sharma",
    "lender": "SMFG India Credit",
    "product": "Personal loan",
    "city": "Jaipur",
    "branch": "MI Road",
    "account_terms": ["loan id", "payment link"],
    "prior_call_entities": ["NACH", "bounce charges"],
    "campaign_vocabulary": ["settlement", "callback"],
    "amounts": ["Rs 12,500", "1 lakh"],
    "dates": ["21/04/2026", "30 April 2026"]
  }
}
```

The gateway forwards this internally to the worker as a single additive request header. Existing clients that never send `session_config` continue to work as-is.

## Demo usage

1. Start the stack the same way as today. If you want NeMo context biasing enabled, keep using the existing context-biasing worker image and env vars.
2. In the frontend, open the `Demo Context Biasing` panel.
3. Turn on `Attach dynamic context`.
4. Pick `shadow` for safe demo evaluation or `active` to let the worker return the biased result when the phrase-hit guard accepts it.
5. Fill any subset of the fields. List fields accept comma-separated entries, and numeric commas inside amounts such as `8,450` are preserved.
6. Start a mic or WAV session. The panel will show the attached phrase count, selected source, and top ranked phrases when the response comes back.

## Phrase-pack behavior

- Static language phrase files still load from `ASR_CONTEXT_BIASING_PHRASES_DIR` and remain the default base lexicon.
- If dynamic context is present, the worker builds a request-scoped phrase pack by:
  - normalizing the provided fields
  - generating conservative text, spacing, amount, and date variants
  - ranking candidates
  - pruning dynamic candidates to `ASR_CONTEXT_BIASING_DYNAMIC_MAX_PHRASES`
  - merging the selected dynamic phrases with the static language file when it exists
- `debtor_name` ranks highest, followed by `lender` and `product`, then amounts/dates/account terms, then city/branch/agent, then prior-call or campaign vocabulary.
- Short ambiguous entries are penalized before pruning.

## Modes and fallback

- No dynamic context: the old behavior is preserved exactly. The worker uses the existing static phrase-file path or skips biasing for the same legacy reasons.
- `shadow`: the worker runs the biased decode, logs the candidate transcript and diagnostics, and still returns the baseline transcript.
- `active`: the worker keeps the existing conservative guard. It only returns the biased transcript when the phrase-hit comparison says the candidate is at least as good as baseline.
- Static file missing with no dynamic context: same legacy fallback, biasing is skipped.
- Static file missing with dynamic context: the worker can still build a dynamic-only request pack for demo use.
- Any biasing error or timeout falls back to the baseline transcript.

## Config

Existing env vars still apply:

- `ASR_CONTEXT_BIASING_MODE`
- `ASR_CONTEXT_BIASING_METHOD`
- `ASR_CONTEXT_BIASING_NEMO_SOURCE`
- `ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS`
- `ASR_CONTEXT_BIASING_PHRASES_DIR`
- `ASR_CONTEXT_BIASING_TIMEOUT_MS`
- `ASR_CONTEXT_BIASING_DEVICE`
- `ASR_CONTEXT_BIASING_SHADOW_SAMPLE_RATE`
- `ASR_CONTEXT_BIASING_BEAM_THRESHOLD`
- `ASR_CONTEXT_BIASING_CONTEXT_SCORE`
- `ASR_CONTEXT_BIASING_CTC_ALI_TOKEN_WEIGHT`

New env var:

- `ASR_CONTEXT_BIASING_DYNAMIC_MAX_PHRASES`
  - default: `32`
  - controls the request-scoped dynamic top-k cap before merging with the static base file

## Observability

Structured worker and gateway logs now include:

- effective mode and requested mode
- whether dynamic context was attached or used
- provided fields
- phrase counts before and after pruning
- top ranked phrases
- phrase source (`static`, `dynamic_only`, `dynamic_merged`)
- baseline transcript, biased transcript, selected transcript
- latency overhead and fallback/error reasons
