# Hindi ASR: Improve WER + End-to-End ITN Tokenizer

> Plan / decision record for retraining the Hindi lane to emit fully-normalized written form. Domain: in-house loan-collection calls. Status: **Active decision**, not yet implemented. Drafted 2026-06-26.

Related: [itn_live_path_gap_analysis.md](itn_live_path_gap_analysis.md) · [implementation_blueprint_INR.md](implementation_blueprint_INR.md) · [vaani_adapter_peft.md](vaani_adapter_peft.md) · [decoders.md](decoders.md) · [itn_service_reference.md](itn_service_reference.md)

---

## Context

The production system runs **IndicConformer 600M** (NeMo hybrid RNN-T + CTC). Its Hindi tokenizer is a **256-token Devanagari BPE** with *no digits, ₹, Latin, or punctuation* — so the model physically cannot emit written/normalized text like `₹2,000`, `12/05/2026`, or `9876543210`. ITN today is a separate WFST/Pynini service that is **not on the live path** (it passes text through unchanged — see [itn_live_path_gap_analysis.md](itn_live_path_gap_analysis.md)), so production output is spoken-form Devanagari. Measured in-house WER is **~27–45%** on 8 kHz telephony (vs ~10–15% published clean) — large headroom.

**Goal:** retrain so the **model itself outputs fully-normalized written form**, backed by a **new tokenizer that contains the complete ITN symbol set** (digits / ₹ / Latin / punctuation), and lower WER — using only **10–30 h** of in-house 8 kHz stereo loan-collection calls, ground-truthed by **bootstrap + human correction**, for **Hindi + English code-mix**.

> [!IMPORTANT]
> **10–30 h is light for a model to *learn* ITN from scratch.** The plan de-risks this with one key move: **use the existing WFST grammars to auto-generate the written-form training targets** from human-corrected spoken transcripts. Annotators only ever type natural spoken Devanagari (fast); the WFST produces the `₹/date/phone` normalization the model learns to imitate — plus acoustic/context cues the WFST lacks. A deterministic WFST **safety net stays at inference for money/IDs**: a model can confidently misread a loan amount, and that must not ship unguarded.

## Decisions locked

| Question | Choice |
|---|---|
| ITN | **Bake into model + tokenizer** (end-to-end written form) |
| Labeled data | **10–30 h**, bootstrap + human correction |
| Language scope | **Hindi + English code-mix** (monolingual code-mix model) |
| Output target | Match WFST surface: `₹1,00,000` · `12/05/2026` (DMY) · `9876543210` · `12%`; English as Latin (`EMI`, `bounce charges`); rest Devanagari |
| GPU | Remote **NVIDIA L4 23 GB** — fits 600M with bf16 + frozen lower encoder + grad-accum |

## Timeline

≈8 weeks to first production candidate; **labeling is the critical path.**

| Wk | Phase |
|---|---|
| 1 | P0 Baseline + P1 audio pipeline (stereo split, VAD seg, 8 kHz manifest) |
| 2–5 | **P2 Bootstrap transcription + human correction** (long pole; overlaps P3) |
| 3–5 | P3 WFST: build FARs, extend loan grammars, generate written-form targets |
| 5 | P4 Train new tokenizer + coverage gate |
| 6–7 | P5 `change_vocabulary` + two-phase fine-tune on L4 |
| 7–8 | P6 Eval (written-form WER + entity F1) + iterate |
| 8 | P7 Wire to live path, regen serving assets, Triton export, deploy candidate |

---

## Phase 0 — Baseline & metrics (Wk 1)

- Lock a **human-verified held-out test set** (~1–2 h, transcribed/corrected independently — *not* bootstrap-drafted, *not* WFST-auto) so eval stays unbiased. This is the one place we don't bootstrap.
- Record baseline: current model **WER** + (current spoken-form output) → WFST → entity accuracy, as the bar the new model must beat.

> [!WARNING]
> **WER alone hides ITN quality.** [tools/compute_wer.py](../tools/compute_wer.py) strips punctuation (≈ line 62) and [tools/asr_text_normalizer.py](../tools/asr_text_normalizer.py) removes Latin and `₹` (≈ lines 108–110) — that would erase exactly the ITN we're adding. We need two metrics (see P6).

## Phase 1 — Audio pipeline (Wk 1)

- **Stereo split** — NEW `tools/split_stereo_telephony.py`. Calls are stereo (borrower + agent). The existing [tools/prepare_vaani_8khz_manifest.py](../tools/prepare_vaani_8khz_manifest.py) `to_mono()` *mean-mixes* channels — wrong; it destroys separation. Write per-channel mono export (`*_ch0/_ch1`), reusing [tools/check_original_audio_channels.py](../tools/check_original_audio_channels.py) (ffprobe detection) and importing its `resample_audio` / `write_telephony_wav` helpers.
- **Segmentation** — Silero VAD → 5–20 s utterances (NeMo cap). Reuse the worker's existing VAD.
- **Manifest** — reuse [tools/prepare_vaani_8khz_manifest.py](../tools/prepare_vaani_8khz_manifest.py) to emit 8 kHz WAVs + NeMo JSONL. Training `--sample-rate` stays **16000** — the NeMo preprocessor upsamples 8 k→16 k in the dataloader (telephony-band sim), as [scripts/run_vaani_adapter_peft_8khz.sh](../scripts/run_vaani_adapter_peft_8khz.sh) already does.

## Phase 2 — Bootstrap transcription + correction (Wk 2–5, long pole)

- Draft with current model (**CTC** decoder for speed/timestamps) over segments → spoken-form Devanagari.
- Human-correct in **spoken form** (natural, fast). Author an **annotation guide** (NEW `docs/annotation_guide_hi.md`): code-mix convention (English words in Latin as spoken), numbers as Devanagari words, unclear-audio markers, speaker norms. Consistency here directly bounds model quality.
- Throughput: noisy code-mix correction ≈ 6–8× realtime → ~120–160 human-hours for 20 h → 1–2 annotators over ~3–4 weeks.
- Output: corrected **spoken-form** train manifest (held-out set from P0 stays separate).

## Phase 3 — WFST written-form targets (Wk 3–5, overlaps P2)

- **Build FARs first:** `python -m itn_service.compile --lang hi` → `itn_service/compiled_grammars/hi.far` (without it the pipeline returns raw text). See [itn_service/compile.py](../itn_service/compile.py).
- **Batch converter** — NEW `tools/wfst_generate_written_targets.py`: spoken-form JSONL → written form. No batch entry exists today. Per [itn_service/runtime/normalizer.py:252](../itn_service/runtime/normalizer.py#L252), call `normalize_segment(raw_text, tokens=[], is_final=True, state=StreamState(), lang_hint="hi", locale_policy=...)` and read `.canonical_text`.

> [!IMPORTANT]
> **Must inject** `make_wfst_classifier(TenantPolicy(date_order="DMY"))` from [itn_service/runtime/wfst_classifier.py](../itn_service/runtime/wfst_classifier.py) — the *default* classifier is passthrough (`canonical == raw`, see [normalizer.py:75-84](../itn_service/runtime/normalizer.py#L75-L84)). Without the injection the converter emits the raw spoken text.

- **Extend Hindi grammars** for loan-collection gaps (outstanding amount, EMI count, due date, installment) in [itn_service/grammars/hi/](../itn_service/grammars/hi/).
- **Human-verify** a sample (~500 utts) of WFST output; flag `deferred` / `fallback_reason` spans (WFST is conservative and refuses ambiguous cases) for correction. This sample also measures the target-quality ceiling.
- Output: parallel manifest with `text` (spoken) + `text_written` (training target).

## Phase 4 — New tokenizer with full ITN symbol set (Wk 5)

- NEW `tools/build_written_form_tokenizer.py` wrapping NeMo's bundled `scripts/tokenizers/process_asr_text_tokenizer.py` (no tokenizer trainer exists in-repo). Train a **single monolingual SentencePiece BPE** (`hi_en_written`, vocab ~768) on the **written-form** corpus. Critical flags:
  - `--spe_split_digits=true` — each digit its own token (prevents digit hallucination / merged-number errors).
  - `--spe_character_coverage=1.0` — ensures `₹` (U+20B9) and rare symbols survive.
- Going monolingual deliberately drops the multilingual `multisoftmax` / `language_masks` machinery (see [tools/run_nemo_adapter_peft.py](../tools/run_nemo_adapter_peft.py), the local-target-remap logic ≈ lines 982–1240) — correct for a Hindi+EN deployment.
- **Coverage gate:** run existing [tools/check_tokenizer_coverage.py](../tools/check_tokenizer_coverage.py) `--model … --manifest written.jsonl`; require `failed_rows: 0` (it reports missing characters via the decoder-vocabulary fallback, ≈ lines 119–128).

## Phase 5 — Vocab swap + fine-tune (Wk 6–7, L4)

NEW `tools/run_written_form_finetune.py`, forked from [tools/run_stable_nemo_finetune.py](../tools/run_stable_nemo_finetune.py). Neither existing script can change vocab; adapters *assert* it's unchanged ([run_nemo_adapter_peft.py](../tools/run_nemo_adapter_peft.py) ≈ line 1980). Insert after `ASRModel.restore_from` (≈ line 716), before `setup_training_data` (≈ line 791), **two** calls (hybrid rebuilds each head separately):

```python
model.change_vocabulary(new_tokenizer_dir=TOK, new_tokenizer_type="bpe", decoder_type="rnnt")  # decoder + joint
model.change_vocabulary(new_tokenizer_dir=TOK, new_tokenizer_type="bpe", decoder_type="ctc")   # aux_ctc head
```

Then via `open_dict`: `decoder.multisoftmax=False`, `joint.multilingual=False`, `train_ds.return_language_id=False` (mirrors [worker/app/nemo_export.py](../worker/app/nemo_export.py) ≈ lines 184/193); rebuild WER/CER metrics.

- **Augmentation (must ADD):** only SpecAugment is active today. Add a `cfg.train_ds.augmentor` block — speed-perturb 0.9/1.0/1.1 (≈3× data) + noise at SNR 5–20 dB — to stretch 10–30 h.
- **Two-phase recipe** (output head is randomly reinitialized; base `--lr 5e-6` is too low for that):
  - **Phase A** head warm-up: `--freeze-encoder-fraction 0.9`, LR ~1e-4–3e-4, short.
  - **Phase B** joint FT: `--freeze-encoder-fraction 0.3–0.5`, LR ~5e-6, `--resume-auto`, early stop `--patience`.
- Reuse existing flags: `--use-duration-bucketing` / `--bucket-edges`, `--curriculum-first-epoch`, `--accumulate-grad-batches`, `--gradient-clip-val`, `--precision` (bf16; the 8 kHz script defaults `32-true` if unstable). L4 23 GB: batch 1–2 + grad-accum.

## Phase 6 — Evaluation (Wk 7–8)

- **(a) Written-form WER:** add a punctuation/Latin-preserving mode — use the existing `--reference-normalization raw` path in [tools/eval_ws_manifest_wer.py](../tools/eval_ws_manifest_wer.py) (the default `vaani` mode would grade a written-form model against a stripped spoken reference).
- **(b) Entity accuracy** — NEW `tools/eval_entity_accuracy.py`, the real ITN metric. Reuse [itn_service/runtime/regex_prefilter.py](../itn_service/runtime/regex_prefilter.py) `prefilter()` to extract typed spans (money/percent/phone/date/time/PAN/Aadhaar/IFSC) from hypothesis vs reference; report **per-class exact-match precision/recall/F1**. One wrong phone digit = a miss WER can't see.
- **Gate before ship:** new model must beat baseline on WER **and** match-or-exceed the spoken+WFST baseline on **money/account entity F1**. If amounts regress → keep those classes on the WFST safety net (hybrid) rather than trusting the model.

## Phase 7 — Wire to live path + deploy (Wk 8)

- Model now emits written form directly; gateway no longer needs WFST for the classes the model handles — but **keep WFST as a validating safety net for money + IDs** (re-route just those spans through the [itn_service](../itn_service/) validators).
- **Regenerate serving assets:** the new vocab invalidates `assets/vocab.json` + `assets/language_masks.json` (loaded in [worker/app/triton.py](../worker/app/triton.py) ≈ line 502; the CTC fast path masks the logit vector ≈ lines 410–420). Rebuild both or the Triton CTC path breaks.
- Export via [worker/app/nemo_export.py](../worker/app/nemo_export.py); smoke-test streaming with [tools/ws_client_send_wav.py](../tools/ws_client_send_wav.py).

---

## New code (all additive)

| File | Purpose |
|---|---|
| `tools/split_stereo_telephony.py` | Per-channel mono export (don't mean-mix) |
| `tools/wfst_generate_written_targets.py` | Batch spoken→written via injected WFST classifier |
| `tools/build_written_form_tokenizer.py` | Train BPE w/ `--spe_split_digits` + coverage 1.0 |
| `tools/run_written_form_finetune.py` | Fork w/ dual `change_vocabulary` + augmentor + 2-phase |
| `tools/eval_entity_accuracy.py` | Per-class entity P/R/F1 (the ITN metric) |
| `docs/annotation_guide_hi.md` | Code-mix / number / markup conventions for correctors |

## Reuse (don't rebuild)

[tools/prepare_vaani_8khz_manifest.py](../tools/prepare_vaani_8khz_manifest.py) · [tools/check_original_audio_channels.py](../tools/check_original_audio_channels.py) · [tools/check_tokenizer_coverage.py](../tools/check_tokenizer_coverage.py) · [tools/run_stable_nemo_finetune.py](../tools/run_stable_nemo_finetune.py) (fork base) · [itn_service/runtime/normalizer.py](../itn_service/runtime/normalizer.py) + [wfst_classifier.py](../itn_service/runtime/wfst_classifier.py) + [regex_prefilter.py](../itn_service/runtime/regex_prefilter.py) · [itn_service/compile.py](../itn_service/compile.py) · [tools/eval_ws_manifest_wer.py](../tools/eval_ws_manifest_wer.py) · [worker/app/nemo_export.py](../worker/app/nemo_export.py)

## Key risks

| Risk | Mitigation |
|---|---|
| **Digit hallucination / wrong amounts** | `--spe_split_digits`, entity-F1 gate, WFST safety net on money/IDs at inference |
| **Catastrophic forgetting** (both heads reinit) | Two-phase freeze, low Phase-B LR, early stop, augmentation |
| **Reinit-head instability** | Phase-A warm-up at higher LR, grad clip, fall back to `--precision 32-true` |
| **Inconsistent targets** (conservative WFST defers) | Human-review deferred spans before training |
| **Serving break** | Regenerate `vocab.json` / `language_masks.json` in P7 |

## Verification (end-to-end)

1. [tools/check_tokenizer_coverage.py](../tools/check_tokenizer_coverage.py) on the written manifest → `failed_rows: 0`.
2. Offline eval on the **human-verified** held-out set: written-form WER (raw normalization) **and** `eval_entity_accuracy.py` per-class F1, both vs baseline.
3. Streaming check via [tools/ws_client_send_wav.py](../tools/ws_client_send_wav.py) → confirm live output shows `₹/date/phone` written form.
4. Ship gate: WER improved **and** money/account F1 ≥ baseline (else hybrid-guard those classes).
