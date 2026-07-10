# Hindi / Indic ASR Model Research — 2026

**Research date:** 2026-06-13  
**Use case:** Production real-time streaming ASR for noisy call-center audio  
**GPU:** NVIDIA L40S (48 GB VRAM, Ada Lovelace)  
**Hard constraints:**
1. Streaming / online decoding (not 30 s batch windows)
2. 8 kHz telephony band, noisy, code-mixed call-center audio
3. 7 Indian languages: Hindi, Telugu, Marathi, Gujarati, Kannada, Tamil, Bengali
4. **Open-source weights** with a usable license (self-hostable; no paid API)

> Note: this doc relies only on externally published benchmarks (Vistaar, IndicVoices, FLEURS, LAHAJA) and vendor model cards. Internal benchmark numbers are intentionally excluded.

---

## TL;DR

- **Only one OSS model covers all 7 languages with streaming: AI4Bharat IndicConformer 600M (MIT).** It stays the production base.
- **NVIDIA Nemotron 3.5 ASR 0.6B (OpenMDW-1.1)** is a new, excellent *Hindi-only* streaming model (~6.8% FLEURS Hindi) — but supports only Hindi of your 7 Indic languages.
- **BharatGen (MIT)** is a small, fast OSS Hindi option (14.85% Vistaar Hindi); its Shrutam-2 sibling adds 12 Indic languages.
- **Sarvam Saaras/Saarika is NOT open-source** — paid API only. Excluded; kept only as a quality "ceiling" reference (~19% IndicVoices, telephony + code-mixed).
- Best architecture: keep IndicConformer for multilingual coverage, **benchmark Nemotron 3.5 (and BharatGen) as a Hindi specialist lane** routed by language-ID.

---

## OSS-Eligible Shortlist (Hindi-capable + streaming)

| Model | Params | License | Indic coverage | Streaming | Published Hindi WER | L40S fit |
|---|---|---|---|---|---|---|
| **IndicConformer 600M** (ai4bharat) | 600M | MIT | ✅ all 22 Indic | ✅ hybrid CTC/RNNT (causal) | ~10–15% (Vistaar/IndicVoices, clean) | ✅ fits + trains |
| **Nemotron 3.5 ASR 0.6B** (NVIDIA) | 600M | OpenMDW-1.1 | ⚠️ Hindi only | ✅ Cache-Aware FastConformer-RNNT | **6.81%** FLEURS @1.12s; 8.13% @80ms | ✅ fits easily |
| **BharatGen Hindi ASR** | 30M | MIT | ⚠️ Hindi only (Shrutam-2 = 12 langs) | ✅ Branchformer hybrid CTC-RNNT | **14.85%** Vistaar (avg) | ✅ trivially |
| **IndicConformer Hindi** (120M) | 120M | MIT | ⚠️ Hindi only | ✅ hybrid CTC-RNNT | comparable to 600M on Hindi | ✅ |
| **IndicConformer lightweight** | 30M | MIT | per-lang variants | ✅ websocket/on-device | higher WER (tiny) | ✅ edge/android |

All licenses above permit commercial self-hosting. The exact WER values are out-of-the-box on clean/read benchmarks — **none reflect your noisy 8 kHz telephony domain yet** (must be benchmarked locally).

---

## NVIDIA Nemotron 3.5 ASR — Detailed Findings

**`nvidia/nemotron-3.5-asr-streaming-0.6b`, released June 6, 2026.** (Separate from the NVIDIA Nemotron *LLM* family for text — unrelated.)

### Architecture
- **Cache-Aware FastConformer-RNNT**, 24 encoder layers, 600M params
- Processes each audio frame exactly once (no overlap buffering) — true streaming
- Language-ID prompt conditioning: one checkpoint covers 40 language-locales
- 8× depth-wise separable conv subsampling → low VRAM, high throughput

### Streaming latency (runtime-configurable, no retraining)
`80ms → 160ms → 320ms → 560ms → 1.12s` via `att_context_size`. Accuracy improves with larger chunks.

### Hindi WER — FLEURS
| Chunk | WER (with lang input) | WER (auto-detect) |
|---|---|---|
| 80ms | 8.13% | ~8–9% |
| 1.12s | **6.81%** | 8.23% |

**Caveat:** FLEURS is clean read speech. Noisy 8 kHz call-center Hindi WER is **unknown — must benchmark locally.**

### Concurrency
| GPU | Concurrent streams @320ms |
|---|---|
| H100 (published) | 560 |
| L40S (reported real-world) | ~96–200 |
| RTX A5000 (published) | ~5× H100 baseline |

### Indic gap
- ✅ Hindi (hi-IN), transcription-ready tier
- ❌ Telugu, Marathi, Gujarati, Kannada, Tamil, Bengali — **not supported**
- Fine-tuning: NeMo-based, documented for adding languages (~290 h/language in NVIDIA's examples)

### Verdict
Best out-of-the-box Hindi streaming WER of any OSS model here. Disqualified as a *single* production model (1 of 7 languages). Strong as a **Hindi-specialist lane** or **offline Hindi teacher**.

---

## AI4Bharat IndicConformer — Variants

| Variant | Params | Coverage | Use |
|---|---|---|---|
| `indic-conformer-600m-multilingual` | 600M | All 22 Indic languages | **Production base — only all-7-language OSS streamer** |
| Hindi Conformer-Large | 120M | Hindi only | Lighter Hindi-specific |
| Lightweight Conformer | 30M | Per-language | On-device / android, websocket, real-time |

- License: **MIT** (HF model card / aikosh).
- Architecture: `EncDecHybridRNNTCTCBPEModel` — hybrid CTC + RNNT, causal conv encoder (streaming-capable).
- NeMo PEFT adapter path is the proven adaptation route; internal Vaani-replay adapter experiments showed meaningful Hindi WER reduction without multilingual regression (directional, internal only).
- Telephony note: an 8 kHz PEFT path exists in-repo for domain robustness.

---

## BharatGen

- **Hindi ASR:** 30M Branchformer, hybrid CTC-RNNT, trained on 1,000+ h Hindi, **14.85% Vistaar (avg)**, MIT license, ~94 MB PyTorch. Streaming-capable, very cheap to serve. "Restricted visibility" download on aikosh (access request).
- **Shrutam-2:** BharatGen's 12-Indian-language model — covers more of your 7 languages; worth evaluating if you want a single lighter alternative to IndicConformer.
- Trade-off: 30M is tiny → fast and cheap, but capacity-limited vs 600M on hard/noisy audio.

---

## Head-to-Head: the 3 real OSS candidates

| Criterion | IndicConformer 600M | Nemotron 3.5 ASR 0.6B | BharatGen (Hindi 30M) |
|---|---|---|---|
| Streaming | ✅ hybrid CTC/RNNT | ✅ FastConformer-RNNT (best) | ✅ Branchformer CTC-RNNT |
| Hindi WER (clean, published) | ~10–15% | **~6.8%** (FLEURS) | 14.85% (Vistaar) |
| Noisy 8 kHz Hindi WER | unknown (test locally) | unknown (test locally) | unknown (test locally) |
| Covers all 7 Indic langs | ✅ all 22 | ❌ Hindi only | ❌ Hindi only (Shrutam-2: 12) |
| L40S fit + concurrency | ✅ | ✅ ~96–200 @320ms | ✅ very high (tiny) |
| Fine-tuning | NeMo PEFT (proven in-repo) | NeMo full FT (documented) | BharatGen toolchain |
| Telephony adaptation tested | ✅ in-repo path | ❌ | ❌ |
| License | MIT | OpenMDW-1.1 | MIT |
| Maturity | Mature (2023→) | Brand new (Jun 2026) | New |

**Conclusion:** IndicConformer 600M wins on the 7-language requirement. Nemotron 3.5 likely wins on raw Hindi quality + throughput. BharatGen is the cheapest Hindi option. Decide Hindi lane by benchmarking all three on *your* call audio.

---

## Excluded Models (and why)

### Not open-source
- **Sarvam Saaras v3 / Saarika** — **paid API only, no open weights.** Best Indic accuracy on the market (19.31% IndicVoices across 10 langs, telephony + code-mixed, WebSocket streaming, 8 kHz), but it cannot be self-hosted on your L40S and fails the OSS requirement. **Kept only as a target/ceiling reference**, not a deployment candidate.

### No Hindi / Indic support
- **NVIDIA Parakeet (TDT 1.1B / v3)** — English/European only.
- **NVIDIA Canary / Canary-Qwen 2.5B** — tops Open ASR Leaderboard (5.63% avg WER) but English-centric (en/de/es/fr); no Hindi.
- **Kyutai STT (1B / 2.6B)** — streaming, but English/French only.
- **IBM Granite-Speech (3.3-8B / 4.x)** — English ASR only.

### Wrong architecture (not real-time streaming)
- **OpenAI Whisper Large v3 / v3-turbo, distil-whisper** — 30 s batch window; degrades on short telephony chunks. Offline only.
- **IndicWhisper (ai4bharat)** — strong Vistaar Hindi but Whisper-based (offline). **Good offline teacher**, not a live streamer.
- **Qwen3-ASR (0.6B / 1.7B)** — LLM-based, batch; 0.6B has repetition-loop instability. **Teacher candidate** only.
- **Microsoft Phi-4-multimodal (5.6B)** — strong ASR (beats Whisper v3 on OpenASR/FLEURS) but multimodal LLM, not streaming; Hindi support limited.
- **Mistral Voxtral-Mini (4B)** — multilingual but batch, not streaming.
- **Meta MMS-1b-all** — covers Hindi + 1000s of languages, but **CC BY-NC 4.0 (non-commercial) license is a blocker**, and CTC arch is weak for low-latency streaming.

---

## Recommendations

### Option A — Single model (today)
**IndicConformer 600M** (`ai4bharat/indic-conformer-600m-multilingual`, MIT)
- Only OSS model covering all 7 languages with streaming + a proven NeMo PEFT path.
- Already deployed in this repo (Triton → WebSocket gateway).
- Improve via multilingual-replay PEFT, **not** Hindi-only fine-tuning (regression risk).

### Option B — Language-routed hybrid (recommended next step)
- **Hindi calls → Nemotron 3.5 ASR 0.6B** (likely best Hindi WER + throughput), with **BharatGen** as a cheaper Hindi fallback to benchmark against.
- **Other 6 Indic languages → IndicConformer 600M** (only option with coverage).
- Language-ID routing already exists at the gateway; all models fit on the L40S together.
- **Gate:** first benchmark Nemotron 3.5 + BharatGen on your real 8 kHz noisy call audio. If Nemotron's clean-speech edge survives telephony noise, route Hindi to it; otherwise stay single-model.

### Best offline teacher (for pseudo-labeling unlabeled call data)
- **Nemotron 3.5 ASR** (Hindi) — ~7% clean WER, fast.
- **IndicWhisper** — strong Vistaar Hindi, good for clean read-speech labels.
- **Qwen3-ASR-1.7B** — LLM-quality phonetics, stable, but slower.

---

## Next Steps

1. Pull `nvidia/nemotron-3.5-asr-streaming-0.6b` + BharatGen Hindi; benchmark both vs IndicConformer 600M on a held-out **8 kHz noisy call-center Hindi** set (this is the decisive test — published WERs are clean-speech). _Harness + commands: [nemotron_vs_indicconformer_hindi_benchmark.md](nemotron_vs_indicconformer_hindi_benchmark.md) (currently wired for public Hindi: contextual\_asr\_benchmark + FLEURS anchor)._
2. Measure L40S concurrency + latency for each at 160/320 ms chunks.
3. If hybrid wins: wire Hindi→Nemotron/BharatGen, others→IndicConformer via gateway language-ID.
4. Continue multilingual-replay PEFT on IndicConformer for the non-Hindi languages.
5. Use the strongest offline model as a teacher for pseudo-labeling the unlabeled call pool.

---

## Open Questions

- Nemotron 3.5 ASR WER on **noisy 8 kHz call-center Hindi** — must benchmark.
- Does BharatGen Shrutam-2 cover all 7 of your target languages well enough to consolidate?
- Official L40S concurrency for Nemotron 3.5 (NVIDIA published H100 only).
- Will NVIDIA add more Indic languages to Nemotron 3.5? (not announced as of Jun 2026)

---

## Sources

- [nvidia/nemotron-3.5-asr-streaming-0.6b — Hugging Face](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [NVIDIA Nemotron 3.5 ASR release — MarkTechPost (Jun 2026)](https://www.marktechpost.com/2026/06/06/nvidia-releases-nemotron-3-5-asr-a-600m-parameter-cache-aware-streaming-model-transcribing-40-language-locales-in-real-time/)
- [How to Fine-Tune Nemotron 3.5 ASR — NVIDIA HF Blog](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr)
- [Scaling Voice Agents with Cache-Aware Streaming ASR — NVIDIA HF Blog](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)
- [ai4bharat/indic-conformer-600m-multilingual — Hugging Face](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
- [AI4Bharat IndicConformer model card](https://ai4bharat.iitm.ac.in/areas/model/ASR/IndicConformer)
- [BharatGen Hindi ASR — aikosh](https://aikosh.indiaai.gov.in/home/models/details/bharatgen_asr_hindi_4.html) · [BharatGen Speech Models](https://bharatgen.com/speech-models/)
- [Sarvam Saaras v3 / Speech-to-Text (API, non-OSS — reference only)](https://www.sarvam.ai/speech-to-text)
- [Open ASR Leaderboard — arXiv 2510.06961](https://arxiv.org/abs/2510.06961)
- [Vistaar Benchmark — arXiv 2305.15386](https://arxiv.org/pdf/2305.15386) · [LAHAJA — arXiv 2408.11440](https://arxiv.org/pdf/2408.11440)
- [Kyutai STT](https://kyutai.org/stt) · [IBM Granite-Speech — arXiv 2505.08699](https://arxiv.org/pdf/2505.08699) · [Phi-4-multimodal — HF](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
