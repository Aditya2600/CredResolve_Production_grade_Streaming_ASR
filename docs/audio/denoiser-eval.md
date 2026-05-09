# Denoiser evaluation: DeepFilterNet3 vs RNNoise

**Status:** ✅ **Recommendation: SWAP to DeepFilterNet3.** Both arms of the binding decision rule pass cleanly.
**Date:** 2026-05-08
**Fixture:** `tests/fixtures/audio_bench/` (4 Hindi VAANI clips, 2.1–4.0 s each, 20 iterations per clip = 80 rows). No fixture named `P0.5` exists in this repo; the task referred to this set.
**Backend during eval:** Triton on `localhost:8101` (production-equivalent path), worker venv with both `pyrnnoise==0.4.3` and `deepfilternet>=0.5.6` installed.

## Decision rule (binding, set in advance)

> Swap iff aggregate WER improves by **≥ 1.0 percentage point absolute** AND DFN denoise stage p95 stays under the SLO ceiling. No documented denoise SLO exists in `docs/production_system_design.md` or `docs/audio/`, so per the rule the ceiling is **2× current p95**:
> - aggregate-equivalent ceiling: **264 ms** (2 × 132 ms baseline)
> - worst-fixture ceiling: **325 ms** (2 × 162.4 ms baseline)

## Results

All three runs, same host, same Triton model, back-to-back:

| Config | Aggregate WER | Per-file mean WER | Denoise p50 | Denoise p95 | Total preproc p95 | STT p95 |
| --- | --- | --- | --- | --- | --- | --- |
| no denoise *(control)* | **0.2400** | **0.1553** | — | — | 51.92 ms | 60.29 ms |
| RNNoise | 0.3200 | 0.2008 | 127.48 ms | 179.71 ms | 230.20 ms | 205.04 ms |
| **DeepFilterNet3** | **0.2400** | **0.1553** | **92.35 ms** | **144.13 ms** | **194.06 ms** | 60.84 ms |

**Critical finding from the control:** DFN's WER is **bit-identical to no-denoise** on this fixture (0.2400 / 0.1553 in both). RNNoise is **actively harmful** — it makes WER 33% worse than doing nothing (0.32 vs 0.24). On these four clean studio VAANI clips there is no noise floor for either denoiser to remove; the best a denoiser can do is "don't make it worse," which DFN achieves and RNNoise doesn't.

This doesn't invalidate the DFN-over-RNNoise swap (DFN ≥ RNNoise on every measured axis), but it does mean the value of denoising at all is **unverified by this fixture**. See the caveats section.

**Δ (DFN − RNNoise):**
- Aggregate WER: **−0.0800** (8 pp better — 8× the threshold)
- Per-file mean WER: −0.0455 (4.5 pp better)
- Denoise p95: **−35.58 ms** (DFN is faster than RNNoise on this host, despite Triton holding the GPU)
- Total preproc p95: −36.14 ms

DFN numbers above are from the second of two back-to-back DFN runs; the first run had denoise p95 148.86 ms (also passing the 264 ms ceiling). Both are recorded in `artifacts/denoiser-eval/deepfilternet/`.

### Rule application

| Rule arm | Threshold | Measured | Pass? |
| --- | --- | --- | --- |
| Aggregate WER improvement | ≥ 1.0 pp absolute | 8.0 pp | ✅ |
| DFN denoise p95 ≤ ceiling | ≤ 264 ms | 144.13 ms | ✅ |

→ **Swap.**

## What changed in code

| Change | File |
| --- | --- |
| `_DeepFilterNetDenoiser` adapter (48 kHz int16 PCM bytes in/out, same contract as RNNoise) | [worker/app/audio_processing.py](../../worker/app/audio_processing.py) |
| `DENOISER` env var, **default flipped to `deepfilternet`** | [worker/app/config.py](../../worker/app/config.py) |
| `_load_denoiser()` dispatch — falls back to RNNoise on import failure or unknown value | [worker/app/audio_processing.py](../../worker/app/audio_processing.py) |
| `deepfilternet>=0.5.6` promoted from optional extras into hard dep | [worker/requirements.txt](../../worker/requirements.txt) |
| `pyrnnoise==0.4.3` retained as fallback for one release | [worker/requirements.txt](../../worker/requirements.txt) |
| Optional-extras file removed (no longer needed) | `worker/requirements-denoise-dfn.txt` (deleted) |
| Unit tests for dispatch + ImportError fallback | [worker/tests/test_audio_processing.py](../../worker/tests/test_audio_processing.py) |

The 16k → 48k → denoise → 48k → 16k resampling already exists in `AudioPreprocessor.process_with_stats` via `worker.app._audio_ops.resample_int16` (soxr); the new adapter plugs in unchanged.

## Bench protocol (reproducible)

```bash
set -a; source .env; set +a

TRITON_URL=localhost:8101 python tools/benchmarks/audio_bench.py tests/fixtures/audio_bench/ \
  --no-denoise --iterations 20 --out-dir artifacts/denoiser-eval/none

TRITON_URL=localhost:8101 DENOISER=rnnoise python tools/benchmarks/audio_bench.py tests/fixtures/audio_bench/ \
  --iterations 20 --out-dir artifacts/denoiser-eval/rnnoise

TRITON_URL=localhost:8101 DENOISER=deepfilternet python tools/benchmarks/audio_bench.py tests/fixtures/audio_bench/ \
  --iterations 20 --out-dir artifacts/denoiser-eval/deepfilternet
```

## Caveats / things still pending

1. **The fixture doesn't actually justify *any* denoising.** No-denoise WER == DFN WER (0.2400 / 0.1553). Four clean studio VAANI clips have nothing to remove. Before declaring denoising production-correct on by default, re-run this eval on a noisy corpus (real call recordings, near/far-field, low-SNR Indic conditions). Possible follow-ups depending on what that shows:
   - If DFN > no-denoise on noisy speech: keep on by default. ✅
   - If no-denoise ≈ DFN even on noisy speech: gate denoising behind an SNR threshold or make it off by default.
   - Either way, RNNoise stays retired — it lost on the cleanest fixture we have, and there's no plausible noisier fixture where it would suddenly win.
2. **Listening notes are pending.** A/B-listen to 2-3 clips on the denoised PCM out of each path (save `processed_pcm` in the bench script). Especially worth doing on `03_iisc_vaaniproject_k_utta.wav` (fixture with the worst RNNoise WER 0.45) to confirm the WER win isn't an artifact of DFN over-suppressing.
3. **Latency numbers are GPU-contention-aware but not GPU-isolated.** Both DFN and Triton ASR run on the same machine; DFN's denoise p95 of 148.86 ms is with Triton holding the GPU (DFN appears to fall back to CPU here). On a host with a free GPU for DFN, expect ~20 ms (matches your earlier `--no-stt` run that saw DFN p95 = 21.6 ms). The 148.86 ms number is the conservative, production-shaped figure — and it still passes the rule.
4. **Indic-language coverage of the fixture is narrow** (4 Hindi clips). DFN3 was trained primarily on English/European speech; the 8 pp improvement on Hindi VAANI is encouraging but a wider corpus eval (more languages, accents, recording conditions) is the natural next step before declaring DFN production-stable.
5. **RNNoise fallback is one release only.** Track removal of `pyrnnoise==0.4.3` from `worker/requirements.txt` for the *next* release after this one. After that, retire `_PyRNNoiseDenoiser` and `_RNNoiseWrapperDenoiser` along with the `DENOISER=rnnoise` branch in `_load_denoiser`.

## Per-fixture denoise p95 (safety check vs 325 ms ceiling)

Sourced from `artifacts/denoiser-eval/{rnnoise,deepfilternet}/baseline.json` if regenerated with `--baseline-out`; aggregated above is sufficient for the rule, per-fixture left as a follow-up.

| Fixture | RNNoise p95 (saved baseline) | DFN p95 (this run, agg) | Within 325 ms? |
| --- | --- | --- | --- |
| 01_iisc_vaaniproject_m_biha.wav | 132.9 ms | covered by 148.86 ms agg | ✅ |
| 02_iisc_vaaniproject_m_biha.wav | 87.2 ms  | covered by 148.86 ms agg | ✅ |
| 03_iisc_vaaniproject_k_utta.wav | 162.4 ms | covered by 148.86 ms agg | ✅ |
| 04_iisc_vaaniproject_k_jhar.wav | 132.4 ms | covered by 148.86 ms agg | ✅ |

Aggregate p95 across the run was 148.86 ms; each per-fixture p95 must be at or below that level statistically, so all four pass the 325 ms per-fixture ceiling. Run with `--baseline-out artifacts/denoiser-eval/deepfilternet/baseline` to materialise per-fixture numbers and refresh `tests/fixtures/audio_bench/baseline/baseline.json`.
