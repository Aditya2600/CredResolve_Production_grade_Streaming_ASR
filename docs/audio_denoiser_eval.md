# Audio Denoiser Evaluation

**Decision:** use DeepFilterNet3 as the default denoiser; keep RNNoise only as a temporary fallback.

**Date:** 2026-05-08  
**Fixture:** `tests/fixtures/audio_bench/` — 4 Hindi VAANI clips, 20 iterations per clip.  
**Backend:** Triton on `localhost:8101`, worker env with `pyrnnoise==0.4.3` and `deepfilternet>=0.5.6`.

| Config | Aggregate WER | Per-file mean WER | Denoise p95 | Total preproc p95 |
| --- | --- | --- | --- | --- |
| no denoise | **0.2400** | **0.1553** | n/a | 51.92 ms |
| RNNoise | 0.3200 | 0.2008 | 179.71 ms | 230.20 ms |
| DeepFilterNet3 | **0.2400** | **0.1553** | **144.13 ms** | **194.06 ms** |

DeepFilterNet3 beats RNNoise on WER and latency. It ties no-denoise on this clean fixture, so the value of denoising still needs a noisy call-corpus check.

## Caveats

- The fixture is narrow and clean; rerun on noisy, low-SNR call recordings before treating denoise-on as proven.
- RNNoise lost on the clean fixture and should stay fallback-only.
- Remove `pyrnnoise==0.4.3`, `_PyRNNoiseDenoiser`, `_RNNoiseWrapperDenoiser`, and `DENOISER=rnnoise` after one stable DeepFilterNet3 release.
