# Audio bench baseline

`baseline.csv` (raw per-iteration rows) and `baseline.json` (per-fixture
aggregates) are produced by
[`tests/fixtures/audio_bench/regenerate_baseline.py`](../regenerate_baseline.py),
which wraps [`tools/benchmarks/audio_bench.py`](../../../../tools/benchmarks/audio_bench.py).

This baseline is a reference snapshot for subsequent PRs that touch
`worker/app/audio_processing.py`. It is **not** run in CI (the bench is too
slow for that). CI only loads `baseline.json` and asserts schema shape — see
`tests/test_audio_bench_baseline_schema.py`.

## Capture metadata

> **NOTE:** Fill these in when committing the first real run, then re-fill
> on every regeneration. They are not auto-detected.

- **Date captured (UTC):** 2026-05-08
- **Machine / GPU:** NVIDIA A40
- **Branch / commit:** `gpu_migration` / `83098b6`
- **STT backend:** `triton` (env: `ASR_BACKEND`)
- **STT model name:** `ai4bharat/indic-conformer-600m-multilingual`
- **STT decoder:** `rnnt`
- **Language:** `hi`
- **Iterations per fixture:** 20
- **Wall-clock cost of one regeneration run:** ~1m10s after the stack was healthy

## Fixtures

The four input WAVs are real Hindi speech sampled deterministically from
the Vaani corpus (`ARTPARK-IISc/Vaani-transcription-part`). See the parent
[`README.md`](../README.md) and [`ATTRIBUTION.md`](../ATTRIBUTION.md) for
provenance, license, and the selection seed.

References are post-`normalize_asr_text` (Vaani markup like
`<noise>...</noise>`, `[unintelligible]`, `{romanization}` glosses already
stripped), so `wer_mean` is computed against clean Devanagari text and is
directly comparable across regenerations.

## Schema (`baseline.json`)

```jsonc
{
  "schema_version": 1,
  "stt_enabled": true,
  "iterations_per_fixture": 20,
  "fixtures": {
    "<fixture-name.wav>": {
      "n_iterations": 20,
      "duration_s": 2.0,
      "wer_mean": 0.0 | null,         // mean per-iteration WER vs reference
      "wer_n": 0,                       // # iterations that contributed to WER
      "vad_p50_ms": 0.0 | null,        // VAD stage latency, p50 across iterations
      "vad_p95_ms": 0.0 | null,        // VAD stage latency, p95
      "denoise_p50_ms": 0.0 | null,
      "denoise_p95_ms": 0.0 | null,
      "total_p50_ms": 0.0,             // full process_with_stats() wall time, p50
      "total_p95_ms": 0.0,             // full process_with_stats() wall time, p95
      "stt_p50_ms": 0.0 | null,        // model.transcribe_pcm16 wall time
      "stt_p95_ms": 0.0 | null,
      "output_duration_ratio": 0.5 | null,  // (mean VAD output samples / 16k) / duration_s
      "n_segments_mean": 1.0 | null    // mean # VAD segments per iteration
    }
  }
}
```

## Metric definitions

| Field                    | Meaning                                                                                  |
|--------------------------|------------------------------------------------------------------------------------------|
| `n_iterations`           | How many bench passes were run for this fixture (matches `iterations_per_fixture`).      |
| `duration_s`             | Input WAV duration in seconds, after mono + 16 kHz resampling.                            |
| `wer_mean`               | Mean of per-iteration WER (Levenshtein on whitespace tokens, lowercase, punct stripped). |
| `wer_n`                  | Iterations that produced a WER datum (i.e. STT enabled AND non-empty reference text).    |
| `vad_p{50,95}_ms`        | Wall-clock cost of the VAD stage inside `process_with_stats`.                            |
| `denoise_p{50,95}_ms`    | Wall-clock cost of the denoise stage.                                                    |
| `total_p{50,95}_ms`      | End-to-end `process_with_stats` cost (preprocessing only, not STT).                      |
| `stt_p{50,95}_ms`        | `model.transcribe_pcm16` wall time on the post-preprocessing PCM.                        |
| `output_duration_ratio`  | Fraction of input duration that survives VAD trimming (1.0 = nothing dropped).           |
| `n_segments_mean`        | Mean number of VAD-detected speech segments.                                             |

## Regression rules for subsequent PRs

When a PR modifies `worker/app/audio_processing.py` (or anything reachable
from `AudioPreprocessor.process_with_stats`):

1. **WER must not regress.** For every fixture where `wer_n > 0`, the new
   `wer_mean` must be `<=` baseline `wer_mean + 0.02` (absolute). The
   per-clip wobble allowance reflects that we have only 4 fixtures —
   single-token edits flip mean WER by 1/N quickly — not that regressions
   are tolerated. A regression spread across multiple fixtures, or
   exceeding 0.02 on any one, is a real regression.
2. **`total_p95_ms` may regress at most 10%.** I.e. new value
   `<= 1.10 * baseline.total_p95_ms` per fixture. Larger regressions need
   either a) explicit justification in the PR description (e.g. correctness
   fix) and an updated baseline, or b) revert.
3. **`vad_p95_ms` and `denoise_p95_ms` individually may regress at most
   25%.** Stage-level budgets are looser than the total because per-stage
   timing is noisier; the binding constraint is `total_p95_ms`.
4. **`output_duration_ratio` and `n_segments_mean` may shift by at most
   ±10%** — larger shifts mean VAD behavior changed, which warrants an
   explicit note + updated baseline.
5. **Schema change?** Bump `schema_version` and update both this README and
   `tests/test_audio_bench_baseline_schema.py`.

If a PR deliberately changes algorithm behavior (denoise threshold, VAD
mode, etc.), regenerate this baseline as part of that PR rather than fudging
the comparison. Document the *why* in the PR description so future readers
know which baseline transitions were intentional.
