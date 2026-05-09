# audio_bench fixtures

Inputs for [`tools/benchmarks/audio_bench.py`](../../../tools/benchmarks/audio_bench.py).
Four real Hindi speech clips drawn from the **Vaani** corpus by ARTPARK @
IISc Bangalore (`ARTPARK-IISc/Vaani-transcription-part`, `audio/Hindi`,
`train` split), redistributed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). See
[`ATTRIBUTION.md`](ATTRIBUTION.md) for the source, snapshot date, and the
exact selection knobs (seed, scan limit, duration filter).

The clips are decoded to 16 kHz mono PCM16 WAV. No other processing is
applied to the audio. Filenames carry an `0X_` prefix plus a Vaani-derived
identifier slug — they are not curated for "case coverage" (clean vs.
noisy vs. multi-segment); they are a deterministic random sample of real
field recordings.

## `reference_transcripts.json`

Vaani transcripts are passed through [`tools.asr_text_normalizer.normalize_asr_text`](../../../tools/asr_text_normalizer.py)
before being written here. That strips Vaani-specific annotation
(`<noise>...</noise>`, `[unintelligible]`, `{romanization}` glosses,
trailing `--` cut-off markers, etc.) and applies NFKC + nukta + digit
normalization, so the references are directly comparable against ASR
output.

## Regenerating the fixtures

Re-run **only** when you want a fresh Vaani snapshot:

```
.venv-tests/bin/python tests/fixtures/audio_bench/_fetch_vaani_fixtures.py
```

Requires `HUGGINGFACE_HUB_TOKEN` (gated dataset) and the `datasets` +
`pyarrow` packages in the venv. The selection is deterministic given the
same Vaani snapshot: same seed → same clips. Re-running rewrites the WAVs,
[`reference_transcripts.json`](reference_transcripts.json), and
[`ATTRIBUTION.md`](ATTRIBUTION.md). After regenerating fixtures you should
also regenerate the baseline (see below) — the WER and latency numbers are
clip-dependent.

## Baseline (`baseline/`)

`baseline/baseline.csv` (raw per-iteration rows) and `baseline/baseline.json`
(per-fixture aggregates) are produced by
[`regenerate_baseline.py`](regenerate_baseline.py), which wraps
`tools/benchmarks/audio_bench.py` with `--iterations 20 --baseline-out
baseline/`. See [`baseline/README.md`](baseline/README.md) for metric
definitions, capture metadata, and the regression rules subsequent PRs
apply.

A pytest at [`tests/test_audio_bench_baseline_schema.py`](../../test_audio_bench_baseline_schema.py)
validates the JSON shape only; the bench itself is not run in CI (too slow).
