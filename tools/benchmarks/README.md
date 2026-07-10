# Benchmark Utilities

Performance and latency harnesses live here instead of `scripts/` so the
top-level script folder stays focused on setup, staging, and repeatable run
commands.

- `analyze_timing_logs.py`: parse Triton timing logs and emit duration-bucketed
  latency summaries.
- `audio_bench.py`: run fixture WAVs through audio preprocessing and optional
  worker STT.
- `bench_worker_sequential.py`: sequentially POST utterances to the worker
  transcription endpoint and capture per-request timings.
- `build_hindi_eval_manifest.py`: materialize a public Hindi set
  (`sarvamai/contextual_asr_benchmark` or FLEURS `hi_in`) into a 16 kHz NeMo
  JSONL manifest for offline ASR eval.
- `run_nemotron_streaming.py`: run Nemotron 3.5 ASR cache-aware streaming
  inference at one or more `att_context_size` latency points, timing each run.
- `score_nemotron_manifest.py`: score Nemotron + IndicConformer hypotheses
  against the same manifest through the shared normalizer + WER (apples-to-apples).

## Nemotron 3.5 ASR vs IndicConformer (Hindi)

End-to-end comparison harness. Nemotron needs an **isolated NeMo 26.06 env**
(it conflicts with the worker's pinned NeMo 2.4.1) — full procedure, commands,
and the results table are in
[docs/nemotron_vs_indicconformer_hindi_benchmark.md](../../docs/nemotron_vs_indicconformer_hindi_benchmark.md).
Pipeline: `build_hindi_eval_manifest.py` → `validate_nemo_manifest_audio.py` →
`run_nemotron_streaming.py` (isolated env) + `eval_nemo_manifest_wer.py` (prod env)
→ `score_nemotron_manifest.py`.

## Docker Notes

`docker compose exec worker ...` requires `worker` to already be running with
the same compose files. For the audio fixture benchmark stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.triton.yml \
  -f tests/fixtures/audio_bench/docker-compose.bench.yml \
  up -d triton worker
```

For quick dependency probes that do not need Triton or the worker HTTP server,
use a one-off container instead:

```bash
docker compose -f docker-compose.yml -f docker-compose.triton.yml \
  -f tests/fixtures/audio_bench/docker-compose.bench.yml \
  run --rm --no-deps worker bash -lc \
  'PYTHONPATH=/repo python -c "import importlib.util; print(importlib.util.find_spec(\"pyrnnoise\"))"'
```
