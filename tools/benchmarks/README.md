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
