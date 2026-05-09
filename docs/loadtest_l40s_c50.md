# L40s `/v1/predict` Load Test — 50 Concurrency, 500 Requests

## What this run is

Direct HTTP load test against the Simplismart-hosted L40s ASR endpoint
(`https://http.ng8pfht5tu.ss-in.s9t.link/v1/predict`). Replays a fixed pool of
pre-cached 8 kHz mono int16 Vaani clips with N concurrent in-flight requests via
a thread pool. This is **not** a gateway/worker WebSocket test — the gateway,
worker, ITN, and APM are bypassed. Numbers here describe the hosted model
endpoint in isolation.

Harness: [tests/loadtest_l40s.py](../tests/loadtest_l40s.py) (depends on
`API_URL`/`HEADERS` from [tests/test_client_raw_streaming.py](../tests/test_client_raw_streaming.py)).

## Run configuration

| Field | Value |
|---|---:|
| Date | 2026-05-09 |
| Endpoint | `https://http.ng8pfht5tu.ss-in.s9t.link/v1/predict` |
| GPU target | L40s |
| Audio format | 8 kHz mono int16 `.raw` |
| Clip pool | `tests/fixtures/loadtest_vaani_8k` (50 clips) |
| Clip duration | mean 2.91 s, min 1.63 s, max 6.53 s |
| Warmup | 5 sequential calls |
| Concurrency | 50 in-flight |
| Total requests | 500 |
| Seed | 42 |

## Headline results

| Metric | Value |
|---|---:|
| Successes / failures | 500 / 0 |
| Wall time | 21.68 s |
| Throughput | 23.07 req/s |
| Audio sent | 1,468.09 s |
| Aggregate RTF | 67.73× real time |
| Latency mean | 2.101 s |
| Latency p50 | 2.087 s |
| Latency p95 | 3.204 s |
| Latency p99 | 3.974 s |
| Latency max | 4.000 s |

## Interpretation

- **Aggregate RTF 67.73×** = 1,468 s of audio processed in 21.68 s of wall
  clock at concurrency 50. Sanity check: `500 × 2.94 s ≈ 1,468 s` of audio,
  consistent with the clip-pool mean.
- **Per-request RTF below 1×.** Mean latency 2.10 s on mean clip duration
  2.91 s ⇒ the endpoint returns the final transcript in ~0.72× the audio
  duration on average. The p99 of 3.97 s sits below the longest clip in the
  pool (6.53 s), so even tail requests beat real time at this concurrency.
- **Tight tail.** p99/p50 = 1.90×, max/p50 = 1.92×. No long-tail blowup, no
  retries, no failures. Suggests the endpoint is not queue-bound at C=50 for
  this clip-length distribution.
- **Not directly comparable to Run B / Run C.** Those runs measured the local
  gateway+worker stack over WebSockets with 10 s streaming sessions and
  reported `server_processing` and `final_before_flush`. This run measures a
  remote HTTP endpoint with one-shot uploads of 1.6–6.5 s clips. Both report
  "p50/p95/p99 latency" but they describe different code paths and different
  workload shapes.

## What this run does not validate

- Streaming partials, VAD, endpointing, ITN, or APM — all bypassed.
- Gateway concurrency limits (`GATEWAY_MAX_INFLIGHT_WORKER`, `WORKER_MAX_JOBS`)
  — not on the path.
- Sustained load. 21.68 s is short; cold-start, autoscaling, and slow-burn
  memory effects are not exercised.
- Higher concurrency. The endpoint was clearly not saturated at C=50;
  ceiling is unknown until C=100 / 200 / 500 are tried.

## Reproducer

```bash
# one-time clip prep
python -m tests.fetch_vaani_loadtest_clips --n 50

# run
.venv/bin/python -m tests.loadtest_l40s --concurrency 50 --total 500
```
