# WebSocket ASR Load Test - Ingest and Inference Capacity

## Executive summary

This run validates **50 concurrent WebSocket clients** and about **40 concurrent
real-time audio ingest streams**. It does **not** mean 40 or 50 simultaneous
model inference jobs.

For model transcription concurrency, the current deployment is bounded by worker
settings:

| Layer | Current limit | Meaning |
|---|---:|---|
| Gateway worker calls | 4 | Gateway will allow up to 4 in-flight calls to the worker. |
| Worker model jobs | 2 per worker | A single worker process will run up to 2 transcription jobs at once. |

So the production-safe statement is:

> The system can ingest about 40 concurrent real-time audio streams while final
> ASR inference is enabled. With the current single-worker configuration, actual
> parallel model transcription execution is capped at 2 jobs at a time, with
> additional requests queued by the gateway/worker.

Recommended wording:

> The current WebSocket ASR gateway sustained 50 concurrent clients over a
> 5-minute test with zero failures, completing 1,233 ten-second streaming
> sessions. This is equivalent to about 41 active real-time audio ingest streams
> on average. Final ASR inference was enabled, but model execution concurrency is
> separately capped by `WORKER_MAX_JOBS`.

## Source artifacts

| Artifact | Purpose |
|---|---|
| `loadtest/results_runC.html` | Primary report matching the pasted Locust table |
| `loadtest/results_runC_stats.csv` | CSV stats snapshot from the same run family |
| `loadtest/results_runC_failures.csv` | Failure export; no failures recorded |
| `loadtest/results_runC_exceptions.csv` | Exception export; no exceptions recorded |
| `loadtest/ws_locustfile.py` | Test definition and metric instrumentation |

Note: `results_runC.html` is the source for the exact counts below because it
matches the pasted table with 5,015 total request events. The CSV snapshot is
near-identical but shows 4,988 events.

## Test setup

| Field | Value |
|---|---:|
| Test date | 2026-05-05 |
| Host | `ws://localhost:8000` |
| Locustfile | `ws_locustfile.py` |
| Start time | `2026-05-05T08:22:37Z` |
| End time | `2026-05-05T08:27:37Z` |
| Duration | 5 minutes |
| Peak Locust users | 50 |
| User class | `WSStreamUser` |
| Task | `stream_session` |
| Audio per session | 10 seconds |
| Total request events | 5,015 |
| Total failures | 0 |
| Total exceptions | 0 |
| Aggregate event rate | 16.73 events/s |

## Claimable concurrency

The key distinction is between **connected sockets**, **active audio ingest**,
and **parallel model transcription jobs**.

| Claim type | Value | Status |
|---|---:|---|
| Concurrent WebSocket clients | 50 | Validated |
| Concurrent active audio ingest streams | 41.1 | Derived |
| Conservative claimable audio ingest streams | 40 | Recommended |
| Completed 10-second stream sessions | 1,233 | Validated |
| Completed stream sessions per second | 4.114/s | Validated |
| Final transcription throughput | ~4.2 finals/s | Validated |
| Gateway in-flight worker calls | 4 | Configured limit |
| Actual parallel model transcription jobs | 2 per worker | Configured limit |

Calculation:

```text
active_real_time_streams = completed_streams_per_second * stream_duration_seconds
active_real_time_streams = 4.114 * 10
active_real_time_streams = 41.14
```

The ingest headline should be rounded down to **40 concurrent live audio
streams** to avoid overstating the result.

## Model inference concurrency

WebSocket concurrency is not transcription concurrency. A machine can hold many
open sockets, but the model can only transcribe as many utterances in parallel as
the worker runtime allows.

In the current configuration:

| Setting | Value | Effect |
|---|---:|---|
| `GATEWAY_MAX_INFLIGHT_WORKER` | 4 | Gateway allows up to 4 concurrent calls into the worker service. |
| `WORKER_MAX_JOBS` | 2 | Worker allows up to 2 concurrent ASR jobs per worker instance. |
| Gateway uvicorn workers | 1 | One gateway process is serving WebSocket traffic in this compose setup. |

Therefore, with one worker instance, the answer to "how many audio files or
utterances can the model transcribe at the exact same time?" is:

```text
parallel_model_transcriptions = worker_replicas * WORKER_MAX_JOBS
parallel_model_transcriptions = 1 * 2
parallel_model_transcriptions = 2
```

The run proves the system can keep about 40 live audio uploads moving while
those final transcription jobs are processed and queued. It does not prove that
40 model forward passes are running simultaneously.

To increase true transcription parallelism, scale one or both of:

| Lever | Effect |
|---|---|
| More worker replicas | Increases total parallel model jobs if each replica has GPU/CPU capacity. |
| Higher `WORKER_MAX_JOBS` | Allows more simultaneous jobs per worker, but only if the model and GPU memory can sustain it. |
| Higher `GATEWAY_MAX_INFLIGHT_WORKER` | Allows the gateway to send more concurrent worker calls, but should not exceed backend capacity by much. |

The formula to plan production capacity is:

```text
true_parallel_model_jobs = worker_replicas * WORKER_MAX_JOBS
sustainable_final_transcripts_per_second ~= true_parallel_model_jobs / p99_model_latency_seconds
```

Using the measured p99 `server_processing` value of 2.3 seconds and the current
2-job worker limit:

```text
conservative_p99_throughput ~= 2 / 2.3 = 0.87 final transcriptions/s
```

The observed run completed about 4.2 finals/s because most requests were much
faster than p99. For production sizing, use both:

| Planning view | Value | Use |
|---|---:|---|
| Observed average throughput | ~4.2 finals/s | What this run actually sustained. |
| Conservative p99 throughput estimate | ~0.87 finals/s | Safer planning if every job behaves like p99. |
| Exact simultaneous model jobs | 2 per worker | Hard configured concurrency limit. |

## Full metric summary

| Metric | Requests | Failures | Avg ms | Min ms | Median ms | Max ms | Avg size | RPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `connect` | 1,264 | 0 | 29.35 | 2 | 10 | 642 | 0 B | 4.22 |
| `final_before_flush` | 1,259 | 0 | 686.39 | 431 | 510 | 3,331 | 252 B | 4.20 |
| `server_processing` | 1,259 | 0 | 327.74 | 89 | 160 | 2,981 | 252 B | 4.20 |
| `stream` | 1,233 | 0 | 10,001.04 | 10,000 | 10,000 | 10,042 | 320,000 B | 4.11 |
| `Aggregated` | 5,015 | 0 | 2,720.87 | 2 | 450 | 10,042 | 78,802 B | 16.73 |

## Latency percentiles

All values are milliseconds.

| Metric | p50 | p66 | p75 | p80 | p90 | p95 | p98 | p99 | p99.9 | p99.99 | p100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `connect` | 10 | 16 | 25 | 36 | 84 | 140 | 170 | 190 | 560 | 640 | 640 |
| `final_before_flush` | 510 | 570 | 640 | 720 | 1,200 | 1,700 | 2,400 | 2,700 | 3,300 | 3,300 | 3,300 |
| `server_processing` | 160 | 210 | 280 | 360 | 830 | 1,400 | 1,900 | 2,300 | 3,000 | 3,000 | 3,000 |
| `stream` | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 |
| `Aggregated` | 450 | 640 | 2,700 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 |

## How to read the `stream` metric

The `stream` row is intentionally around 10,000 ms because the load test sends
10 seconds of audio in real time. It is not backend inference latency.

In `loadtest/ws_locustfile.py`, the test:

1. Sets `STREAM_SECONDS` to 10 by default.
2. Sends 20 ms PCM frames.
3. Sleeps between frames to preserve real-time pacing.
4. Records the `stream` metric after all 10 seconds of audio have been sent.

Because of that, aggregate p80/p90/p95/p99 values are dominated by the `stream`
metric and should not be used as backend latency numbers.

## Latency numbers to report

Use these latency figures under the validated 50-client / 40-ingest-stream load:

| Latency metric | p50 | p95 | p99 | Meaning |
|---|---:|---:|---:|---|
| `connect` | 10 ms | 140 ms | 190 ms | WebSocket connection setup |
| `server_processing` | 160 ms | 1.4 s | 2.3 s | Worker/gateway final processing latency |
| `final_before_flush` | 510 ms | 1.7 s | 2.7 s | Client-observed final emission before flush |

## Approved external wording

Use this when summarizing the result:

> We validated 50 concurrent WebSocket clients and about 40 concurrent real-time
> audio ingest streams with zero failures over a 5-minute run. Final ASR
> inference was enabled. The current single-worker configuration runs 2 model
> transcription jobs at a time, while the gateway can hold up to 4 in-flight
> worker calls. Under this load, server processing latency was 160 ms p50,
> 1.4 s p95, and 2.3 s p99.

## What not to claim yet

Do not claim these without another test:

| Claim | Reason |
|---|---|
| 50 active speech streams | The run had 50 WS clients, but only about 41 active real-time audio streams on average. |
| 40 or 50 parallel model transcriptions | The current worker setting is `WORKER_MAX_JOBS=2`, so one worker runs 2 ASR jobs at a time. |
| Production max capacity | This was a 5-minute validation run, not a saturation test. |
| Real-call speech accuracy under load | The load test used the configured audio source; if no `AUDIO_FILE` was set, it used synthetic low-amplitude noise. |
| Backend p95 is 10 seconds | The 10-second value is the intentional audio streaming duration, not backend processing latency. |

## Next validation step

For a stronger production claim, run a longer test with representative call audio
and a stepped concurrency ramp, for example:

| Target | Purpose |
|---|---|
| 60 minutes at 40 active streams | Soak validation for the current claim |
| 60, 80, 100 active ingest streams | Find the real ingest ceiling |
| 2, 4, 8 worker model jobs | Find the real model-inference ceiling and GPU memory limit |
| Representative Hindi/Indic call audio | Validate latency and accuracy on production-like input |
| Separate dashboard excluding `stream` from aggregate latency | Prevent intentional 10-second streaming duration from masking backend latency |
