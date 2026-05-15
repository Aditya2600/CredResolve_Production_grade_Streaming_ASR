# WebSocket Barrier Burst Test - 30 Concurrent Vaani Streams

## Summary

This run validated a barrier-synchronized burst of **30 concurrent WebSocket
clients**, each streaming the same 10-second Vaani Hindi audio clip in real
time to the gateway `/ws/stt` endpoint.

The first run with the old worker timeout produced `worker-fallback` on the
first utterance for 17 clients. The timeout was then tuned and the same 30-client
burst was rerun successfully with **no `worker-fallback` transcripts**.

The run completed successfully at the WebSocket/session level:

| Metric | Value |
|---|---:|
| Clients | 30 |
| Connected | 30 |
| Successful sessions | 30 |
| Failures | 0 |
| Audio per client | 10 seconds |
| Payload per client | 320,000 bytes |
| Total audio payload | 9,600,000 bytes |
| Wall time | 32.895 s |

The final verification run produced model transcripts for all three utterances
for all 30 clients.

## Test Setup

| Field | Value |
|---|---|
| Test date | 2026-05-05 |
| Test style | asyncio barrier burst |
| Script | `tools/ws_burst_barrier.py` |
| Endpoint | `ws://localhost:8000/ws/stt` |
| Language | `hi` |
| Model | `credresolve:v1` |
| Mode | `transcribe` |
| Audio codec | `pcm_s16le` |
| Audio transport | binary WebSocket frames |
| Send mode | real-time paced 20 ms frames |
| Gateway worker in-flight limit | `GATEWAY_MAX_INFLIGHT_WORKER=30` |
| Worker model job limit | `WORKER_MAX_JOBS=30` |
| Worker ASR inference timeout | `ASR_INFERENCE_TIMEOUT_MS=30000` |
| Gateway worker timeout | `WORKER_TIMEOUT_MS=35000` |

Timeout fix:

| Setting | Before | After | Reason |
|---|---:|---:|---|
| `ASR_INFERENCE_TIMEOUT_MS` | 4,000 ms | 30,000 ms | The old 4s worker inference timeout caused `fallback_timeout` for some first utterances under 30-way concurrency. |
| `WORKER_TIMEOUT_MS` | 15,000 ms | 35,000 ms | Gateway HTTP timeout is kept above the worker inference timeout so the worker can return a real result. |

## Audio Input

A random eligible Vaani Hindi WAV was selected from the local Vaani cache:

```text
artifacts/vaani_hindi_all/audio/Hindi_train_13269.wav
```

Audio metadata:

| Field | Value |
|---|---:|
| Channels | 1 |
| Sample width | 2 bytes |
| Sample rate | 16,000 Hz |
| Source duration | 11.815 s |
| Streamed duration | first 10.000 s |
| SHA-256 | `5f652fff6811b88546a323e0d8a8d380e3c91e03a08ccd9c48636787d0745b63` |

The script clipped the file to the first 10 seconds and sent 320,000 raw PCM
bytes per client.

## Command

```bash
python3 tools/ws_burst_barrier.py \
  --ws ws://localhost:8000/ws/stt \
  --n 30 \
  --wav artifacts/vaani_hindi_all/audio/Hindi_train_13269.wav \
  --max-audio-sec 10 \
  --send-mode realtime \
  --language-code hi \
  --response-timeout 120 \
  --idle-timeout 20 \
  --expect-data
```

## Client-Side Results

```text
Clients: 30
Connected: 30
Successful: 30
Failures: 0
First-audio release spread: 5.37ms
Flush send spread: 3.73ms
Bytes sent total: 9600000
Data messages: 90
Pre-flush data messages: 30
Post-flush data messages: 60
VAD messages: 0
Connect: avg=40.3ms p50=40.0ms p95=44.4ms min=17.9ms max=78.9ms
Stream send: avg=10000.6ms p50=10000.6ms p95=10001.1ms min=10000.1ms max=10001.1ms
Flush barrier wait: avg=4.1ms p50=4.1ms p95=4.9ms min=0.0ms max=5.8ms
Response wait: avg=22717.5ms p50=22770.4ms p95=22798.7ms min=22482.4ms max=22799.6ms
Total session: avg=32769.4ms p50=32823.1ms p95=32849.2ms min=32530.5ms max=32854.4ms
Wall time: 32894.7ms
```

Interpretation:

- The start barrier worked: all clients began sending audio within about 5 ms.
- The flush barrier worked: all clients sent flush within about 4 ms in the
  final no-fallback verification run.
- The 10-second `stream` timing is intentional real-time pacing, not backend
  inference latency.
- The `response_wait` and `total_session` values include the configured
  20-second client idle grace used to avoid closing early after removing the
  worker timeout fallback.

## Transcript Output

Gateway logs showed 90 `data` messages, which means each of the 30 clients
emitted 3 utterance-level data messages.

Grouped transcript output:

| Count | Utterance | Transcript | Language source |
|---:|---|---|---|
| 30 | `utt-0001` | `बहुत ही मस्त नज़ारा लगता है बहुत ही मस्त देखो तमस् सजा कितनी भीड़` | `client` |
| 30 | `utt-0002` | `खेत मस्त डांस करना इंडिया मस्त` | `client` |
| 30 | `utt-0003` | `सारे देव किती भीड़ भीड़` | `client` |

The combined transcript was:

```text
बहुत ही मस्त नज़ारा लगता है बहुत ही मस्त देखो तमस् सजा कितनी भीड़ खेत मस्त डांस करना इंडिया मस्त सारे देव किती भीड़ भीड़
```

## Distinct-Audio Follow-Up Run

A second 30-client burst was run with one different eligible Vaani Hindi WAV per
client. The script sampled the files from `artifacts/vaani_hindi_all/audio`
using fixed seed `20260505`, clipped each file to the first 10 seconds, and sent
320,000 raw PCM bytes per client.

Command:

```bash
python3 tools/ws_burst_barrier.py \
  --ws ws://localhost:8000/ws/stt \
  --n 30 \
  --wav-dir artifacts/vaani_hindi_all/audio \
  --audio-seed 20260505 \
  --max-audio-sec 10 \
  --send-mode realtime \
  --language-code hi \
  --response-timeout 120 \
  --idle-timeout 20 \
  --expect-data
```

Selected audio:

```text
client 00: artifacts/vaani_hindi_all/audio/Hindi_train_18469.wav (16.277s source)
client 01: artifacts/vaani_hindi_all/audio/Hindi_train_4403.wav (10.685s source)
client 02: artifacts/vaani_hindi_all/audio/Hindi_train_5440.wav (11.683s source)
client 03: artifacts/vaani_hindi_all/audio/Hindi_train_5087.wav (10.127s source)
client 04: artifacts/vaani_hindi_all/audio/Hindi_train_6830.wav (10.747s source)
client 05: artifacts/vaani_hindi_all/audio/Hindi_train_17199.wav (13.226s source)
client 06: artifacts/vaani_hindi_all/audio/Hindi_train_4186.wav (11.828s source)
client 07: artifacts/vaani_hindi_all/audio/Hindi_train_13817.wav (12.369s source)
client 08: artifacts/vaani_hindi_all/audio/Hindi_train_1639.wav (13.793s source)
client 09: artifacts/vaani_hindi_all/audio/Hindi_train_17375.wav (12.207s source)
client 10: artifacts/vaani_hindi_all/audio/Hindi_train_1316.wav (10.074s source)
client 11: artifacts/vaani_hindi_all/audio/Hindi_train_3954.wav (11.010s source)
client 12: artifacts/vaani_hindi_all/audio/Hindi_train_11326.wav (16.475s source)
client 13: artifacts/vaani_hindi_all/audio/Hindi_train_2108.wav (13.952s source)
client 14: artifacts/vaani_hindi_all/audio/Hindi_train_3826.wav (11.726s source)
client 15: artifacts/vaani_hindi_all/audio/Hindi_train_10784.wav (11.891s source)
client 16: artifacts/vaani_hindi_all/audio/Hindi_train_3483.wav (13.290s source)
client 17: artifacts/vaani_hindi_all/audio/Hindi_train_6241.wav (12.726s source)
client 18: artifacts/vaani_hindi_all/audio/Hindi_train_4558.wav (14.063s source)
client 19: artifacts/vaani_hindi_all/audio/Hindi_train_887.wav (10.218s source)
client 20: artifacts/vaani_hindi_all/audio/Hindi_train_10818.wav (10.398s source)
client 21: artifacts/vaani_hindi_all/audio/Hindi_train_10638.wav (11.120s source)
client 22: artifacts/vaani_hindi_all/audio/Hindi_train_4135.wav (10.825s source)
client 23: artifacts/vaani_hindi_all/audio/Hindi_train_7111.wav (12.302s source)
client 24: artifacts/vaani_hindi_all/audio/Hindi_train_1402.wav (11.093s source)
client 25: artifacts/vaani_hindi_all/audio/Hindi_train_1684.wav (12.715s source)
client 26: artifacts/vaani_hindi_all/audio/Hindi_train_3992.wav (10.180s source)
client 27: artifacts/vaani_hindi_all/audio/Hindi_train_13703.wav (10.539s source)
client 28: artifacts/vaani_hindi_all/audio/Hindi_train_1337.wav (10.283s source)
client 29: artifacts/vaani_hindi_all/audio/Hindi_train_5153.wav (12.404s source)
```

Client-side results:

```text
Clients: 30
Connected: 30
Successful: 30
Failures: 0
First-audio release spread: 4.07ms
Flush send spread: 3.16ms
Bytes sent total: 9600000
Data messages: 69
Pre-flush data messages: 39
Post-flush data messages: 30
VAD messages: 0
Connect: avg=35.6ms p50=35.7ms p95=40.5ms min=28.5ms max=41.2ms
Stream send: avg=10000.5ms p50=10000.5ms p95=10001.0ms min=10000.0ms max=10001.1ms
Flush barrier wait: avg=3.4ms p50=3.5ms p95=4.1ms min=0.0ms max=4.2ms
Response wait: avg=22673.5ms p50=22524.0ms p95=23819.0ms min=20041.9ms max=23850.6ms
Total session: avg=32719.3ms p50=32570.7ms p95=33865.6ms min=30087.2ms max=33894.8ms
Wall time: 33915.6ms
```

Immediate gateway log checks after the run found:

| Marker | Count |
|---|---:|
| `fallback_timeout` | 0 |
| `worker-fallback` | 0 |

The lower `data` message count compared with the same-audio run is expected:
different audio files produced different utterance segmentation.

## 8-Concurrency Distinct-Audio Run

The same distinct-audio setup was also run at 8 concurrent clients.

Command:

```bash
python3 tools/ws_burst_barrier.py \
  --ws ws://localhost:8000/ws/stt \
  --n 8 \
  --wav-dir artifacts/vaani_hindi_all/audio \
  --audio-seed 20260505 \
  --max-audio-sec 10 \
  --send-mode realtime \
  --language-code hi \
  --response-timeout 120 \
  --idle-timeout 20 \
  --expect-data
```

Selected audio:

```text
client 00: artifacts/vaani_hindi_all/audio/Hindi_train_18469.wav (16.277s source)
client 01: artifacts/vaani_hindi_all/audio/Hindi_train_4403.wav (10.685s source)
client 02: artifacts/vaani_hindi_all/audio/Hindi_train_5440.wav (11.683s source)
client 03: artifacts/vaani_hindi_all/audio/Hindi_train_5087.wav (10.127s source)
client 04: artifacts/vaani_hindi_all/audio/Hindi_train_6830.wav (10.747s source)
client 05: artifacts/vaani_hindi_all/audio/Hindi_train_17199.wav (13.226s source)
client 06: artifacts/vaani_hindi_all/audio/Hindi_train_4186.wav (11.828s source)
client 07: artifacts/vaani_hindi_all/audio/Hindi_train_13817.wav (12.369s source)
```

Client-side results:

```text
Clients: 8
Connected: 8
Successful: 8
Failures: 0
First-audio release spread: 1.29ms
Flush send spread: 0.69ms
Bytes sent total: 2560000
Data messages: 17
Pre-flush data messages: 9
Post-flush data messages: 8
VAD messages: 0
Connect: avg=16.4ms p50=15.3ms p95=26.7ms min=7.2ms max=32.2ms
Stream send: avg=10000.8ms p50=10000.8ms p95=10001.4ms min=10000.3ms max=10001.4ms
Flush barrier wait: avg=0.4ms p50=0.5ms p95=0.7ms min=0.0ms max=0.7ms
Response wait: avg=20915.3ms p50=20960.2ms p95=21245.6ms min=20526.3ms max=21256.7ms
Total session: avg=30935.2ms p50=30978.7ms p95=31261.9ms min=30543.4ms max=31272.2ms
Wall time: 31291.8ms
```

Immediate gateway log checks after the run found:

| Marker | Count |
|---|---:|
| `fallback_timeout` | 0 |
| `worker-fallback` | 0 |

## 50-Client Distinct-Audio Run With Observability Capture

On 2026-05-15, a fresh deterministic 50-clip Vaani Hindi sample was fetched
outside the SimpliSmart path and used for a 50-client barrier burst:

```text
artifacts/vaani_hindi_10s_c50_seed20260515/
```

All 50 WAVs were unique, mono PCM16, 16 kHz, and exactly 10 seconds long.
The run used the current compose-published gateway port and stored the client
output plus exported Prometheus window here:

```text
artifacts/ws50_20260515T105208Z/
```

Command:

```bash
PYTHONPATH=/tmp/vaani_deps310 python3 tools/ws_burst_barrier.py \
  --ws ws://localhost:8001/ws/stt \
  --n 50 \
  --wav-dir artifacts/vaani_hindi_10s_c50_seed20260515 \
  --audio-seed 20260515 \
  --max-audio-sec 10 \
  --send-mode realtime \
  --language-code hi \
  --response-timeout 120 \
  --idle-timeout 20 \
  --expect-data
```

Client-side results:

```text
Clients: 50
Connected: 50
Successful: 50
Failures: 0
First-audio release spread: 2.79ms
Flush send spread: 1.16ms
Bytes sent total: 16000000
Data messages: 140
Pre-flush data messages: 89
Post-flush data messages: 51
VAD messages: 0
Connect: avg=22.3ms p50=22.3ms p95=24.8ms min=18.8ms max=31.8ms
Stream send: avg=10000.8ms p50=10000.9ms p95=10001.5ms min=10000.0ms max=10001.6ms
Flush barrier wait: avg=2.0ms p50=1.9ms p95=3.0ms min=0.1ms max=3.0ms
Response wait: avg=22375.8ms p50=22298.9ms p95=24637.6ms min=20140.1ms max=24961.8ms
Total session: avg=32404.6ms p50=32326.7ms p95=34663.6ms min=30166.8ms max=34991.0ms
Wall time: 35003.1ms
```

The Prometheus export confirms the distinction between session concurrency and
backend-job concurrency for this run:

| Metric from exported window | Observation |
|---|---:|
| Peak `sum(asr_ws_connections)` | 50 |
| Peak `sum(asr_worker_inflight_requests)` | 4 |
| Worker latency p95 over the captured 1-minute window | ~0.62-0.69 s |
| Triton client round-trip p95 over the captured 1-minute window | ~4.75-4.82 ms |
| GPU VRAM used | 9,108 -> 9,738 MiB |

Interpretation:

- The test successfully validated 50 synchronized WebSocket sessions.
- It did **not** create 50 simultaneous worker jobs: 89 of 140 transcript
  messages arrived before the synchronized flush, so natural pauses in the
  distinct clips spread the ASR work across the stream.
- The exported `asr_worker_fallback_total[5m]` query should not be treated as a
  per-run fallback count here because it is a rolling five-minute window that
  includes earlier traffic.
- `histogram_quantile(... asr_e2e_seconds_bucket ...)` saturated at the
  histogram's top `8s` bucket during this run, so it is a bucket ceiling rather
  than an exact p95 latency.

### Compared With The Prior `n=50 latest` Row

The earlier `n=50 latest` row and this fresh distinct-audio run are both useful,
but they describe different workload shapes:

| Test | Success | Data msgs | Pre-flush | Post-flush | Response wait avg | Est. real avg latency | Est. P95 latency | Est. max / approx P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior `n=50 latest` | 50/50 | 116 | 67 | 49 | 20,130.7 ms | ~131 ms | ~186 ms | ~189 ms |
| 2026-05-15 distinct-audio + Prometheus capture | 50/50 | 140 | 89 | 51 | 22,375.8 ms | ~2,376 ms | ~4,638 ms | ~4,962 ms |

The new run is the harsher mixed-utterance session-level validation. It should
not replace the prior row as a direct latency headline, because the fresh random
clips produced substantially more utterance finalization work and only reached
4 observed in-flight worker requests despite 50 live sockets.

## 50-Client True Worker-Concurrency Run

After the mixed-audio run above, the service concurrency limits were raised and
the timeout rails restored:

| Setting | Value |
|---|---:|
| `GATEWAY_MAX_INFLIGHT_WORKER` | `50` |
| `WORKER_MAX_JOBS` | `50` |
| `ASR_INFERENCE_TIMEOUT_MS` | `30000` |
| `WORKER_TIMEOUT_MS` | `35000` |

The worker and gateway containers were force-recreated so the new environment
was live before the next burst.

The first probe against `vaani_042.wav` with `n=1` showed that the clip was not
a single-final clip:

```text
Data messages: 3
Pre-flush data messages: 2
Post-flush data messages: 1
```

That means it was not a perfect "all work waits until flush" forcing function.
However, the subsequent `n=50` run still reached true backend concurrency,
which the worker metric proves directly.

Artifacts:

```text
artifacts/ws50_true_20260515T110840Z/
```

Command:

```bash
PYTHONPATH=/tmp/vaani_deps310 python3 tools/ws_burst_barrier.py \
  --ws ws://localhost:8001/ws/stt \
  --n 50 \
  --wav artifacts/vaani_hindi_10s_c50_seed20260515/vaani_042.wav \
  --max-audio-sec 10 \
  --send-mode realtime \
  --language-code hi \
  --response-timeout 120 \
  --idle-timeout 20 \
  --expect-data
```

Client-side results:

```text
Clients: 50
Connected: 50
Successful: 50
Failures: 0
First-audio release spread: 0.86ms
Flush send spread: 1.00ms
Bytes sent total: 16000000
Data messages: 150
Pre-flush data messages: 45
Post-flush data messages: 105
VAD messages: 0
Connect: avg=33.5ms p50=25.1ms p95=49.6ms min=18.1ms max=53.6ms
Stream send: avg=10000.4ms p50=10000.2ms p95=10001.0ms min=10000.1ms max=10001.0ms
Flush barrier wait: avg=1.5ms p50=1.7ms p95=1.8ms min=0.1ms max=1.8ms
Response wait: avg=30118.8ms p50=30178.7ms p95=30629.9ms min=29226.1ms max=30681.0ms
Total session: avg=40172.8ms p50=40228.8ms p95=40684.9ms min=39279.5ms max=40735.5ms
Wall time: 40747.9ms
```

Prometheus validation from the exported window
`2026-05-15T11:08:40Z -> 2026-05-15T11:09:31Z`:

| Metric | Observation |
|---|---:|
| Peak `sum(asr_ws_connections)` | 50 |
| Peak `sum(asr_worker_inflight_requests)` | 50 |
| `asr_worker_inflight_requests` samples at peak | 50 at `11:08:50Z`, `11:08:55Z`, and `11:09:00Z` |
| `sum(increase(asr_worker_fallback_total[5m]))` | 0 throughout the captured window |
| Worker inference p95 | reached the histogram top bucket at `5s` |
| Worker latency p95 | reached the histogram top bucket at `8s` |
| Triton client round-trip p95 | reached the histogram top bucket at `2.5s` |
| Triton average queue duration | peaked at ~`1.95s` |
| GPU VRAM used | 9,852 -> 9,910 MiB |

Interpretation:

- This is the first run in this series that validates **50 simultaneous worker
  jobs**, not only 50 live WebSocket sessions.
- The proof is the worker-side gauge itself: `asr_worker_inflight_requests`
  reached 50 and remained at 50 across three consecutive 5-second Prometheus
  scrapes.
- The run stayed clean at the session level: 50/50 clients succeeded and the
  rolling fallback query remained at 0 for the captured window.
- Several latency histograms saturated at their current top buckets during the
  run. That means this run proves concurrency, but the existing histogram bucket
  layout is too shallow to report exact p95 tails once the system is stressed at
  50-way worker concurrency.
- A separately started 100 ms direct sampler under
  `artifacts/ws50_true_20260515T110941Z/worker_inflight_100ms.csv` captured only
  zeros because it was started after the burst had already completed; it should
  not be used as evidence for or against the run.

## Validated Concurrency Comparison

Latency validation rule:

```text
estimated_real_latency_ms = response_wait_avg_ms - idle_timeout_ms
```

This removes the client-side idle grace from `response_wait`. The resulting
numbers are useful only for rows where real ASR inference was enabled and the
run is not dominated by fallback or non-inference responses.

| Test | Inference status | Clients | Idle timeout | Success | Data msgs | Pre-flush | Post-flush | Response wait avg | Est. real avg latency | Est. P95 latency | Est. max / approx P99 | Valid for ASR inference claim? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Old n=30 single WAV | Inference on, but fallback seen | 30 | 3s | 30/30 | 90 | 30 | 60 | 5,665.1 ms | ~2,665 ms | ~2,875 ms | ~2,878 ms | Partial, because 17/30 first utterances had `worker-fallback` |
| n=6 earlier | No inference / not valid | 6 | 20s | 6/6 | 12 | 6 | 6 | 20,818.7 ms | ~819 ms | ~996 ms | ~998 ms | No |
| n=8 earlier | No inference / not valid | 8 | 20s | 8/8 | 17 | 9 | 8 | 20,915.3 ms | ~915 ms | ~1,246 ms | ~1,257 ms | No |
| n=6 latest before correction | No inference / not valid | 6 | 20s | 6/6 | 12 | 6 | 6 | 20,019.9 ms | ~20 ms | ~21 ms | ~21 ms | No |
| n=16 earlier | No inference / not valid | 16 | 20s | 16/16 | 40 | 25 | 15 | 20,051.0 ms | ~51 ms | ~60 ms | ~61 ms | No |
| n=30 earlier | No inference / not valid | 30 | 20s | 30/30 | 69 | 40 | 29 | 20,085.6 ms | ~86 ms | ~106 ms | ~107 ms | No |
| n=50 prior latest | Inference on | 50 | 20s | 50/50 | 116 | 67 | 49 | 20,130.7 ms | ~131 ms | ~186 ms | ~189 ms | Yes |
| n=50 distinct-audio + Prometheus capture | Inference on | 50 | 20s | 50/50 | 140 | 89 | 51 | 22,375.8 ms | ~2,376 ms | ~4,638 ms | ~4,962 ms | Yes for session-level inference; no for a 50-way worker-concurrency claim |
| n=50 true worker-concurrency run | Inference on | 50 | 20s | 50/50 | 150 | 45 | 105 | 30,118.8 ms | ~10,119 ms | ~10,630 ms | ~10,681 ms | Yes; Prometheus observed 50 in-flight worker requests |

Validation notes:

- The old 30-client single-WAV row is only a partial inference result because
  the first utterance fell back for 17 of 30 clients.
- Rows marked `No inference / not valid` are useful as transport/session
  checks, but should not be used to claim ASR inference latency.
- The `n=50 prior latest` row remains the cleanest low-latency ASR inference
  claim in this comparison: inference was on, all 50 sessions succeeded, and
  the estimated latency is computed after removing the 20-second idle grace.
- The newer 50-client distinct-audio run is also an inference-on session-level
  validation, but exported Prometheus telemetry shows it is **not** evidence of
  50 simultaneous worker jobs: `asr_ws_connections` peaked at 50 while
  `asr_worker_inflight_requests` peaked at 4.
- The newest 50-client run is the valid **true worker-concurrency** claim:
  inference was on, all 50 sessions succeeded, fallbacks stayed at 0 in the
  captured window, and Prometheus observed
  `asr_worker_inflight_requests = 50`.

## What This Proves

- The gateway and worker completed a 30-client barrier-synchronized WebSocket
  burst with zero session failures.
- The deployment accepted 30 simultaneous real-time audio streams using the
  configured 30/30 gateway and worker concurrency limits.
- Raising `ASR_INFERENCE_TIMEOUT_MS` from 4s to 30s removed the first-utterance
  `fallback_timeout` responses for this 30-client Vaani burst.
- The updated barrier script can run a reproducible 30-client burst using 30
  different Vaani WAV files instead of replaying one shared file.
- The validated comparison table now separates the prior low-latency 50-client
  inference-on row, the distinct-audio 50-client session-level run, and the
  newest 50-client true worker-concurrency run.
- With `GATEWAY_MAX_INFLIGHT_WORKER=50` and `WORKER_MAX_JOBS=50`, the deployment
  executed 50 simultaneous worker jobs during the latest same-WAV burst.
- The barrier script produced a tight concurrency probe: audio start and flush
  spread were both within roughly 5 ms in the final verification run.

## What It Does Not Prove

- It does not prove production saturation capacity; these were 30-client
  verification bursts, not a long-running concurrency sweep.
- It does not measure model accuracy. The generated transcript should be treated
  as observed output, not a quality score.

## Follow-Up

For stricter ASR validation, update `tools/ws_burst_barrier.py` to optionally
fail a client when any data message contains `transcript="worker-fallback"` or
`language_source="fallback_timeout"`. That would make session success reflect
model completion, not only WebSocket completion.
