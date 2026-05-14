# Streaming ASR concurrency test — 50 users × 10s Hindi audio (realtime)

## Results

| total_requests | RPS  |    p50 |     p90 |     p95 | average | users |
| -------------: | ---: | -----: | ------: | ------: | ------: | ----: |
|             50 | 3.97 | 896 ms | 1452 ms | 1875 ms |  946 ms |    50 |

- **Throughput:** 39.74 audio-sec/sec (500s of audio processed in 12.58s wall time)
- **Real-time factor (RTF):** p50 = 1.09x, p95 = 1.19x
- **Success rate:** 50 / 50 (no failures)
- **Tail latency p99 / max:** 2470 ms

Latency in the table is **tail latency** = `total_ms − 10000ms` — i.e. the overhead on top of the mandatory 10s realtime-pacing time. It captures WS handshake + last-chunk-to-final turnaround + bridge / SimpliSmart processing.

## Flow

1. Pulled 50 Hindi clips from `ARTPARK-IISc/Vaani-transcription-part`, truncated each to exactly 10s.
2. Opened 50 concurrent WebSocket streams to the local bridge, sending audio in 32ms chunks at realtime speed (1x).
3. Bridge forwarded speech segments to SimpliSmart `/v1/predict` over HTTP (64-worker thread pool).
4. Timed each request WS-connect → final ack, subtracted the 10s pacing time, then aggregated p50 / p90 / p95 / avg across all 50.

## Setup

- **Dataset:** [ARTPARK-IISc/Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part), Hindi config, seed=42.
- **Sample prep:** [SimpliSmart/download_vaani_sample.py](../SimpliSmart/download_vaani_sample.py) with `--min-duration 10 --target-duration 10 --count 50`.
- **Client:** [SimpliSmart/benchmark_asr.py](../SimpliSmart/benchmark_asr.py), invoked with `--concurrency 50 --random 50 --realtime --language hi`.
- **Bridge:** [SimpliSmart/stt_test_client_ws.py](../SimpliSmart/stt_test_client_ws.py), `ThreadPoolExecutor(max_workers=64)`, Silero VAD segmenting before HTTP forward.
- **Upstream:** SimpliSmart HTTP `/v1/predict`.

### Commands

```bash
# 1. Download 50 clips, each truncated to 10s
python3 SimpliSmart/download_vaani_sample.py \
  --language Hindi --count 50 --seed 42 \
  --min-duration 10 --target-duration 10 \
  --out SimpliSmart/vaani_hi_10s

# 2. Start the bridge (Terminal 1)
python3 SimpliSmart/stt_test_client_ws.py

# 3. Run the benchmark (Terminal 2)
python3 SimpliSmart/benchmark_asr.py \
  --dataset SimpliSmart/vaani_hi_10s \
  --random 50 --seed 42 \
  --concurrency 50 --language hi --realtime \
  --output SimpliSmart/vaani_hi_10s_c50.csv
```

## Interpreting the numbers

- **Why RPS is only 3.97**, despite 50 concurrent users: each session streams 10s of audio at 1x speed, so a single request cannot complete in less than 10s. The ceiling for this setup is `50 users / 10s = 5 RPS`; we hit 80% of that ceiling, with the remaining 20% explained by the ~946 ms average tail. RPS in a streaming benchmark is a property of the test (concurrency × clip length), not server capacity — **throughput (audio-sec/sec) and RTF are the right capacity metrics here.**
- **RTF p95 = 1.19x** means the slowest 5% of requests finished within 19% of audio duration. Anything under ~1.5x is healthy for a realtime streaming service.
- **No saturation observed.** Wall time (12.58s) was barely larger than the longest single request (11.88s), meaning all 50 sessions ran nearly in lockstep without queueing. The bridge thread pool (64) and SimpliSmart upstream both have headroom — a sweep at 75 / 100 / 150 / 200 concurrency is needed to find the saturation knee.

## Raw output (summary line)

```
SUMMARY
============================================================
  Completed       : 50/50 ok, 0 failed
  Wall time       : 12.58s
  Audio processed : 500.00s
  Throughput      : 39.74 audio-sec/sec
  Request ms      : avg=10946, p50=10896, p95=11875
  RTF             : avg=1.09x, p50=1.09x, p95=1.19x
```

Per-request rows: [SimpliSmart/vaani_hi_10s_c50.csv](../SimpliSmart/vaani_hi_10s_c50.csv).
