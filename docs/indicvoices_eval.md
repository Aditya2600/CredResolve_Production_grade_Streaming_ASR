# IndicVoices Hindi WER Evaluation

This guide benchmarks the Triton-backed ASR through the gateway WebSocket path so the results reflect the real serving stack rather than a direct model call.

## Preconditions

- The Triton stack is running and the gateway WebSocket endpoint is reachable.
- The gateway accepts your API key, for example `Api-Subscription-Key: dev`.
- You have access to the gated `ai4bharat/IndicVoices` dataset.
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is set in your shell, or you pass `--hf-token` explicitly.

Example Triton startup:

```bash
cp .env.example .env
export ASR_SUPPORTED_LANGS=hi,en,bn,ta,te
docker compose -f docker-compose.yml -f docker-compose.triton.yml up --build -d
```

## Smoke Test

Run a quick 5-sample Hindi validation first:

```bash
python tools/eval_indicvoices_wer.py \
  --ws ws://localhost/ws/stt \
  --api-key dev \
  --dataset-config hindi \
  --split valid \
  --language hi \
  --limit 5 \
  --hf-token "$HF_TOKEN" \
  --out-tsv /tmp/indicvoices_hindi_smoke.tsv \
  --out-summary-json /tmp/indicvoices_hindi_smoke_summary.json \
  --out-errors-jsonl /tmp/indicvoices_hindi_smoke.jsonl \
  --top-errors 5
```

## Dataset Profile

Observed for `ai4bharat/IndicVoices`, config `hindi`:

- `train` rows: `383004`
- `valid` rows: `4740`
- average words per row: `15.47`
- average audio duration per row: `5.55s`
- min/max words per row: `1` / `103`
- min/max duration per row: `0.159s` / `29.104s`
- total audio in `valid`: `7.31` hours

Approximate scale by benchmark size:

| Benchmark Size | Rows | Approx. Words | Approx. Audio | Best Use |
|---|---:|---:|---:|---|
| Smoke | 5 | 77 | 28s | Sanity check setup and websocket path |
| Development | 500 | 7737 | 46.3 min | Fast iteration and error-pattern discovery |
| Confirmation | 1000 | 15473 | 92.6 min | Compare close model/config changes |
| Full valid | 4740 | 73343 | 7.31 h | Final reporting or release-quality benchmark |

Why `500` first instead of `1000`:

- `500` rows are already about `10.5%` of the full Hindi `valid` split.
- `500` is large enough to expose the main failure modes and provide a stable directional WER for iteration.
- `1000` nearly doubles wall-clock runtime and serving cost for much less added debugging value.
- This benchmark runs through websocket + gateway + Triton, so total runtime includes request overhead, segmentation, and finalization, not just raw audio duration.

Recommendation:

- Use `500` for development and fast comparison between model/settings changes.
- Use `1000` when differences are small and you need higher confidence.
- Use `4740` for a full benchmark when preparing a final report.

Note: the current evaluator reads the first `N` rows from the streaming split. If you need a more representative slice later, add reproducible sampling rather than relying on the first `500` rows.

## Development Benchmark

```bash
python tools/eval_indicvoices_wer.py \
  --ws ws://localhost/ws/stt \
  --api-key dev \
  --dataset-config hindi \
  --split valid \
  --language hi \
  --limit 500 \
  --out-tsv artifacts/indicvoices_hindi_500.tsv \
  --out-summary-json artifacts/indicvoices_hindi_500_summary.json \
  --out-errors-jsonl artifacts/indicvoices_hindi_500_errors.jsonl \
  --top-errors 20
```

## Confirmation Benchmark

```bash
python tools/eval_indicvoices_wer.py \
  --ws ws://localhost/ws/stt \
  --api-key dev \
  --dataset-config hindi \
  --split valid \
  --language hi \
  --limit 1000 \
  --out-tsv artifacts/indicvoices_hindi_1000.tsv \
  --out-summary-json artifacts/indicvoices_hindi_1000_summary.json \
  --out-errors-jsonl artifacts/indicvoices_hindi_1000_errors.jsonl \
  --top-errors 20
```

## Full Benchmark

```bash
python tools/eval_indicvoices_wer.py \
  --ws ws://localhost/ws/stt \
  --api-key dev \
  --dataset-config hindi \
  --split valid \
  --language hi \
  --limit 4740 \
  --out-tsv artifacts/indicvoices_hindi_valid_full.tsv \
  --out-summary-json artifacts/indicvoices_hindi_valid_full_summary.json \
  --out-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --top-errors 20
```

## Iteration Comparison

After a second full run, compare the first and second iterations with:

```bash
python tools/compare_indicvoices_iterations.py \
  --first-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --second-errors-jsonl artifacts/indicvoices_hindi_valid_full_iter2_errors.jsonl \
  --out-summary-json artifacts/indicvoices_hindi_valid_comparison_summary.json \
  --out-comparison-jsonl artifacts/indicvoices_hindi_valid_comparison.jsonl \
  --out-repeated-error-indices artifacts/indicvoices_hindi_valid_repeated_error_indices.txt \
  --top-persistent 20
```

If you are evaluating context biasing against a domain phrase file, add `--phrases-file` to compute keyword precision, recall, and F1 for both iterations:

```bash
python tools/compare_indicvoices_iterations.py \
  --first-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --second-errors-jsonl artifacts/indicvoices_hindi_valid_full_iter2_errors.jsonl \
  --phrases-file context_biasing/phrases/hi.txt \
  --out-summary-json artifacts/indicvoices_hindi_valid_comparison_summary.json \
  --out-comparison-jsonl artifacts/indicvoices_hindi_valid_comparison.jsonl
```

To export the samples that were wrong in both iterations:

```bash
python tools/compare_indicvoices_iterations.py \
  --first-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --second-errors-jsonl artifacts/indicvoices_hindi_valid_full_iter2_errors.jsonl \
  --export-repeated-errors-dir artifacts/indicvoices_hindi_valid_repeated_errors \
  --dataset-config hindi \
  --split valid
```

Important:

- `error_both` means the sample was wrong in both iterations.
- `fixed_in_second` means the sample was wrong in the first iteration and clean in the second.
- `regressed_in_second` means the sample was clean in the first iteration and wrong in the second.
- the repeated-error export comes from the `valid` split, so it is useful for analysis and hard-example mining, but it should not remain your final held-out benchmark after training.

## Monitoring During Eval

Useful Prometheus checks while the benchmark is running:

```bash
./tools/prom_query.sh 'sum(asr_ws_connections)'
./tools/prom_query.sh 'sum(asr_worker_inflight_requests)'
./tools/prom_query.sh 'histogram_quantile(0.95, sum(rate(asr_worker_latency_seconds_bucket[5m])) by (le))'
./tools/prom_query.sh 'histogram_quantile(0.95, sum(rate(asr_worker_inference_seconds_bucket[5m])) by (le))'
./tools/prom_query.sh 'histogram_quantile(0.95, sum(rate(asr_e2e_seconds_bucket[5m])) by (le))'
./tools/prom_query.sh 'increase(asr_worker_fallback_total[30m])'
```

Interpretation:

- `asr_ws_connections`: active websocket sessions
- `asr_worker_inflight_requests`: current backend concurrency
- `asr_worker_latency_seconds`: worker request latency
- `asr_worker_inference_seconds`: pure model/Triton inference latency
- `asr_e2e_seconds`: end-to-end utterance latency
- `asr_worker_fallback_total`: timeout/error/not-ready fallbacks returned by the worker

Important:

- For `tools/load_ws.py`, reported session duration includes real-time audio streaming plus the post-`flush` silence wait.
- For `tools/eval_indicvoices_wer.py`, the benchmark is effectively sequential, so high `worker-fallback` counts or high `asr_worker_latency_seconds` are more important than websocket concurrency.

## Outputs

- `--out-tsv`: reference and hypothesis pairs for the processed samples.
- `--out-summary-json`: aggregate metrics including processed count, edit totals, WER, and latency aggregates when available.
- `--out-errors-jsonl`: one JSON object per sample, ranked worst-first for successful samples and followed by any failed samples.
- stdout summary: processed count, failures, substitutions, deletions, insertions, raw corpus WER, and clean ASR-only WER.
- stdout top errors: the worst clean ASR samples by `sample_wer` descending, tie-broken by total edit count descending.

## Reading The Ranked Error Report

Each JSONL record includes:

- `index`
- `reference`
- `hypothesis`
- `ref_words` (normalized token list)
- `hyp_words` (normalized token list)
- `substitutions`
- `deletions`
- `insertions`
- `sample_wer`
- `evaluation_bucket`
- `counted_in_clean_wer`
- `status`
- `error_detail` for failed samples only

Interpretation tips:

- High substitutions usually indicate wrong words or language confusions.
- High deletions usually indicate dropped words, weak audio, or segmentation misses.
- High insertions usually indicate hallucinated extra words or merge errors.
- `evaluation_bucket=dataset-noise` means the reference contains `<unintelligible>`, so that row is excluded from clean ASR-only WER.
- `evaluation_bucket=infra-failure` means the hypothesis contains `worker-fallback`, so that row is excluded from clean ASR-only WER.
- `evaluation_bucket=asr` is the normal bucket used for clean ASR-only WER.
- The printed top-error section is the fastest place to inspect where the model is failing on Hindi utterances after excluding dataset noise and infra failures.

## Raw vs Clean WER

The evaluator now reports two WER views:

- `wer` / `wer_percent`: raw benchmark WER across all successful samples.
- `clean_wer` / `clean_wer_percent`: ASR-only WER after excluding:
  - `dataset-noise`: references containing `<unintelligible>`
  - `infra-failure`: hypotheses containing `worker-fallback`

The clean summary also includes:

- `dataset_noise_samples`
- `infra_failure_samples`
- `clean_asr_processed`
- `clean_reference_words`
- `clean_substitutions`
- `clean_deletions`
- `clean_insertions`

Use `clean_wer` for model-quality reporting and training-target selection. Keep raw `wer` for auditability and dataset/infra health tracking.

## Mine Train Examples From Valid Error Patterns

Do not fine-tune directly on the `valid` benchmark rows if you want to preserve an honest held-out evaluation set.

Instead:

1. Run `tools/eval_indicvoices_wer.py` on `valid`.
2. Use the resulting error JSONL to identify trainable error patterns.
3. Mine similar examples from `train` plus optional call manifests.
4. Fine-tune on the mined `train`/domain data.
5. Re-evaluate on the untouched `valid` split.

Example mining command:

```bash
python tools/mine_indicvoices_train_examples.py \
  --errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --dataset-config hindi \
  --split train \
  --phrases-file context_biasing/phrases/hi.txt \
  --call-manifest-jsonl artifacts/domain_calls_train_manifest.jsonl \
  --scan-limit 50000 \
  --per-bucket-limit 300 \
  --normal-limit 300 \
  --export-dir artifacts/indicvoices_hindi_train_mined
```

Outputs:

- `artifacts/indicvoices_hindi_train_mined/summary.json`
- `artifacts/indicvoices_hindi_train_mined/manifest.jsonl`
- `artifacts/indicvoices_hindi_train_mined/audio/`

The mining tool focuses on the main trainable buckets from the valid error profile:

- `short_acknowledgement`
- `short_conversational`
- `numbers_currency`
- `names_entities`
- `filler_insertion`
- `generic_asr_error`

Ignored at mining time:

- `dataset_noise`: references containing `<unintelligible>`
- `infra_failure`: hypotheses containing `worker-fallback`
