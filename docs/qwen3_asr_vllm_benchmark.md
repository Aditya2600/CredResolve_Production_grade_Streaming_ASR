# Qwen3-ASR vLLM Benchmarking Instructions

This document provides instructions on how to set up, run, and evaluate Qwen3-ASR models (`Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B`) using a vLLM/OpenAI-compatible server endpoint.

## 1. Environment Setup

Install a separate conda environment (do NOT install inside the NeMo environment to prevent dependency conflicts):

```bash
conda create -n qwen3-asr-vllm python=3.12 -y
conda activate qwen3-asr-vllm
pip install -r requirements-qwen3-asr-vllm.txt
```

---

## 2. Benchmark Qwen3-ASR-0.6B

### Start the 0.6B vLLM Server

From your `qwen3-asr-vllm` environment:

```bash
CUDA_VISIBLE_DEVICES=0 qwen-asr-serve Qwen/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 8000
```

### Run the Benchmark

From your main ASR/NeMo environment, export the required environment configurations and run the contextual benchmark script:

```bash
export QWEN_ASR_ENABLE=1
export QWEN_ASR_VLLM_BASE_URL=http://localhost:8000/v1
export QWEN_ASR_MODELS=Qwen/Qwen3-ASR-0.6B
export QWEN_ASR_LANGUAGE=Hindi
export QWEN_ASR_MODE=chat_completions

python tools/benchmarks/contextual_asr_hindi_benchmark.py
```

---

## 3. Benchmark Qwen3-ASR-1.7B

### Start the 1.7B vLLM Server

From your `qwen3-asr-vllm` environment:

```bash
CUDA_VISIBLE_DEVICES=0 qwen-asr-serve Qwen/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 8000
```

### Run the Benchmark

From your main ASR/NeMo environment, export the required environment configurations and run the contextual benchmark script:

```bash
export QWEN_ASR_ENABLE=1
export QWEN_ASR_VLLM_BASE_URL=http://localhost:8000/v1
export QWEN_ASR_MODELS=Qwen/Qwen3-ASR-1.7B
export QWEN_ASR_LANGUAGE=Hindi
export QWEN_ASR_MODE=chat_completions

python tools/benchmarks/contextual_asr_hindi_benchmark.py
```

---

## 4. Key Notes & Server Limitations

* **Single Server Limitation**: A single running instance of a vLLM server normally serves one model at a time. 
* **Automatic Detection**: If both models are configured in `QWEN_ASR_MODELS`, the benchmark script automatically queries the vLLM server via `/v1/models` and skips any requested model not matching the currently served model.
* **Failure Isolation**: Any failure or lack of server support for Qwen models will be fully isolated and won't affect other ASR evaluations. Failed/skipped models will be listed under the **Failed / Skipped Models** section of the report instead of skewing the aggregate WER performance table.

---

## 5. Benchmarking Analysis & Outlier Handling

When evaluating Qwen3-ASR models (especially the smaller `0.6B` model) over a diverse benchmark such as the `sarvamai/contextual_asr_benchmark`, you may observe a very high raw aggregate Word Error Rate (WER) (e.g. >1000%).

### The Repetition Loop Issue
This is a known issue where smaller autoregressive models can get stuck in infinite repetition loops (e.g., repeating a Chinese character like `，对` over 30,000 times) when encountering short, silent, or noisy audio segments.

* **Raw WER (0.6B)**: **1217.62%** (inflated by a single repetition loop at sample index 474).
* **Filtered WER (0.6B) - Excluding Index 474**: **70.34%** (Substitutions: 782, Deletions: 1000, Insertions: 224, Reference Words: 2852).

Despite this outlier, the model's successfully decoded samples show highly accurate phonetic transcription capabilities in Hindi (e.g., `ाँ बोल` for `हाँ बोल` and `ी मुझसे बात कर रही हो आप बोलिए` for `जी मुझसे बात कर रही हो आप बोलिए`).

### Mitigation Strategies
1. **Model Upgrading**: Use the `1.7B` parameter model (`Qwen/Qwen3-ASR-1.7B`) which is significantly more robust against infinite loops.
2. **Generation Parameters**: Configure a `repetition_penalty` (e.g. `1.1` or `1.2`) or `temperature` values on your vLLM server to deter infinite repetition cycles.
3. **Outlier Filtering**: Apply an insertions threshold (e.g., discard predictions with word counts exceeding 5x of the reference length) to prevent outliers from skewing aggregate corpus WER statistics.
