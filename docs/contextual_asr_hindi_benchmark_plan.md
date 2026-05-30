# Contextual ASR Hindi Benchmark Plan

This document details the plan to benchmark five Automatic Speech Recognition (ASR) models on the Hindi subset of the `sarvamai/contextual_asr_benchmark` dataset, calculate the Word Error Rate (WER), and generate a comparison report.

## Model List
1. **Qwen3-ASR**: `Qwen/Qwen3-ASR-1.7B` (or `0.6B`)
2. **audioX-south**: `jiviai/audioX-south-v1`
3. **Voxtral-Mini**: `mistralai/Voxtral-Mini-4B-Realtime-2602`
4. **Wav2Vec2-Hindi**: `theainerd/Wav2Vec2-large-xlsr-hindi`
5. **IndicConformer**: `ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large` (loaded from local base model `/home/ubuntu/models/indicconformer/IndicConformer.nemo`)

## Evaluation Dataset
- **Name**: `sarvamai/contextual_asr_benchmark`
- **Filter**: Hindi only (`language` matches `hi` or `hi-IN` or starts with `hi`)

## Implementation Details
1. **Unified Normalization & Evaluation Tool**:
   - Reuses `tools/asr_text_normalizer.py` and `tools/compute_wer.py`.
2. **Model Running Strategy**:
   - Sequential model evaluation to prevent CUDA Out-of-Memory (OOM) errors.
   - Clears CUDA memory after each model evaluation.
3. **Output Files**:
   - Predictions: `artifacts/benchmarks/contextual_asr_hindi/predictions.jsonl`
   - Comparison Report: `artifacts/benchmarks/contextual_asr_hindi/comparison_report.md`
