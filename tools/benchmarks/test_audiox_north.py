#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np
import torch

# Configure HF cache directory under local workspace
def configure_hf_cache_env() -> None:
    cache_root = (Path.cwd() / ".cache").resolve()
    hf_home = cache_root / "huggingface"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))

configure_hf_cache_env()

# Import ASR text normalizer from repository
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tools.asr_text_normalizer import normalize_asr_text
    from tools.compute_wer import edit_distance, normalize as compute_wer_normalize
except ImportError:
    from asr_text_normalizer import normalize_asr_text
    from compute_wer import edit_distance, normalize as compute_wer_normalize

def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    src_positions = np.arange(audio.shape[0], dtype=np.float32)
    dst_length = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)
    
def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)

def load_audiox_model_and_processor(device: torch.device, hf_token: str = None):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print(">>> Loading audioX-north-v1 Processor (jiviai/audioX-north-v1)...", flush=True)
    processor = WhisperProcessor.from_pretrained(
        "jiviai/audioX-north-v1",
        token=hf_token
    )

    print(">>> Loading audioX-north-v1 Model weights (jiviai/audioX-north-v1)...", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        "jiviai/audioX-north-v1",
        token=hf_token
    ).to(device)
    
    if device.type == "cuda":
        model = model.half()
        
    model.eval()
    return model, processor

def transcribe_audiox(model, processor, audio_arr, device: torch.device) -> str:
    inputs = processor(audio_arr, sampling_rate=16000, return_tensors="pt").to(device)
    if device.type == "cuda":
        inputs = inputs.to(dtype=torch.float16)

    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
        
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()

def main():
    parser = argparse.ArgumentParser(description="Test audioX-north-v1 model on a single audio file or dataset sample.")
    parser.add_argument("--audio", type=str, help="Path to a local audio file to transcribe.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the dataset sample to transcribe if --audio is not provided (default: 0).")
    parser.add_argument("--all-samples", action="store_true", help="Transcribe and evaluate the entire 611-sample benchmark dataset.")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace User Access Token (needed if gated/authorized model). Can also be set via HF_TOKEN environment variable.")
    
    args = parser.parse_args()
    
    hf_token = args.token or os.environ.get("HF_TOKEN")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print(" AUDIOX-NORTH-V1 LOCAL DIAGNOSTIC & TEST UTILITY")
    print("="*80)
    print(f"Device: {device}")
    print("-"*80)

    # Load Model & Processor
    try:
        model, processor = load_audiox_model_and_processor(device, hf_token)
    except Exception as e:
        print(f"Failed to load audioX Model/Processor: {e}")
        print("\nNote: 'jiviai/audioX-north-v1' is a gated Hugging Face repository.")
        print("Please ensure you have accepted the license terms on the HF page and provide a valid token.")
        print("You can pass the token using the `--token <your_token>` flag or setting the `HF_TOKEN` environment variable.")
        sys.exit(1)

    # 1. Handle all-samples evaluation
    if args.all_samples:
        print("Loading all samples from 'sarvamai/contextual_asr_benchmark'...")
        try:
            from datasets import load_dataset
            dataset_dict = load_dataset("sarvamai/contextual_asr_benchmark", "hi-IN")
            splits = list(dataset_dict.keys())
            ds = dataset_dict[splits[0]]
            samples = list(ds)
            print(f"Successfully loaded {len(samples)} samples.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)
            
        print("-"*80)
        print(f"Evaluating {len(samples)} samples with audioX-north-v1 locally...")
        
        predictions_log = []
        total_s, total_d, total_i, total_words = 0, 0, 0, 0
        
        for idx, sample in enumerate(samples):
            audio_data = sample["audio"]
            audio_arr = to_mono(audio_data["array"])
            audio_arr = resample_linear(audio_arr, audio_data["sampling_rate"], 16000)
            reference = sample.get("transcript") or sample.get("text") or ""
            
            try:
                transcript = transcribe_audiox(model, processor, audio_arr, device)
            except Exception as e:
                print(f"\nSample {idx} failed: {e}")
                continue
                
            # Calculate WER
            clean_ref = normalize_asr_text(reference)
            clean_hyp = normalize_asr_text(transcript)
            
            ref_words = compute_wer_normalize(clean_ref)
            hyp_words = compute_wer_normalize(clean_hyp)
            
            s, d, i = edit_distance(ref_words, hyp_words)
            words = len(ref_words)
            wer = ((s + d + i) / words) if words else (0.0 if not hyp_words else 1.0)
            
            total_s += s
            total_d += d
            total_i += i
            total_words += words
            
            predictions_log.append({
                "sample_index": idx,
                "reference": reference,
                "hypothesis": transcript,
                "wer": round(wer, 4),
                "words": words,
                "substitutions": s,
                "deletions": d,
                "insertions": i
            })
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
                print(f"Processed {idx + 1}/{len(samples)} samples", flush=True)
                
        # Write results to timestamped JSONL
        import datetime
        import json
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("artifacts/benchmarks/contextual_asr_hindi")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        pred_file = out_dir / f"predictions_audiox_only_{timestamp}.jsonl"
        with pred_file.open("w", encoding="utf-8") as f:
            for log_entry in predictions_log:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        final_wer = (total_s + total_d + total_i) / total_words if total_words else 0.0
        
        print("\n" + "="*80)
        print(" AUDIOX-NORTH FULL BENCHMARK RESULTS")
        print("="*80)
        print(f"Model ID:          jiviai/audioX-north-v1")
        print(f"Total Evaluated:   {len(predictions_log)} / {len(samples)} samples")
        print(f"Total Ref Words:   {total_words}")
        print(f"Substitutions:     {total_s}")
        print(f"Deletions:         {total_d}")
        print(f"Insertions:        {total_i}")
        print(f"Aggregate WER:     {final_wer * 100:.2f}%")
        print("-"*80)
        print(f"Wrote predictions to: {pred_file}")
        print("="*80 + "\n")
        
        sys.exit(0)

    # 2. Handle single audio / single sample evaluation
    audio_arr = None
    sample_rate = 16000
    reference = None
    
    if args.audio:
        print(f"Loading local audio file: {args.audio}")
        try:
            data, sr = sf.read(args.audio)
            audio_arr = to_mono(data)
            audio_arr = resample_linear(audio_arr, sr, 16000)
            print(f"Successfully loaded and resampled audio from {sr}Hz to 16000Hz. Duration: {len(audio_arr)/16000:.2f}s")
        except Exception as e:
            print(f"Error loading audio file: {e}")
            sys.exit(1)
    else:
        print(f"Loading sample at index {args.sample_index} from 'sarvamai/contextual_asr_benchmark'...")
        try:
            from datasets import load_dataset
            dataset_dict = load_dataset("sarvamai/contextual_asr_benchmark", "hi-IN")
            splits = list(dataset_dict.keys())
            ds = dataset_dict[splits[0]]
            sample = ds[args.sample_index]
            
            audio_data = sample["audio"]
            audio_arr = to_mono(audio_data["array"])
            audio_arr = resample_linear(audio_arr, audio_data["sampling_rate"], 16000)
            reference = sample.get("transcript") or sample.get("text") or ""
            print("Successfully loaded dataset sample.")
            print(f"Dataset Reference Text: '{reference}'")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    print("-"*80)
    print("Transcribing audio locally with audioX-north-v1...")
    try:
        transcript = transcribe_audiox(model, processor, audio_arr, device)
    except Exception as e:
        print(f"Transcription Failed: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print(" RESULTS")
    print("="*80)
    print(f"Transcription:\n{repr(transcript)}")
    
    if reference:
        print("-"*80)
        print(f"Reference Text:\n{repr(reference)}")
        
        # Calculate sample WER
        clean_ref = normalize_asr_text(reference)
        clean_hyp = normalize_asr_text(transcript)
        
        ref_words = compute_wer_normalize(clean_ref)
        hyp_words = compute_wer_normalize(clean_hyp)
        
        s, d, i = edit_distance(ref_words, hyp_words)
        total_words = len(ref_words)
        wer = ((s + d + i) / total_words) * 100 if total_words else (0.0 if not hyp_words else 100.0)
        
        print("-"*80)
        print("WER Calculation:")
        print(f"  - Normalized Reference: '{clean_ref}'")
        print(f"  - Normalized Hypothesis: '{clean_hyp}'")
        print(f"  - Substitutions: {s}")
        print(f"  - Deletions:     {d}")
        print(f"  - Insertions:    {i}")
        print(f"  - Total Words:   {total_words}")
        print(f"  - Sample WER:    {wer:.2f}%")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
