#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np

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

try:
    from tools.benchmarks.contextual_asr_hindi_benchmark import clean_qwen_transcript, resample_linear, to_mono
except ImportError:
    # Inline clean_qwen_transcript backup if imports fail
    import re
    def clean_qwen_transcript(text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"(?i)\blanguage\s*:\s*[a-zA-Z0-9_]+", "", text)
        text = re.sub(r"(?i)\blanguage\s+[a-zA-Z0-9_]+", "", text)
        text = text.strip()
        quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
        while len(text) >= 2 and text[0] in quote_pairs and text[-1] == quote_pairs[text[0]]:
            text = text[1:-1].strip()
        return text.strip()
    
    def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio.astype(np.float32, copy=False)
        src_positions = np.arange(audio.shape[0], dtype=np.float32)
        dst_length = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
        dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
        return np.interp(dst_positions, src_positions, audio).astype(np.float32)
        
    def to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio.astype(np.float32, copy=False)
        return audio.mean(axis=1, dtype=np.float32)

def transcribe_qwen_raw_and_cleaned(
    audio_array,
    sample_rate: int,
    base_url: str,
    model: str,
    language: str = "Hindi",
    mode: str = "chat_completions",
) -> tuple[str, str]:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_name = tmp_wav.name
    try:
        sf.write(tmp_wav_name, audio_array, sample_rate, subtype="PCM_16")
        
        if mode == "chat_completions":
            with open(tmp_wav_name, "rb") as f:
                wav_bytes = f.read()
            import base64
            base64_audio = base64.b64encode(wav_bytes).decode("utf-8")
            audio_url = f"data:audio/wav;base64,{base64_audio}"
            
            prompt = f"Transcribe the following audio into {language}. Return only the transcription text."
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "audio_url", "audio_url": {"url": audio_url}}
                        ]
                    }
                ]
            )
            raw_transcript = response.choices[0].message.content
            cleaned_transcript = clean_qwen_transcript(raw_transcript)
            return raw_transcript, cleaned_transcript
            
        elif mode == "audio_transcriptions":
            with open(tmp_wav_name, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language="hi" if language.lower() in ["hindi", "hi"] else None
                )
                raw_transcript = response.text
                cleaned_transcript = clean_qwen_transcript(raw_transcript)
                return raw_transcript, cleaned_transcript
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    finally:
        if os.path.exists(tmp_wav_name):
            os.remove(tmp_wav_name)

def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR model on a single audio file or dataset sample.")
    parser.add_argument("--audio", type=str, help="Path to a local audio file to transcribe.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the dataset sample to transcribe if --audio is not provided (default: 0).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-1.7B", help="Model name registered on the vLLM server (default: Qwen/Qwen3-ASR-1.7B).")
    parser.add_argument("--base-url", type=str, default=None, help="vLLM OpenAI base URL (defaults to $QWEN_ASR_VLLM_BASE_URL or http://localhost:8002/v1).")
    parser.add_argument("--language", type=str, default="Hindi", help="Target language (default: Hindi).")
    parser.add_argument("--mode", type=str, default="chat_completions", choices=["chat_completions", "audio_transcriptions"], help="API interaction mode (default: chat_completions).")
    
    parser.add_argument("--all-samples", action="store_true", help="Transcribe and evaluate the entire 611-sample benchmark dataset.")
    
    args = parser.parse_args()
    
    # Resolve base URL
    base_url = args.base_url
    if not base_url:
        base_url = os.environ.get("QWEN_ASR_VLLM_BASE_URL", "http://localhost:8002/v1")
        
    print("="*80)
    print(" QWEN3-ASR ENDPOINT DIAGNOSTIC & TEST UTILITY")
    print("="*80)
    print(f"Model ID:  {args.model}")
    print(f"Base URL:  {base_url}")
    print(f"Language:  {args.language}")
    print(f"Mode:      {args.mode}")
    print("-"*80)

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
        print(f"Evaluating {len(samples)} samples with Qwen vLLM...")
        
        predictions_log = []
        total_s, total_d, total_i, total_words = 0, 0, 0, 0
        
        for idx, sample in enumerate(samples):
            audio_data = sample["audio"]
            audio_arr = to_mono(audio_data["array"])
            audio_arr = resample_linear(audio_arr, audio_data["sampling_rate"], 16000)
            reference = sample.get("transcript") or sample.get("text") or ""
            
            try:
                raw_text, cleaned_text = transcribe_qwen_raw_and_cleaned(
                    audio_array=audio_arr,
                    sample_rate=16000,
                    base_url=base_url,
                    model=args.model,
                    language=args.language,
                    mode=args.mode
                )
            except Exception as e:
                print(f"\nSample {idx} failed: {e}")
                continue
                
            # Calculate WER
            clean_ref = normalize_asr_text(reference)
            clean_hyp = normalize_asr_text(cleaned_text)
            
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
                "hypothesis": cleaned_text,
                "raw_output": raw_text,
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
        
        pred_file = out_dir / f"predictions_qwen_only_{timestamp}.jsonl"
        with pred_file.open("w", encoding="utf-8") as f:
            for log_entry in predictions_log:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        final_wer = (total_s + total_d + total_i) / total_words if total_words else 0.0
        
        print("\n" + "="*80)
        print(" QWEN-ONLY FULL BENCHMARK RESULTS")
        print("="*80)
        print(f"Model ID:          {args.model}")
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
    print("Sending transcription request to vLLM server...")
    try:
        raw_text, cleaned_text = transcribe_qwen_raw_and_cleaned(
            audio_array=audio_arr,
            sample_rate=16000,
            base_url=base_url,
            model=args.model,
            language=args.language,
            mode=args.mode
        )
    except Exception as e:
        print(f"API Request Failed: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print(" RESULTS")
    print("="*80)
    print(f"Raw Output from Qwen API:\n{repr(raw_text)}")
    print("-"*80)
    print(f"Cleaned Transcript:\n{repr(cleaned_text)}")
    
    if reference:
        print("-"*80)
        print(f"Reference Text:\n{repr(reference)}")
        
        # Calculate sample WER
        clean_ref = normalize_asr_text(reference)
        clean_hyp = normalize_asr_text(cleaned_text)
        
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
