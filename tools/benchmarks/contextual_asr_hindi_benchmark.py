#!/usr/bin/env python3
import os
import gc
import json
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
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

# Import datasets safely
try:
    from datasets import load_dataset
except ImportError:
    print("Error: HF 'datasets' package is not installed. Please run: pip install datasets")
    raise SystemExit(1)

# Import ASR text normalizer from repository
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tools.asr_text_normalizer import normalize_asr_text
    from tools.compute_wer import edit_distance, normalize as compute_wer_normalize
except ImportError:
    # Standard fallback if not run from the correct directory
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


def compute_sample_wer(ref: str, hyp: str) -> tuple[float, int, int, int, int]:
    # Normalize reference using Vaani Hindi normalizer
    clean_ref = normalize_asr_text(ref)
    clean_hyp = normalize_asr_text(hyp)

    ref_words = compute_wer_normalize(clean_ref)
    hyp_words = compute_wer_normalize(clean_hyp)

    s, d, i = edit_distance(ref_words, hyp_words)
    total_words = len(ref_words)

    wer = ((s + d + i) / total_words) if total_words else (0.0 if not hyp_words else 1.0)
    return wer, s, d, i, total_words


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_wav2vec2(samples: list[dict[str, Any]], device: torch.device) -> list[str]:
    print("\n>>> Loading Wav2Vec2-Hindi Model (theainerd/Wav2Vec2-large-xlsr-hindi)...", flush=True)
    from transformers import AutoProcessor, AutoModelForCTC

    processor = AutoProcessor.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")
    model = AutoModelForCTC.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi").to(device)
    model.eval()

    hypotheses = []
    for idx, sample in enumerate(samples):
        audio_data = sample["audio"]
        audio_arr = to_mono(audio_data["array"])
        audio_arr = resample_linear(audio_arr, audio_data["sampling_rate"], 16000)

        inputs = processor(audio_arr, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        hypotheses.append(transcription)
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            print(f"Processed {idx + 1}/{len(samples)} samples", flush=True)

    del model
    del processor
    free_memory()
    return hypotheses


def eval_audiox_north(samples: list[dict[str, Any]], device: torch.device) -> list[str]:
    print("\n>>> Loading audioX-north-v1 Model (jiviai/audioX-north-v1)...", flush=True)
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained("jiviai/audioX-north-v1")
    model = WhisperForConditionalGeneration.from_pretrained("jiviai/audioX-north-v1").to(device)
    model.eval()

    hypotheses = []
    for idx, sample in enumerate(samples):
        audio_data = sample["audio"]
        audio_arr = to_mono(audio_data["array"])
        audio_arr = resample_linear(audio_arr, audio_data["sampling_rate"], 16000)

        inputs = processor(audio_arr, sampling_rate=16000, return_tensors="pt").to(device)
        # Handle float16/bfloat16 precision if necessary, otherwise use float32
        if device.type == "cuda":
            inputs = inputs.to(dtype=torch.float16)
            model = model.half()
        
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        hypotheses.append(transcription)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            print(f"Processed {idx + 1}/{len(samples)} samples", flush=True)

    del model
    del processor
    free_memory()
    return hypotheses


def eval_voxtral_mini(samples: list[dict[str, Any]], device: torch.device) -> list[str]:
    print("\n>>> Loading Voxtral-Mini-4B Model (mistralai/Voxtral-Mini-4B-Realtime-2602)...", flush=True)
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers import VoxtralConfig, VoxtralEncoderConfig, LlamaConfig, VoxtralProcessor, VoxtralForConditionalGeneration

    # Register mappings dynamically to prevent KeyErrors due to HF model type naming mismatches
    CONFIG_MAPPING.register('voxtral_realtime', VoxtralConfig)
    CONFIG_MAPPING.register('voxtral_realtime_encoder', VoxtralEncoderConfig)
    CONFIG_MAPPING.register('voxtral_realtime_text', LlamaConfig)

    processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Mini-4B-Realtime-2602", trust_remote_code=True)
    
    # Voxtral requires large memory, using float16/bfloat16 and trust_remote_code
    model = VoxtralForConditionalGeneration.from_pretrained(
        "mistralai/Voxtral-Mini-4B-Realtime-2602",
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    hypotheses = []
    for idx, sample in enumerate(samples):
        audio_data = sample["audio"]
        audio_arr = to_mono(audio_data["array"])
        audio_arr = resample_linear(audio_arr, audio_data["sampling_rate"], 16000)

        inputs = processor(audio_arr, return_tensors="pt").to(model.device)
        if device.type == "cuda":
            inputs = inputs.to(dtype=torch.float16)

        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        hypotheses.append(transcription)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            print(f"Processed {idx + 1}/{len(samples)} samples", flush=True)

    del model
    del processor
    free_memory()
    return hypotheses


def clean_qwen_transcript(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    
    # Strip markdown fences if wrapping the whole text
    import re
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        
    # Strip xml-like tags like <asr_text>, </asr_text>, <asr_text> ...
    text = re.sub(r"<[^>]+>", "", text)
    
    # Strip language headers like "language Hindi" or "language hindi" or "language: Hindi"
    text = re.sub(r"(?i)\blanguage\s*:\s*[a-zA-Z0-9_]+", "", text)
    text = re.sub(r"(?i)\blanguage\s+[a-zA-Z0-9_]+", "", text)
    
    # Strip leading/trailing quotes (single, double, smart quotes)
    text = text.strip()
    quote_pairs = {
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’"
    }
    while len(text) >= 2 and text[0] in quote_pairs and text[-1] == quote_pairs[text[0]]:
        text = text[1:-1].strip()
        
    return text.strip()


def transcribe_qwen_vllm_openai(
    audio_array,
    sample_rate: int,
    base_url: str,
    model: str,
    language: str = "Hindi",
    mode: str = "chat_completions",
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI Python client is not installed in this environment. Please run: pip install openai")
        
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    
    # Convert dataset audio to temporary WAV using soundfile
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
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "audio_url",
                                "audio_url": {
                                    "url": audio_url
                                }
                            }
                        ]
                    }
                ]
            )
            transcript = response.choices[0].message.content
            return clean_qwen_transcript(transcript)
            
        elif mode == "audio_transcriptions":
            with open(tmp_wav_name, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language="hi" if language.lower() in ["hindi", "hi"] else None
                )
                transcript = response.text
                return clean_qwen_transcript(transcript)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    finally:
        if os.path.exists(tmp_wav_name):
            os.remove(tmp_wav_name)


def eval_qwen_vllm_openai(
    samples: list[dict[str, Any]],
    model_id: str,
    base_url: str,
    language: str = "Hindi",
    mode: str = "chat_completions"
) -> list[str]:
    # 1. Health check first!
    if not samples:
        return []
    
    first_sample = samples[0]
    audio_data = first_sample["audio"]
    audio_arr = to_mono(audio_data["array"])
    audio_arr = resample_linear(audio_arr, audio_data["sampling_rate"], 16000)
    
    name = "Qwen3-ASR-vLLM-0.6B" if "0.6B" in model_id else "Qwen3-ASR-vLLM-1.7B"
    try:
        test_transcript = transcribe_qwen_vllm_openai(
            audio_array=audio_arr,
            sample_rate=16000,
            base_url=base_url,
            model=model_id,
            language=language,
            mode=mode
        )
        print(f"Health check passed for model '{model_id}'. First transcription: '{test_transcript}'", flush=True)
    except Exception as e:
        raise Exception(f"{name} failed health check at {base_url}: {e}")

    # 2. Transcribe all samples
    print(f"Evaluating {len(samples)} samples with '{model_id}'...", flush=True)
    hypotheses = []
    for idx, sample in enumerate(samples):
        audio_data = sample["audio"]
        audio_arr = to_mono(audio_data["array"])
        audio_arr = resample_linear(audio_arr, audio_data["sampling_rate"], 16000)
        
        try:
            transcript = transcribe_qwen_vllm_openai(
                audio_array=audio_arr,
                sample_rate=16000,
                base_url=base_url,
                model=model_id,
                language=language,
                mode=mode
            )
        except Exception as e:
            raise Exception(f"{name} transcription failed: {e}")
            
        hypotheses.append(transcript)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            print(f"Processed {idx + 1}/{len(samples)} samples", flush=True)
            
    return hypotheses


def make_eval_qwen(name, model_id, base_url, language, mode):
    def eval_func(samples, device):
        print(f"\n>>> Loading {name} Model via vLLM endpoint ({model_id})...", flush=True)
        
        # Check endpoint model availability
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI Python client is not installed in this environment. Please run: pip install openai")
            
        try:
            client = OpenAI(base_url=base_url, api_key="EMPTY")
            models_list = client.models.list()
            available_ids = [m.id for m in models_list.data]
            is_available = (model_id in available_ids) or any(model_id in aid or aid in model_id for aid in available_ids)
            if not is_available:
                raise Exception(f"Model {model_id} is not available on the configured vLLM endpoint.")
        except Exception as e:
            raise Exception(f"{name} failed health check at {base_url}: {e}")
            
        return eval_qwen_vllm_openai(samples, model_id, base_url, language, mode)
    return eval_func


def eval_indicconformer(samples: list[dict[str, Any]], device: torch.device) -> list[str]:
    print("\n>>> Loading Local IndicConformer Model (/home/ubuntu/models/indicconformer/IndicConformer.nemo)...", flush=True)
    from nemo.collections.asr.models import ASRModel

    nemo_path = "/home/ubuntu/models/indicconformer/IndicConformer.nemo"
    if not Path(nemo_path).exists():
        print(f"Warning: Local file {nemo_path} does not exist. Skipping IndicConformer.")
        return [""] * len(samples)

    # Use cpu for NeMo if cuda is not used, or load map_location
    model = ASRModel.restore_from(nemo_path, map_location=device)
    model.eval()

    hypotheses = []
    # NeMo transcribe expects a list of file paths. We write audio arrays to temporary WAV files.
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_paths = []
        for idx, sample in enumerate(samples):
            audio_data = sample["audio"]
            audio_arr = to_mono(audio_data["array"])
            audio_arr = resample_linear(audio_arr, audio_data["sampling_rate"], 16000)

            tmp_wav = Path(tmpdir) / f"sample_{idx:04d}.wav"
            sf.write(str(tmp_wav), audio_arr, 16000, subtype="PCM_16")
            audio_paths.append(str(tmp_wav))

        # Perform batched NeMo transcription
        predictions = model.transcribe(audio_paths, batch_size=4, verbose=False, language_id="hi")
        
        # NeMo hybrid returns prediction tuples or lists of text. Let's extract clean predictions.
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        for pred in predictions:
            if hasattr(pred, "text"):
                hypotheses.append(str(pred.text))
            elif isinstance(pred, dict) and "text" in pred:
                hypotheses.append(str(pred["text"]))
            else:
                hypotheses.append(str(pred))

    del model
    free_memory()
    return hypotheses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load contextual ASR benchmark dataset
    print("Loading 'sarvamai/contextual_asr_benchmark' dataset (hi-IN) from Hugging Face...")
    dataset_dict = load_dataset("sarvamai/contextual_asr_benchmark", "hi-IN")
    
    # Identify splits
    splits = list(dataset_dict.keys())
    print(f"Dataset splits: {splits}")
    ds = dataset_dict[splits[0]]

    # All samples in hi-IN are Hindi
    hindi_samples = list(ds)

    print(f"Total samples in split: {len(ds)}")
    print(f"Hindi samples: {len(hindi_samples)}")

    if not hindi_samples:
        print("Error: No Hindi samples found in the dataset.")
        return

    # Predictions structure
    predictions_log = []

    eval_funcs = [
        ("Wav2Vec2-Hindi", eval_wav2vec2),
        ("audioX-north", eval_audiox_north),
        ("Voxtral-Mini", eval_voxtral_mini),
        ("IndicConformer", eval_indicconformer),
    ]

    # Environment-driven configuration for Qwen
    enable_qwen = os.environ.get("QWEN_ASR_ENABLE", "0") == "1"
    base_url = os.environ.get("QWEN_ASR_VLLM_BASE_URL", "http://localhost:8000/v1")
    qwen_models_env = os.environ.get("QWEN_ASR_MODELS", "Qwen/Qwen3-ASR-0.6B,Qwen/Qwen3-ASR-1.7B")
    language = os.environ.get("QWEN_ASR_LANGUAGE", "Hindi")
    mode = os.environ.get("QWEN_ASR_MODE", "chat_completions")

    print("Qwen vLLM enabled:", "true" if enable_qwen else "false")
    print("Qwen vLLM base URL:", base_url)
    print("Qwen requested models:", qwen_models_env)
    print("Qwen mode:", mode)
    print("Qwen language:", language)
    print(flush=True)

    requested_qwen_models = [m.strip() for m in qwen_models_env.split(",") if m.strip()]

    QWEN_MODELS_MAP = {
        "Qwen3-ASR-vLLM-0.6B": "Qwen/Qwen3-ASR-0.6B",
        "Qwen3-ASR-vLLM-1.7B": "Qwen/Qwen3-ASR-1.7B"
    }

    model_statuses = {}
    model_hypotheses = {}
    
    # Setup Qwen models dynamically to eval_funcs if enabled
    if enable_qwen:
        for name, model_id in QWEN_MODELS_MAP.items():
            if model_id in requested_qwen_models:
                eval_funcs.append((name, make_eval_qwen(name, model_id, base_url, language, mode)))
            else:
                model_statuses[name] = {
                    "status": "failed",
                    "error": f"Model {model_id} not requested in QWEN_ASR_MODELS."
                }
    else:
        for name, model_id in QWEN_MODELS_MAP.items():
            model_statuses[name] = {
                "status": "failed",
                "error": "Qwen3-ASR vLLM endpoint disabled. Set QWEN_ASR_ENABLE=1 and run qwen-asr-serve/vLLM server."
            }

    # Evaluate standard models and registered Qwen models
    for name, func in eval_funcs:
        try:
            hyps = func(hindi_samples, device)
            model_hypotheses[name] = hyps
            model_statuses[name] = {"status": "success"}
        except Exception as e:
            error_msg = str(e)
            print(f"Skipping {name} because model loading failed: {error_msg}")
            model_statuses[name] = {"status": "failed", "error": error_msg}

    # Calculate WERs
    results_summary = {}
    for name, status_info in model_statuses.items():
        if status_info["status"] == "success":
            results_summary[name] = {
                "status": "success",
                "total_s": 0,
                "total_d": 0,
                "total_i": 0,
                "total_words": 0,
                "total_wer": 0.0,
                "sentences_evaluated": 0,
            }
        else:
            results_summary[name] = {
                "status": "failed",
                "error": status_info["error"]
            }

    for idx, sample in enumerate(hindi_samples):
        ref = sample.get("transcript") or sample.get("text") or ""
        
        sample_log = {
            "sample_index": idx,
            "reference": ref,
            "predictions": {}
        }

        for name, status_info in model_statuses.items():
            if status_info["status"] == "success":
                hyps = model_hypotheses[name]
                hyp = hyps[idx]
                wer, s, d, i, words = compute_sample_wer(ref, hyp)

                results_summary[name]["total_s"] += s
                results_summary[name]["total_d"] += d
                results_summary[name]["total_i"] += i
                results_summary[name]["total_words"] += words
                results_summary[name]["sentences_evaluated"] += 1

                sample_log["predictions"][name] = {
                    "hypothesis": hyp,
                    "wer": round(wer, 4),
                    "words": words,
                    "substitutions": s,
                    "deletions": d,
                    "insertions": i
                }

        predictions_log.append(sample_log)

    # Compute final aggregate WER percentages
    for name, stats in results_summary.items():
        if stats["status"] == "success":
            w = stats["total_words"]
            if w > 0:
                stats["total_wer"] = (stats["total_s"] + stats["total_d"] + stats["total_i"]) / w
            else:
                stats["total_wer"] = 0.0

    # Print stats and WER to console
    print("\n" + "="*80)
    print("Aggregate Performance Summary:")
    print("="*80)
    print(f"{'Model Name':<25} | {'Evaluated':<10} | {'Ref Words':<10} | {'Sub':<5} | {'Del':<5} | {'Ins':<5} | {'WER (%)':<10}")
    print("-"*80)
    successful_results = {k: v for k, v in results_summary.items() if v["status"] == "success"}
    sorted_results = sorted(successful_results.items(), key=lambda item: item[1]["total_wer"])
    for name, stats in sorted_results:
        wer_pct = f"{stats['total_wer'] * 100:.2f}%"
        print(f"{name:<25} | {stats['sentences_evaluated']:<10} | {stats['total_words']:<10} | {stats['total_s']:<5} | {stats['total_d']:<5} | {stats['total_i']:<5} | {wer_pct:<10}")
    
    failed_results = {k: v for k, v in results_summary.items() if v["status"] == "failed"}
    if failed_results:
        print("-"*80)
        print("Failed / Skipped Models:")
        for name, stats in failed_results.items():
            print(f"- {name}: {stats['error']}")
    print("="*80 + "\n")

    # Write per-sample predictions to predictions_{timestamp}.jsonl
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts/benchmarks/contextual_asr_hindi")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pred_file = out_dir / f"predictions_{timestamp}.jsonl"
    with pred_file.open("w", encoding="utf-8") as f:
        for log_entry in predictions_log:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print(f"\nWrote per-sample details to {pred_file}")

    # Generate a gorgeous comparison report
    report_file = out_dir / f"comparison_report_{timestamp}.md"
    
    report_md = []
    report_md.append("# ASR Models Hindi Benchmark Comparison Report\n")
    report_md.append(f"Evaluated on **{len(hindi_samples)}** Hindi utterances from the `sarvamai/contextual_asr_benchmark` dataset.\n")
    report_md.append("> [!NOTE]\n> Qwen3-ASR models are evaluated through an external vLLM/OpenAI-compatible endpoint, not through local Transformers loading.\n")
    
    # In the aggregate WER table, include only models that were actually evaluated successfully.
    report_md.append("## Aggregate Performance Summary\n")
    report_md.append("| Model Name | Total Evaluated | Reference Words | Substitutions | Deletions | Insertions | WER (%) |")
    report_md.append("| --- | --- | --- | --- | --- | --- | --- |")

    # Sort successful models by performance (WER ascending)
    successful_results = {k: v for k, v in results_summary.items() if v["status"] == "success"}
    sorted_results = sorted(successful_results.items(), key=lambda item: item[1]["total_wer"])
    for name, stats in sorted_results:
        wer_pct = f"{stats['total_wer'] * 100:.2f}%"
        report_md.append(
            f"| **{name}** | {stats['sentences_evaluated']} | {stats['total_words']} | "
            f"{stats['total_s']} | {stats['total_d']} | {stats['total_i']} | **{wer_pct}** |"
        )

    # In comparison_report.md, show failed models in a separate section like:
    # ## Failed / Skipped Models
    # with columns:
    # * Model Name
    # * Status
    # * Error
    failed_results = {k: v for k, v in results_summary.items() if v["status"] == "failed"}
    if failed_results:
        report_md.append("\n## Failed / Skipped Models\n")
        report_md.append("| Model Name | Status | Error |")
        report_md.append("| --- | --- | --- |")
        for name, stats in failed_results.items():
            clean_error = stats["error"].replace("\n", " ").replace("|", "\\|")
            report_md.append(f"| **{name}** | failed | {clean_error} |")

    report_md.append("\n## Selected Transcription Examples\n")
    
    # Show 5 random samples as transcription comparisons
    show_samples = min(5, len(predictions_log))
    for i in range(show_samples):
        log_entry = predictions_log[i]
        report_md.append(f"### Example {i + 1} (Index {log_entry['sample_index']})")
        report_md.append(f"- **Reference**: `{log_entry['reference']}`")
        for name, pred in log_entry["predictions"].items():
            report_md.append(f"  - **{name}** (WER: {pred['wer'] * 100:.1f}%): `{pred['hypothesis']}`")
        report_md.append("")

    report_file.write_text("\n".join(report_md) + "\n", encoding="utf-8")
    print(f"Generated comparison report at {report_file}")


if __name__ == "__main__":
    main()
