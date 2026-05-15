import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_gate")

print("Starting validation script...")
try:
    from worker.app.triton import TritonIndicASRWorker
    from tools.benchmarks.audio_bench import read_wav_pcm16_mono, wer
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)

def validate(triton_url, audio_dir, reference_json, decoder):
    log.info(f"Starting validation gate for decoder={decoder} against {triton_url}")
    
    # Load references
    with open(reference_json) as f:
        references = json.load(f)
    
    # Initialize worker
    print(f"Initializing Triton worker for {triton_url}...")
    worker = TritonIndicASRWorker(
        triton_url=triton_url,
        triton_model_name="indic_asr",
        triton_ctc_model_name="indic_asr_ctc",
        asr_asset_repo=os.environ.get("ASR_MODEL_NAME", "ai4bharat/indic-conformer-600m-multilingual"),
        hf_token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        supported_language_allowlist=["hi"],
        default_language="hi"
    )
    
    try:
        print("Loading worker...")
        worker.load()
        print("Worker loaded successfully.")
    except Exception as e:
        print(f"Failed to load Triton worker: {e}")
        return False, "TRITON_NOT_READY"

    failed_files = []
    total_wer = 0
    count = 0

    for filename, expected_text in references.items():
        audio_path = audio_dir / filename
        if not audio_path.exists():
            log.warning(f"Audio file {audio_path} not found, skipping")
            continue
            
        try:
            pcm, sr, duration = read_wav_pcm16_mono(audio_path)
            
            t0 = time.perf_counter()
            result = worker.transcribe_pcm16(
                pcm16le=pcm,
                sample_rate=sr,
                decoder=decoder,
                language="hi"
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            
            text = (result.text or "").strip()
            error_rate = wer(expected_text, text)
            
            log.info(f"File: {filename:35} | WER: {error_rate:.4f} | Latency: {latency_ms:6.1f}ms | Text: {text}")
            
            # CRITICAL CHECKS
            if not text:
                log.error(f"FAILURE: Empty transcript for {filename}")
                failed_files.append((filename, "EMPTY_TRANSCRIPT"))
                continue
            
            # Simple suspicious blank rate check: if WER is extremely high (e.g. > 80%) 
            # and the reference is not empty, something is likely wrong.
            if error_rate > 0.8:
                log.error(f"FAILURE: Suspiciously high WER ({error_rate:.4f}) for {filename}")
                failed_files.append((filename, "HIGH_WER"))
                continue

            total_wer += error_rate
            count += 1
            
        except Exception as e:
            log.error(f"Error processing {filename}: {e}")
            failed_files.append((filename, f"ERROR: {e}"))

    if failed_files:
        return False, failed_files
    
    avg_wer = total_wer / count if count > 0 else 0
    log.info(f"Validation PASSED. Average WER: {avg_wer:.4f}")
    return True, avg_wer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton-url", default="localhost:8001")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--decoder", choices=["rnnt", "ctc"], default="rnnt")
    args = parser.parse_args()
    
    success, detail = validate(args.triton_url, args.audio_dir, args.reference_json, args.decoder)
    if not success:
        log.error(f"DEPLOY GATE FAILED: {detail}")
        sys.exit(1)
    else:
        sys.exit(0)
