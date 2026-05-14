#!/usr/bin/env python3
"""
Simplismart Streaming ASR - Live Microphone Client

Self-contained: captures mic audio, runs Silero VAD locally,
and sends speech segments directly to the HTTP /v1/predict endpoint.
No bridge server needed.

pip install sounddevice numpy requests torch

Usage:
  python3 live_mic_test.py
  Speak into your mic -- transcriptions appear in real time.
  Press Ctrl+C to stop.
"""

import copy
import sys
import time
import threading

import numpy as np
import requests
import sounddevice as sd
import torch

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

API_URL  = "https://http.ng8pfht5tu.ss-in.s9t.link/v1/predict"
API_KEY  = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiMWZhODMyNDAtNDlmMC00YTg3LWI5MjAtM2E2NzA4MTFhNzMxIiwiZXhwIjoxNzc4NjU2MTgyLCJvcmdfdXVpZCI6IjExMzlkM2NiLWE2ZTQtNDg2Mi04MTMzLWNhNmI4MzYwZGZjYSJ9.ECeR6Ko84G9zKnqR3uiYe_EP-SdxhmYvE9ndP1oe2ME"
SAMPLE_RATE = 16000
CHUNK_MS    = 32
CHUNK_SIZE  = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 512 samples @ 16kHz

VAD_THRESHOLD      = 0.5
MIN_SILENCE_CHUNKS = 8      # ~256ms of silence ends a segment

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"


def log(tag, color, msg):
    ts = time.strftime("%H:%M:%S")
    print(f"{DIM}{ts}{RESET}  {color}{BOLD}[{tag}]{RESET}  {msg}", flush=True)


def log_info(msg):
    log("INFO", CYAN, msg)


def log_speech(msg):
    log("SPEECH", GREEN, msg)


def log_result(msg):
    log("RESULT", YELLOW, msg)


def log_error(msg):
    log("ERROR", RED, msg)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VAD + TRANSCRIPTION ENGINE                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MicASR:
    def __init__(self):
        self._vad = None
        self._speech_buf: list[np.ndarray] = []
        self._is_speech = False
        self._silence_count = 0
        self._total_samples = 0
        self._segment_start = 0
        self._segment_count = 0
        self._running = False

    def _load_vad(self):
        log_info("Loading Silero VAD model...")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        self._vad = copy.deepcopy(model)
        self._vad.reset_states()
        log_info("VAD model ready.")

    def _transcribe(self, pcm_float: np.ndarray) -> str:
        raw_bytes = (pcm_float * 32767).astype(np.int16).tobytes()
        form_data = {"sample_rate": str(SAMPLE_RATE)}
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        try:
            resp = requests.post(
                API_URL,
                files={"audio": ("chunk.raw", raw_bytes, "application/octet-stream")},
                data=form_data,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            log_error(f"HTTP error: {exc}")
            return ""
        if isinstance(data, list) and data:
            return data[0].get("text", "")
        if isinstance(data, dict):
            return data.get("text", "")
        return ""

    def _on_speech_end(self):
        if not self._speech_buf:
            return
        pcm = np.concatenate(self._speech_buf)
        start_s = self._segment_start / SAMPLE_RATE
        dur_s = len(pcm) / SAMPLE_RATE
        self._segment_count += 1
        seg_num = self._segment_count

        log_speech(f"Segment #{seg_num} detected  ({dur_s:.1f}s from {start_s:.1f}s)  — transcribing...")

        self._speech_buf.clear()
        self._is_speech = False
        self._silence_count = 0

        def do_transcribe():
            t0 = time.monotonic()
            text = self._transcribe(pcm)
            elapsed = (time.monotonic() - t0) * 1000
            if text.strip():
                log_result(f"#{seg_num}  [{elapsed:.0f}ms]  \"{text}\"")
            else:
                log_result(f"#{seg_num}  [{elapsed:.0f}ms]  (empty — silence or too short)")

        threading.Thread(target=do_transcribe, daemon=True).start()

    def _process_chunk(self, pcm_int16: np.ndarray):
        pcm_float = pcm_int16.astype(np.float32) / 32768.0
        if len(pcm_float) < CHUNK_SIZE:
            pcm_float = np.pad(pcm_float, (0, CHUNK_SIZE - len(pcm_float)))

        prob = self._vad(torch.from_numpy(pcm_float).unsqueeze(0), SAMPLE_RATE).item()

        if prob >= VAD_THRESHOLD:
            self._silence_count = 0
            if not self._is_speech:
                self._is_speech = True
                self._segment_start = self._total_samples
                log_speech("Voice activity detected — listening...")
            self._speech_buf.append(pcm_float)
        elif self._is_speech:
            self._silence_count += 1
            self._speech_buf.append(pcm_float)
            if self._silence_count >= MIN_SILENCE_CHUNKS:
                log_speech("Silence detected — processing segment.")
                self._on_speech_end()

        self._total_samples += CHUNK_SIZE

    def run(self):
        self._load_vad()
        self._running = True

        print()
        print(f"  {BOLD}{'=' * 52}{RESET}")
        print(f"  {BOLD}{CYAN}  Simplismart ASR — Live Microphone{RESET}")
        print(f"  {BOLD}{'=' * 52}{RESET}")
        print(f"    Endpoint : {API_URL}")
        print(f"    Rate     : {SAMPLE_RATE}Hz")
        print(f"    Chunk    : {CHUNK_MS}ms ({CHUNK_SIZE} samples)")
        print(f"    VAD thr  : {VAD_THRESHOLD}")
        print(f"  {BOLD}{'=' * 52}{RESET}")
        print(flush=True)

        log_info(f"Opening microphone at {SAMPLE_RATE}Hz...")

        def audio_callback(indata, frames, time_info, status):
            if status:
                log_error(f"Audio stream: {status}")
            self._process_chunk(indata[:, 0].copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=audio_callback,
            ):
                log_info("Microphone open. Listening for speech...")
                log_info(f"{BOLD}Speak now!{RESET}  (Ctrl+C to stop)\n")
                while self._running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            remaining = self._speech_buf
            if remaining:
                log_speech("Flushing final segment...")
                self._on_speech_end()
            time.sleep(1)  # let in-flight transcription threads finish
            print()
            log_info(f"Session ended. {self._segment_count} segment(s) processed.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    MicASR().run()
