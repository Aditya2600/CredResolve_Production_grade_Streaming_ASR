#!/usr/bin/env python3
"""
WebSocket-to-HTTP Bridge Server for Streaming ASR.

Exposes a Simplismart-API-compatible WebSocket endpoint. Clients connect via WebSocket,
stream audio, and receive transcription results. Internally uses Silero VAD
and forwards speech segments to the Simplismart-API /v1/predict HTTP endpoint.

pip install numpy requests torch torchaudio websockets packaging onnxruntime

Then just run:   python "stt_test_client_ws.py"
"""

import asyncio
import copy
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
import websockets

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit these three values and run the script             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

API_URL  = "https://http.ng8pfht5tu.ss-in.s9t.link/v1/predict"
API_KEY  = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiMWZhODMyNDAtNDlmMC00YTg3LWI5MjAtM2E2NzA4MTFhNzMxIiwiZXhwIjoxNzc4NjU2MTgyLCJvcmdfdXVpZCI6IjExMzlkM2NiLWE2ZTQtNDg2Mi04MTMzLWNhNmI4MzYwZGZjYSJ9.ECeR6Ko84G9zKnqR3uiYe_EP-SdxhmYvE9ndP1oe2ME"
WS_PORT  = 8765

# ── Advanced (usually no need to change) ─────────────────────────────────
WS_HOST            = "0.0.0.0"
VAD_THRESHOLD      = 0.5
MIN_SILENCE_CHUNKS = 8

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ws-bridge")

# ── Load Silero VAD once at startup ──────────────────────────────────────
logger.info("Loading Silero VAD model...")
_vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
logger.info("Silero VAD loaded.")

_thread_pool = ThreadPoolExecutor(max_workers=64)


# ── Audio Pipeline (one per WebSocket connection) ────────────────────────

class AudioPipeline:
    """Per-connection VAD segmentation + HTTP transcription."""

    def __init__(self, sample_rate, language, enable_partials, partial_interval_s):
        self.sample_rate = sample_rate
        self.language = language
        self.enable_partials = enable_partials
        self.partial_interval_s = partial_interval_s

        self._vad = copy.deepcopy(_vad_model)
        self._vad.reset_states()

        self.chunk_samples = 512 if sample_rate >= 16000 else 256
        self._bytes_per_chunk = self.chunk_samples * 2

        self._byte_buffer = bytearray()
        self._speech_buf: list[np.ndarray] = []
        self._is_speech = False
        self._silence_count = 0
        self._samples_processed = 0
        self._speech_start_sample = 0
        self._transcription_num = 0
        self._last_partial_time = time.monotonic()

    def process_bytes(self, data: bytes) -> list[tuple]:
        self._byte_buffer.extend(data)
        events: list[tuple] = []
        while len(self._byte_buffer) >= self._bytes_per_chunk:
            chunk_bytes = bytes(self._byte_buffer[: self._bytes_per_chunk])
            del self._byte_buffer[: self._bytes_per_chunk]
            events.extend(self._process_vad_chunk(chunk_bytes))
        return events

    def flush(self) -> tuple | None:
        if self._speech_buf:
            pcm = np.concatenate(self._speech_buf)
            start = self._speech_start_sample
            self._speech_buf.clear()
            self._is_speech = False
            self._silence_count = 0
            return ("final", pcm, start)
        return None

    def transcribe(self, pcm_float: np.ndarray) -> str:
        raw_bytes = (pcm_float * 32767).astype(np.int16).tobytes()
        form_data = {"sample_rate": str(self.sample_rate)}
        if self.language:
            form_data["audio_language"] = self.language
        try:
            resp = requests.post(
                API_URL,
                files={"audio": ("chunk.raw", raw_bytes, "application/octet-stream")},
                data=form_data,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("Transcription HTTP error: %s", exc)
            return ""
        if isinstance(data, list) and data:
            return data[0].get("text", "")
        if isinstance(data, dict):
            return data.get("text", "")
        return ""

    def next_transcription_num(self) -> int:
        num = self._transcription_num
        self._transcription_num += 1
        return num

    def _process_vad_chunk(self, chunk_bytes: bytes) -> list[tuple]:
        events: list[tuple] = []
        pcm_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0

        if len(pcm_float) < self.chunk_samples:
            pcm_float = np.pad(pcm_float, (0, self.chunk_samples - len(pcm_float)))

        prob = self._vad(torch.from_numpy(pcm_float).unsqueeze(0), self.sample_rate).item()

        if prob >= VAD_THRESHOLD:
            self._silence_count = 0
            if not self._is_speech:
                self._is_speech = True
                self._speech_start_sample = self._samples_processed
                self._last_partial_time = time.monotonic()
            self._speech_buf.append(pcm_float)
        elif self._is_speech:
            self._silence_count += 1
            self._speech_buf.append(pcm_float)
            if self._silence_count >= MIN_SILENCE_CHUNKS:
                events.append(("final", np.concatenate(self._speech_buf), self._speech_start_sample))
                self._speech_buf.clear()
                self._is_speech = False
                self._silence_count = 0

        self._samples_processed += self.chunk_samples

        if (
            self.enable_partials
            and self._is_speech
            and self._speech_buf
            and (time.monotonic() - self._last_partial_time) >= self.partial_interval_s
        ):
            events.append(("partial", np.concatenate(self._speech_buf), self._speech_start_sample))
            self._last_partial_time = time.monotonic()

        return events


# ── WebSocket handler ────────────────────────────────────────────────────

async def handle_client(websocket):
    remote = websocket.remote_address
    conn_id = f"{remote[0]}:{remote[1]}"
    logger.info("[%s] Client connected", conn_id)
    loop = asyncio.get_running_loop()

    try:
        raw_metadata = await asyncio.wait_for(websocket.recv(), timeout=30)
        metadata = json.loads(raw_metadata)
        logger.info("[%s] Metadata: %s", conn_id, json.dumps(metadata))

        wp = metadata.get("whisper_params", {})
        sp = metadata.get("streaming_params", {})

        pipeline = AudioPipeline(
            sample_rate=sp.get("sample_rate", 16000),
            language=wp.get("audio_language"),
            enable_partials=sp.get("enable_partial_transcripts", False),
            partial_interval_s=sp.get("partial_transcript_interval_s", 0.5),
        )
        sample_rate = sp.get("sample_rate", 16000)

        while True:
            message = await websocket.recv()

            if isinstance(message, bytes):
                for kind, pcm, start_sample in pipeline.process_bytes(message):
                    is_final = kind == "final"
                    start_t = start_sample / sample_rate
                    end_t = start_t + len(pcm) / sample_rate
                    text = await loop.run_in_executor(_thread_pool, pipeline.transcribe, pcm)
                    tx_num = pipeline.next_transcription_num() if is_final else 0
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "segments": [{"start_time": round(start_t, 3), "end_time": round(end_t, 3), "text": text}],
                        "is_final": is_final,
                        "transcription_num": tx_num,
                    }))
                    logger.info("[%s] %s [%.2fs-%.2fs]: %s", conn_id, "FINAL" if is_final else "partial", start_t, end_t, text[:80])

            elif isinstance(message, str):
                text_msg = json.loads(message)

                if text_msg.get("type") == "end_audio":
                    trace_id = text_msg.get("trace_id")
                    logger.info("[%s] end_audio (trace=%s)", conn_id, trace_id)

                    ack = {"type": "end_audio", "body": {"status": "acknowledged"}}
                    if trace_id:
                        ack["trace_id"] = trace_id
                    await websocket.send(json.dumps(ack))

                    remaining = pipeline.flush()
                    if remaining:
                        _, pcm, start_sample = remaining
                        start_t = start_sample / sample_rate
                        end_t = start_t + len(pcm) / sample_rate
                        text = await loop.run_in_executor(_thread_pool, pipeline.transcribe, pcm)
                        await websocket.send(json.dumps({
                            "type": "transcription",
                            "segments": [{"start_time": round(start_t, 3), "end_time": round(end_t, 3), "text": text}],
                            "is_final": True,
                            "transcription_num": pipeline.next_transcription_num(),
                        }))
                        logger.info("[%s] FINAL (flush) [%.2fs-%.2fs]: %s", conn_id, start_t, end_t, text[:80])

                    await websocket.send(json.dumps({"type": "end_audio", "body": {"status": "finished"}}))
                    logger.info("[%s] Done.", conn_id)
                    break

                elif text_msg.get("type") == "health_check":
                    await websocket.send(json.dumps({
                        "type": "health_check",
                        "trace_id": text_msg.get("trace_id"),
                        "timestamp": time.time(),
                        "body": {"healthy": True},
                    }))

    except websockets.exceptions.ConnectionClosed as e:
        logger.warning("[%s] Connection closed: %s", conn_id, e)
    except Exception:
        logger.exception("[%s] Error", conn_id)
        try:
            await websocket.send(json.dumps({
                "type": "error",
                "body": {"error_type": "BridgeError", "message": "Internal server error", "recoverable": False},
            }))
        except Exception:
            pass
    finally:
        logger.info("[%s] Disconnected", conn_id)


# ── Start server ─────────────────────────────────────────────────────────

async def _run():
    async with websockets.serve(handle_client, WS_HOST, WS_PORT, max_size=10 * 1024 * 1024):
        logger.info("Bridge listening on ws://%s:%d  ->  %s", WS_HOST, WS_PORT, API_URL)
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(_run())