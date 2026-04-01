#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import sys
import wave
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import websockets


def update_ws_query(base_ws_url: str, params: dict[str, str]) -> str:
    parsed = urlparse(base_ws_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    return urlunparse(parsed._replace(query=urlencode(query)))


def pcm_frames(path: str, frame_bytes: int, expected_sample_rate: int):
    with wave.open(path, "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise SystemExit("WAV must be mono PCM16")
        if wav_file.getframerate() != expected_sample_rate:
            raise SystemExit(
                f"WAV sample rate {wav_file.getframerate()}Hz does not match expected {expected_sample_rate}Hz"
            )
        while True:
            chunk = wav_file.readframes(frame_bytes // 2)
            if not chunk:
                break
            if len(chunk) < frame_bytes:
                chunk += b"\x00" * (frame_bytes - len(chunk))
            yield chunk


async def test_metadata(args):
    uri = update_ws_query(
        args.ws,
        {
            "language-code": args.language_code,
            "model": args.model,
            "mode": args.mode,
            "sample_rate": str(args.sample_rate),
            "high_vad_sensitivity": "false",
            "vad_signals": str(args.vad_signals).lower(),
            "flush_signal": "true",
            "input_audio_codec": "pcm_s16le",
        },
    )

    async with websockets.connect(uri, additional_headers={"Api-Subscription-Key": args.api_key}) as ws:
        print("Connected. Sending JSON audio messages...")
        frame_bytes = int(args.sample_rate * 0.02 * 2)
        for chunk in pcm_frames(args.wav, frame_bytes, args.sample_rate):
            await ws.send(
                json.dumps(
                    {
                        "audio": {
                            "data": base64.b64encode(chunk).decode("ascii"),
                            "sample_rate": str(args.sample_rate),
                            "encoding": "pcm_s16le",
                        }
                    }
                )
            )
            await asyncio.sleep(0.02)

        await ws.send(json.dumps({"type": "flush"}))

        transcript_verified = False
        metrics_verified = False
        request_id_verified = False
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            except asyncio.TimeoutError:
                break

            if msg.get("type") == "error":
                print(f"FAILED: Received error payload {msg}")
                sys.exit(1)

            if msg.get("type") == "data":
                print(f"Received Data: {msg}")
                payload = msg.get("data", {})
                transcript_verified = isinstance(payload.get("transcript"), str)
                request_id_verified = bool(payload.get("request_id"))
                metrics = payload.get("metrics", {})
                metrics_verified = isinstance(metrics.get("audio_duration"), (int, float)) and isinstance(
                    metrics.get("processing_latency"), (int, float)
                )
                if transcript_verified and request_id_verified and metrics_verified:
                    print("SUCCESS: data payload includes request_id, transcript, and metrics")
                    return

            if msg.get("type") == "vad":
                print(f"Received VAD: {msg}")

        print(
            "FAILED: no valid data payload received "
            f"(transcript={transcript_verified} request_id={request_id_verified} metrics={metrics_verified})"
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Sarvam-like websocket data payloads")
    parser.add_argument("--ws", default="ws://localhost/ws/stt")
    parser.add_argument("--wav", default="sample_data/sample_16k_mono.wav")
    parser.add_argument("--api-key", default="dev")
    parser.add_argument("--language-code", default="hi")
    parser.add_argument("--model", default="credresolve:v1")
    parser.add_argument("--mode", default="transcribe")
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000])
    parser.add_argument("--vad-signals", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(test_metadata(parse_args()))
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
