import argparse
import asyncio
import base64
import json
import wave
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import websockets
from websockets.exceptions import ConnectionClosedOK


def update_ws_query(base_ws_url: str, params: dict[str, str]) -> str:
    parsed = urlparse(base_ws_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    return urlunparse(parsed._replace(query=urlencode(query)))


def pcm_frames(path: str, frame_bytes: int):
    with wave.open(path, "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise SystemExit("WAV must be mono PCM16")
        sample_rate = wav_file.getframerate()
        while True:
            chunk = wav_file.readframes(frame_bytes // 2)
            if not chunk:
                break
            if len(chunk) < frame_bytes:
                chunk += b"\x00" * (frame_bytes - len(chunk))
            yield chunk, sample_rate


def pcm_l16_bytes(pcm_s16le: bytes) -> bytes:
    if len(pcm_s16le) % 2 != 0:
        raise SystemExit("PCM payload must contain an even number of bytes")
    swapped = bytearray(len(pcm_s16le))
    for index in range(0, len(pcm_s16le), 2):
        swapped[index] = pcm_s16le[index + 1]
        swapped[index + 1] = pcm_s16le[index]
    return bytes(swapped)


def prepare_audio_chunk(raw_audio: bytes, encoding: str) -> bytes:
    if encoding == "pcm_l16":
        raw_audio = pcm_l16_bytes(raw_audio)
    return raw_audio


def encode_audio_chunk(raw_audio: bytes, encoding: str) -> str:
    raw_audio = prepare_audio_chunk(raw_audio, encoding)
    return base64.b64encode(raw_audio).decode("ascii")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--language-code", required=True)
    parser.add_argument("--model", default="credresolve:v1")
    parser.add_argument("--mode", default="transcribe")
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000])
    parser.add_argument(
        "--input-audio-codec",
        default="pcm_s16le",
        choices=["wav", "pcm_s16le", "pcm_l16", "pcm_raw"],
    )
    parser.add_argument("--vad-signals", action="store_true")
    parser.add_argument("--high-vad-sensitivity", action="store_true")
    parser.add_argument("--flush-signal", action="store_true")
    parser.add_argument(
        "--json-base64",
        action="store_true",
        help="Send legacy JSON/base64 PCM frames instead of binary websocket audio frames",
    )
    args = parser.parse_args()

    binary_audio = args.input_audio_codec != "wav" and not args.json_base64

    ws_url = update_ws_query(
        args.ws,
        {
            "language-code": args.language_code,
            "model": args.model,
            "mode": args.mode,
            "sample_rate": str(args.sample_rate),
            "high_vad_sensitivity": str(args.high_vad_sensitivity).lower(),
            "vad_signals": str(args.vad_signals).lower(),
            "flush_signal": str(args.flush_signal).lower(),
            "input_audio_codec": args.input_audio_codec,
            "binary_audio": "1" if binary_audio else "0",
        },
    )
    auth_headers = {"Api-Subscription-Key": args.api_key}
    frame_bytes = int(args.sample_rate * 0.02 * 2)

    async with websockets.connect(ws_url, max_size=20_000_000, additional_headers=auth_headers) as ws:
        if args.input_audio_codec == "wav":
            with open(args.wav, "rb") as wav_file:
                wav_bytes = wav_file.read()
            await ws.send(
                json.dumps(
                    {
                        "audio": {
                            "data": base64.b64encode(wav_bytes).decode("ascii"),
                            "sample_rate": str(args.sample_rate),
                            "encoding": "wav",
                        }
                    }
                )
            )
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    print("<-", msg)
            except asyncio.TimeoutError:
                pass
        else:
            for frame, wav_sample_rate in pcm_frames(args.wav, frame_bytes):
                if wav_sample_rate != args.sample_rate:
                    raise SystemExit(
                        f"WAV sample rate {wav_sample_rate}Hz does not match --sample-rate {args.sample_rate}Hz"
                    )
                if binary_audio:
                    await ws.send(prepare_audio_chunk(frame, args.input_audio_codec))
                else:
                    await ws.send(
                        json.dumps(
                            {
                                "audio": {
                                    "data": encode_audio_chunk(frame, args.input_audio_codec),
                                    "sample_rate": str(args.sample_rate),
                                    "encoding": args.input_audio_codec,
                                }
                            }
                        )
                    )
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                        print("<-", msg)
                except asyncio.TimeoutError:
                    pass

        await ws.send(json.dumps({"type": "flush"}))
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2)
                print("<-", msg)
            except asyncio.TimeoutError:
                break
            except ConnectionClosedOK:
                break


if __name__ == "__main__":
    asyncio.run(main())
