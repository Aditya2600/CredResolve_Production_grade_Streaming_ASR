import argparse
import asyncio
import json
import wave
import websockets
from websockets.exceptions import ConnectionClosedOK

def wav_frames(path: str, frame_bytes: int):
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit("WAV must be 16kHz mono PCM16")
        while True:
            b = w.readframes(frame_bytes // 2)
            if not b:
                break
            if len(b) < frame_bytes:
                b += b"\x00" * (frame_bytes - len(b))
            yield b

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--call-id", default="c1")
    ap.add_argument("--frame-ms", type=int, default=20)
    ap.add_argument("--language", default="hi")
    args = ap.parse_args()

    frame_bytes = int(16000 * (args.frame_ms / 1000.0) * 2)

    async with websockets.connect(args.ws, max_size=20_000_000) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "api_key": args.api_key,
            "call_id": args.call_id,
            "language": args.language,
            "sample_rate": 16000,
            "encoding": "pcm_s16le",
            "frame_ms": args.frame_ms
        }))
        print("<-", await ws.recv())

        for frame in wav_frames(args.wav, frame_bytes):
            await ws.send(frame)
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    print("<-", msg)
            except asyncio.TimeoutError:
                pass

        await ws.send(json.dumps({"type": "stop"}))
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
