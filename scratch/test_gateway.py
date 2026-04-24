import asyncio
import json
import soundfile as sf
import websockets
import base64
from pathlib import Path

async def test_transcribe(audio_path, ws_url="ws://localhost/ws/stt", lang="hi"):
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        # Simplistic resampling or just use 16k files
        pass
        
    async with websockets.connect(f"{ws_url}?language={lang}") as ws:
        # Handshake
        await ws.send(json.dumps({
            "config": {
                "encoding": "LINEAR16",
                "sample_rate_hertz": 16000,
                "language_code": lang
            }
        }))
        
        # Send audio
        with open(audio_path, "rb") as f:
            await ws.send(f.read())
            
        # Get response
        result = await ws.recv()
        print(f"Result: {result}")

if __name__ == "__main__":
    # Test with one of the segments already created
    sample_wav = "/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/akshay-docs/data/segments/Result_7__1759296238626_audio_file_77a8838d-7339-4521-abc6-6d0c3adbe539_turn_2.wav"
    if Path(sample_wav).exists():
        asyncio.run(test_transcribe(sample_wav))
    else:
        print("Sample WAV not found")
