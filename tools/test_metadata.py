#!/usr/bin/env python3
import asyncio
import json
import websockets
import sys

async def test_metadata():
    uri = "ws://localhost:8000/ws/stt"
    
    # Load real audio
    with open("recordings/call_recording_16k.wav", "rb") as f:
        # Skip header (44 bytes) to get raw PCM
        f.seek(44)
        noise = f.read(16000 * 5) # Read 5 seconds 
    
    async with websockets.connect(uri) as ws:
        # Start
        await ws.send(json.dumps({
            "type": "start",
            "api_key": "dev",
            "language": "hi",  # Explicit language
            "sample_rate": 16000
        }))
        
        # Wait for ready
        resp = json.loads(await ws.recv())
        if resp.get("type") != "ready":
            print(f"FAILED: Expected ready, got {resp}")
            sys.exit(1)
            
        print("Connected. Sending audio...")
        
        # Send audio
        chunk_size = 640
        for i in range(0, len(noise), chunk_size):
            await ws.send(noise[i:i+chunk_size])
            await asyncio.sleep(0.02)
            
        # Stop
        await ws.send(json.dumps({"type": "stop"}))
        
        # Read final
        meta_verified = False
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "final":
                print(f"Received Final: {msg}")
                if "language" in msg and "language_source" in msg:
                    meta_verified = True
                    print(f"SUCCESS: Metadata found! Lang={msg['language']} Source={msg['language_source']}")
                else:
                    print("FAILED: Missing metadata fields")
            
            if msg["type"] == "done":
                break
                
        if not meta_verified:
            print("FAILED: No final result received or metadata missing")
            sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(test_metadata())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
