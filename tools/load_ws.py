#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
import wave
import sys
import statistics
from contextlib import suppress
from datetime import datetime
try:
    import websockets
except ImportError:
    print("Error: 'websockets' module not found. Install it with: pip install websockets")
    sys.exit(1)

# Default configuration
DEFAULT_WS_URL = "ws://localhost:8000/ws/stt"
DEFAULT_WAV_PATH = "sample_data/sample_16k_mono.wav"

class LoadTestStats:
    def __init__(self):
        self.connected = 0
        self.completed_ok = 0
        self.errors = 0
        self.latencies = []
        self.error_types = {}
        self.start_timestampts = {}

    def add_error(self, err):
        self.errors += 1
        # Extract a succinct error type
        msg = str(err)
        if "Protocol Error" in msg:
            etype = "ProtocolError"
        elif "Server Error" in msg:
            etype = f"ServerError: {msg.split(': ')[-1]}"
        elif "Timeout" in msg:
            etype = "Timeout"
        elif "Connection Closed" in msg:
            code = msg.split(':')[1].strip().split(' ')[0]
            etype = f"ConnectionClosed: {code}"
        else:
            etype = type(err).__name__
            
        self.error_types[etype] = self.error_types.get(etype, 0) + 1

    def add_latency(self, latency):
        self.latencies.append(latency)

    def print_summary(self):
        print("\n--- Load Test Summary ---")
        # 'connected' counts successful connections. 'errors' handles connection failures + runtime failures.
        # Total attempts is roughly n (passed in args), but let's just show stats relative to what happened.
        print(f"Sessions Established: {self.connected}")
        print(f"Successful Completions: {self.completed_ok}")
        print(f"Failures: {self.errors}")
        
        if self.latencies:
            print(f"Avg Session Duration: {statistics.mean(self.latencies):.2f}s")
            print(f"P95 Session Duration: {statistics.quantiles(self.latencies, n=20)[18]:.2f}s") # 19th quantile is 95%
            print(f"Min/Max Duration: {min(self.latencies):.2f}s / {max(self.latencies):.2f}s")
        
        if self.errors > 0:
            print("\nError Breakdown:")
            for etype, count in self.error_types.items():
                print(f"  {etype}: {count}")

stats = LoadTestStats()


async def cleanup_stream_task(task: asyncio.Task | None):
    if task is None:
        return
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, websockets.exceptions.ConnectionClosed, TimeoutError):
        await task

async def run_client(client_id, args, audio_data):
    """
    Simulates a single ASR client:
    1. Connects
    2. Sends Start
    3. Streams Audio
    4. Expects final result
    """
    uri = args.ws
    call_id = f"loadtest-{client_id}-{int(time.time())}"
    
    # Ramp-up delay
    if args.ramp_ms > 0:
        delay = (client_id * args.ramp_ms) / 1000.0
        await asyncio.sleep(delay)

    start_time = time.time()
    
    stream_task: asyncio.Task | None = None
    try:
        async with websockets.connect(uri, max_size=None, close_timeout=5) as websocket:
            stats.connected += 1
            
            # 1. Send Start
            start_msg = {
                "type": "start",
                "api_key": args.api_key,
                "call_id": call_id,
                "sample_rate": 16000,
                "encoding": "pcm_s16le",
                "frame_ms": args.frame_ms,
                "language": args.language
            }
            await websocket.send(json.dumps(start_msg))
            
            # Wait for ready
            resp = await websocket.recv()
            resp_data = json.loads(resp)
            if resp_data.get("type") == "error":
                raise Exception(f"Server Error: {resp_data.get('code')}")
            if resp_data.get("type") != "ready":
                raise Exception(f"Protocol Error: Expected ready, got {resp}")

            # 2. Stream Audio
            chunk_size = int(16000 * (args.frame_ms / 1000.0) * 2) # 2 bytes per sample
            offset = 0
            
            # Send audio loop
            stream_task = asyncio.create_task(stream_audio(websocket, audio_data, chunk_size, args.frame_ms))
            
            # Read loop
            final_received = False
            try:
                while True:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=args.timeout_sec)
                    data = json.loads(msg)
                    
                    if data.get("type") == "final":
                        final_received = True

                    if data.get("type") == "error":
                        detail = data.get("detail", "")
                        trace = data.get("trace", "")
                        msg = f"Server Error: {data.get('code')} - {detail}"
                        if trace:
                            msg += f"\nREMOTE TRACE:\n{trace}"
                        raise Exception(msg)
                        
                    if data.get("type") == "done":
                        break
                        
            except asyncio.TimeoutError:
                raise Exception("Timeout waiting for response")
            except websockets.exceptions.ConnectionClosed as e:
                if e.code != 1000: # 1000 is normal closure
                     raise Exception(f"Connection Closed: {e.code} {e.reason}")
            
            await cleanup_stream_task(stream_task)
            stream_task = None

            if args.expect_final and not final_received:
                 raise Exception("Missing Final Result")

            duration = time.time() - start_time
            stats.add_latency(duration)
            stats.completed_ok += 1
            # print(f"Client {client_id}: OK ({duration:.2f}s)")

    except Exception as e:
        # Keep the original exception so add_error can classify by message/code.
        stats.add_error(e)
        print(f"Client {client_id} Error: {e}")
    finally:
        await cleanup_stream_task(stream_task)

async def stream_audio(ws, audio_data, chunk_size, frame_ms):
    total_len = len(audio_data)
    offset = 0
    sleep_time = frame_ms / 1000.0

    try:
        while offset < total_len:
            chunk = audio_data[offset:offset+chunk_size]
            await ws.send(chunk)
            offset += chunk_size
            await asyncio.sleep(sleep_time) # Real-time simulation

        # Send stop
        if ws.close_code is None:
            await ws.send(json.dumps({"type": "stop"}))
    except (websockets.exceptions.ConnectionClosed, TimeoutError):
        return


def load_wav(path, max_sec):
    try:
        with wave.open(path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
                raise ValueError("WAV must be 16kHz, Mono, PCM16")
            
            frames = wf.readframes(wf.getnframes())
            if max_sec:
                max_bytes = int(16000 * 2 * max_sec)
                frames = frames[:max_bytes]
            return frames
    except Exception as e:
        print(f"Error loading WAV: {e}")
        sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="WebSocket ASR Load Generator")
    parser.add_argument("--ws", default=DEFAULT_WS_URL, help="WebSocket URL")
    parser.add_argument("--wav", default=DEFAULT_WAV_PATH, help="Path to 16k mono wav")
    parser.add_argument("--n", type=int, default=10, help="Number of concurrent clients")
    parser.add_argument("--api-key", default="dev", help="API Key")
    parser.add_argument("--language", default="hi", help="Language code")
    parser.add_argument("--frame-ms", type=int, default=20, help="Frame size in ms")
    parser.add_argument("--ramp-ms", type=int, default=100, help="Ramp-up delay per client (ms)")
    parser.add_argument("--timeout-sec", type=int, default=20, help="Receive timeout")
    parser.add_argument("--max-audio-sec", type=float, default=30.0, help="Max audio duration to send")
    parser.add_argument("--expect-final", action="store_true", help="Fail if no final result received")

    args = parser.parse_args()

    print(f"Loading WAV: {args.wav}")
    audio_data = load_wav(args.wav, args.max_audio_sec)
    print(f"Audio Size: {len(audio_data)} bytes ({len(audio_data)/32000:.2f}s)")
    
    print(f"Starting {args.n} clients...")
    tasks = []
    for i in range(args.n):
        tasks.append(run_client(i, args, audio_data))
    
    start_global = time.time()
    await asyncio.gather(*tasks)
    total_time = time.time() - start_global
    
    print(f"\nTotal Test Time: {total_time:.2f}s")
    stats.print_summary()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest Cancelled")
