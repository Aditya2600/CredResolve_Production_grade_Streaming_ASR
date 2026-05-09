import io
import wave

import torch
import numpy as np
import requests

API_URL = "https://http.ng8pfht5tu.ss-in.s9t.link/v1/predict"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiMWZhODMyNDAtNDlmMC00YTg3LWI5MjAtM2E2NzA4MTFhNzMxIiwiZXhwIjoxNzc4NjU2MTgyLCJvcmdfdXVpZCI6IjExMzlkM2NiLWE2ZTQtNDg2Mi04MTMzLWNhNmI4MzYwZGZjYSJ9.ECeR6Ko84G9zKnqR3uiYe_EP-SdxhmYvE9ndP1oe2ME"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
}
SAMPLE_RATE = 8000
CHUNK_SAMPLES = 256  # 32ms at 8kHz (512 for 16kHz)

# ── Silero VAD ──────────────────────────────────────────────────────
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)


def read_audio(path):
    """Read a .wav or .raw file and return a float32 torch tensor at SAMPLE_RATE."""
    if path.endswith(".raw"):
        pcm = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1, "expected mono audio"
            assert wf.getframerate() == SAMPLE_RATE, f"expected {SAMPLE_RATE}Hz"
            raw = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(pcm)


def segment_speech(path, threshold=0.5, min_silence_chunks=8):
    vad_model.reset_states()
    audio = read_audio(path)
    speech_buf = []
    is_speech = False
    silence_count = 0

    for i in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[i : i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = torch.nn.functional.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        prob = vad_model(chunk.unsqueeze(0), SAMPLE_RATE).item()

        if prob >= threshold:
            silence_count = 0
            is_speech = True
            speech_buf.append(chunk.numpy())
        elif is_speech:
            silence_count += 1
            speech_buf.append(chunk.numpy())
            if silence_count >= min_silence_chunks:
                yield np.concatenate(speech_buf)
                speech_buf.clear()
                is_speech = False
                silence_count = 0

    if speech_buf:
        yield np.concatenate(speech_buf)


def pcm_to_raw_bytes(pcm):
    """Convert float32 PCM to raw int16 bytes."""
    return (pcm * 32767).astype(np.int16).tobytes()


def transcribe_chunked(audio_path, language=None):
    for i, segment in enumerate(segment_speech(audio_path)):
        duration = len(segment) / SAMPLE_RATE
        lang_info = language if language else "auto-detect"
        print(f"[segment {i + 1}] {duration:.2f}s ({lang_info}) — sending to /v1/predict ...")

        raw_bytes = pcm_to_raw_bytes(segment)
        form_data = {"sample_rate": str(SAMPLE_RATE)}
        if language:
            form_data["language"] = language
        resp = requests.post(
            API_URL,
            files={"audio": ("chunk.raw", raw_bytes, "application/octet-stream")},
            data=form_data,
            headers=HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        text = ""
        if isinstance(data, list) and data:
            text = data[0].get("text", "")
        elif isinstance(data, dict):
            text = data.get("text", "")

        print(f'  → "{text}"')


if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_audio_8k.raw"
    language = sys.argv[2] if len(sys.argv) > 2 else None
    transcribe_chunked(audio_file, language=language)
