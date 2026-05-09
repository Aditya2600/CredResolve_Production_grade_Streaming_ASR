import io
import wave
import base64

import torch
import numpy as np
import requests

API_URL = ""
API_KEY = ""
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz

# ── Silero VAD ──────────────────────────────────────────────────────
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)


def read_wav(path):
    """Read a WAV file and return a float32 torch tensor at SAMPLE_RATE."""
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "expected mono audio"
        assert wf.getframerate() == SAMPLE_RATE, f"expected {SAMPLE_RATE}Hz"
        raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(pcm)


def segment_speech(path, threshold=0.5, min_silence_chunks=8):
    vad_model.reset_states()
    audio = read_wav(path)
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


def pcm_to_wav_b64(pcm):
    int16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def transcribe_chunked(audio_path, language="hi"):
    for i, segment in enumerate(segment_speech(audio_path)):
        duration = len(segment) / SAMPLE_RATE
        print(f"[segment {i + 1}] {duration:.2f}s — sending to /predict ...")

        resp = requests.post(API_URL, json={"audio": pcm_to_wav_b64(segment), "language": language}, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()

        text = ""
        if isinstance(data, list) and data:
            text = data[0].get("text", "")
        elif isinstance(data, dict):
            text = data.get("text", "")

        print(f'  → "{text}"')


if __name__ == "__main__":
    transcribe_chunked("sample_audio.wav", language="hi")
