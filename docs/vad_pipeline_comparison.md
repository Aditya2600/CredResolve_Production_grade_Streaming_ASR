# VAD Pipeline: Test Script vs Production

Yeh document `tests/test_client_vad_streaming.py` aur production gateway ke VAD pipeline ka comparison hai — kaise kaam karte hain, kahan differ karte hain, aur kyon.

---

## Test Script Flow (`tests/test_client_vad_streaming.py`)

```
sample_audio.wav
   ↓
WAV file read karo (must be mono, 16 kHz)
   ↓
32 ms / 512-sample chunks mein todo
   ↓
Silero VAD (torch.hub, neural model) har chunk ka speech probability nikalta hai
   ↓
prob >= 0.5 → speech chunk, speech buffer mein add karo
prob < 0.5  → silence counter badhao, phir bhi buffer mein add karo
   ↓
8 consecutive silent chunks (~256 ms) → segment close karo, yield karo
   ↓
Segment ko base64-encoded WAV mein pack karo
   ↓
HTTP POST /predict pe bhejo (ek segment = ek blocking request)
   ↓
Response parse karo → text print karo
(next segment ke liye repeat)
```

**Engine:** Silero VAD (neural, PyTorch)  
**Transport:** HTTP batch — poora segment ban'ne ke baad ek POST  
**Output latency:** Segment khatam hone ke baad hi transcription milti hai

---

## Production Pipeline Flow (Gateway → Triton)

```
Client (browser / phone) raw PCM 16 kHz mono stream bhejta hai via WebSocket
   ↓
Gateway WebSocket handler frames receive karta hai
   ↓
20 ms frames (320 samples = 640 bytes) mein slice karo
   ↓
webrtcvad (mode=3, most aggressive) har frame ko voiced / unvoiced label karta hai
   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ VADGateStateMachine  (gateway/app/vad_gate.py)                     │
│                                                                     │
│  CLOSED                                                             │
│    last 3 frames mein 2+ voiced → OPEN                              │
│    ← single blip se trigger nahi hota (noise robustness)           │
│                                                                     │
│  OPEN                                                               │
│    last 5 frames mein 4+ unvoiced → HANGOVER                       │
│                                                                     │
│  HANGOVER  (200 ms grace window)                                   │
│    speech wapas aayi → OPEN (same utterance continue)              │
│    200 ms beet gaye → CLOSED (utterance finalize)                  │
└─────────────────────────────────────────────────────────────────────┘
   ↓
VADSegmenter  (gateway/app/vad.py)  buffer maintain karta hai:
   - speech_start → buffering chalu
   - silence frames bhi buffer mein (word endings na katein)
   - 320 ms continuous silence (end_silence_ms) → speech_end, segment ready
   - trailing silence 120 ms (keep_silence_ms) tak trim hota hai
   - utterance 12 sec se lambi → forced "max_utt" cut (GPU waste avoid)
   ↓
Ready segment gRPC se Triton ASR worker ko bheja jata hai
   ↓
Triton partial + final transcripts WebSocket pe stream karta hai client ko
(jab bhi bol rahe ho, partial transcripts aate rehte hain)
```

**Engine:** webrtcvad (DSP, lightweight, no GPU)  
**Transport:** WebSocket streaming + gRPC to Triton  
**Output latency:** Partial transcripts as-you-speak, final on utterance close

---

## Side-by-Side Comparison

| Aspect | Test Script | Production |
|---|---|---|
| **VAD engine** | Silero VAD (neural, PyTorch) | webrtcvad (DSP) |
| **Frame size** | 32 ms (512 samples) | 20 ms (320 samples) |
| **Speech open rule** | Single chunk `prob ≥ 0.5` | 2 voiced frames out of last 3 |
| **Speech close rule** | 8 silent chunks (~256 ms) | 4 unvoiced out of 5 frames + 200 ms hangover + 320 ms end-silence |
| **Hangover / resume** | Nahi hai | 200 ms window — short pause utterance nahi todta |
| **Max utterance cap** | Nahi (infinite buffer possible) | 12 sec pe forced cut |
| **Trailing silence** | Poora include hota hai | 120 ms tak trim |
| **Transport** | HTTP POST per segment, base64 WAV | WebSocket stream → gRPC raw PCM |
| **ASR endpoint** | `/predict` (legacy batch HTTP) | Triton gRPC streaming |
| **Transcription style** | Batch — segment ke baad ek response | Streaming partials while speaking |
| **Dependencies** | torch, torchaudio, requests | webrtcvad only (no GPU needed) |

---

## Key Differences Explained

### 1. VAD Engine Mismatch
Test script Silero use karta hai (neural model, GPU/CPU). Production webrtcvad use karta hai (DSP, microsecond latency). Dono same audio pe alag decisions lenge — koi bhi tuning (threshold, silence counts) jo test script se derive ho, production pe directly apply nahi hogi.

### 2. Hysteresis — Noise Robustness
Test script ka simple threshold approach har `prob ≥ 0.5` blip pe speech start maar deta hai. Production ka `VADGateStateMachine` require karta hai ki **window mein consistently voiced frames hon** (2-of-3 to open, 4-of-5 to close). Isse noisy phone audio pe test script over-segment karega jab production nahi karega.

### 3. Hangover Window
Production mein 200 ms ka hangover state hai. Agar speaker ek pal ke liye ruke aur fir bole, toh ek hi utterance continue rahegi — do alag segments nahi banenge. Test script mein yeh nahi hai: 256 ms silence ke baad immediately segment close ho jaata hai, chahe speaker waqfon mein bol raha ho.

### 4. Streaming vs Batch
Yahi sabse bada practical antar hai. Test script ek **synchronous loop** hai: segment bano, POST karo, response lo, print karo, repeat. User ko transcription tab milti hai jab bol chuka ho. Production **low-latency streaming** hai: jab bhi Triton ke paas transcribe karne layak frames hain, partial text client ko push hota rehta hai.

### 5. Max Utterance Guard
Production `vad.py:67-70` mein 12 sec ki hard cap hai. Agar koi lambi unbroken speech de toh segment force-close hota hai — GPU ko infinite buffer se protect karta hai. Test script mein yeh guard nahi hai.

---

## Test Script Ki Known Defects

Yeh issues production comparison se independent hain:

- `API_URL = ""`, `API_KEY = ""` — script runtime pe fail karega
- `tests/` mein hai lekin koi pytest assertions nahi hain; yeh ek `__main__` script hai
- Hard-coded `"sample_audio.wav"` path
- `torch.hub.load(...)` module import pe — slow, offline pe fail
- Hard assert on mono + 16 kHz (no resampling, limited test fixtures)
- Per-segment sequential requests, koi retry nahi

---

## Related Files

| File | Role |
|---|---|
| [gateway/app/vad.py](../gateway/app/vad.py) | `VADSegmenter` — frame buffer, speech_end, max_utt logic |
| [gateway/app/vad_gate.py](../gateway/app/vad_gate.py) | `VADGateStateMachine` — CLOSED/OPEN/HANGOVER states |
| [gateway/app/pipeline.py](../gateway/app/pipeline.py) | Gateway pipeline joining VAD + worker client |
| [tests/test_client_vad_streaming.py](../tests/test_client_vad_streaming.py) | Silero-based offline batch client (not a test of production VAD) |
