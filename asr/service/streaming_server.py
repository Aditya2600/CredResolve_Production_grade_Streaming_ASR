from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, Request
from fastapi.responses import JSONResponse

from asr.decoding.beam_with_lid_bias import Hypothesis, select_best_hypothesis
from asr.models.load_indicconformer import ASRModel, load_indicconformer, read_env_model_source
from asr.models.script_head import ScriptHead
from asr.models.tokenlid_head import TokenLIDHead
from asr.postprocess.normalize import normalize_text
from asr.postprocess.script_fix import fix_text
from asr.preprocess import TARGET_SAMPLE_RATE, VADConfig, ensure_16k_mono, pcm16_bytes_to_float32, split_by_vad


@dataclass
class DecodeOptions:
    decode_mode: str = "rnnt"
    use_lid_bias: bool = False
    lid_bias_alpha: float = 1.5


class StreamingASREngine:
    def __init__(self, model: ASRModel, options: DecodeOptions | None = None):
        self.model = model
        self.options = options or DecodeOptions()
        self.token_lid = TokenLIDHead()
        self.script_head = ScriptHead()
        self.vad_cfg = VADConfig()

    def _decode_segment(self, segment: np.ndarray) -> str:
        text = self.model.decode(segment.astype(np.float32), TARGET_SAMPLE_RATE, mode=self.options.decode_mode)
        return normalize_text(text)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        lang_posterior: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        audio16k = ensure_16k_mono(audio, sample_rate)
        segments = split_by_vad(audio16k, TARGET_SAMPLE_RATE, config=self.vad_cfg)
        if not segments:
            segments = [audio16k]

        raw_parts = [self._decode_segment(seg) for seg in segments if len(seg) > 0]
        raw_text = normalize_text(" ".join(raw_parts))

        # Baseline path: no LID bias.
        chosen_text = raw_text
        if self.options.use_lid_bias and lang_posterior:
            vocab = {
                "hi": {"मैं", "घर", "नहीं", "है", "हलो", "हेलो"},
                "te": {"నేను", "ఇంటి", "లేదు", "తెలుగు", "హలో"},
                "en": {"hello", "payment", "call"},
            }
            nbest = [Hypothesis(text=raw_text, score=0.0)]
            chosen = select_best_hypothesis(
                nbest,
                use_lid_bias=True,
                lang_posterior=lang_posterior,
                vocab_by_lang=vocab,
                alpha=self.options.lid_bias_alpha,
            )
            chosen_text = chosen.text

        tokens = [t for t in chosen_text.split() if t]
        token_lang = [pred.language for pred in self.token_lid.predict_tokens(tokens)]
        token_script = [pred.script for pred in self.script_head.predict_tokens(tokens)]
        final_text, events = fix_text(
            chosen_text,
            token_lang=token_lang,
            token_script=token_script,
            token_lid_head=self.token_lid,
            script_head=self.script_head,
        )

        return {
            "raw_text": raw_text,
            "text": final_text,
            "token_lang": token_lang,
            "token_script": token_script,
            "script_fix_count": sum(1 for e in events if e.action != "keep"),
        }


def _get_default_engine() -> StreamingASREngine:
    source = os.environ.get("ASR_MODEL_SOURCE", read_env_model_source())
    backend = os.environ.get("ASR_MODEL_BACKEND", "hf")
    decode_mode = os.environ.get("ASR_DECODE_MODE", "rnnt")

    if os.environ.get("ASR_USE_MOCK", "1").strip().lower() in {"1", "true", "yes"}:
        model = load_indicconformer("mock://indicconformer", backend="mock")
    else:
        model = load_indicconformer(source, backend=backend)

    opts = DecodeOptions(
        decode_mode=decode_mode,
        use_lid_bias=os.environ.get("ASR_USE_LID_BIAS", "0").strip().lower() in {"1", "true", "yes"},
        lid_bias_alpha=float(os.environ.get("ASR_LID_BIAS_ALPHA", "1.5")),
    )
    return StreamingASREngine(model=model, options=opts)


app = FastAPI(title="IndicConformer Streaming Service Stub")
ENGINE = _get_default_engine()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/transcribe")
async def transcribe_http(
    request: Request,
    x_sample_rate: int = Header(default=8000),
    x_mode: str = Header(default="final"),
) -> JSONResponse:
    _ = x_mode
    pcm = await request.body()
    audio = pcm16_bytes_to_float32(pcm)
    t0 = time.time()
    out = ENGINE.transcribe(audio, sample_rate=int(x_sample_rate))
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return JSONResponse(out)


@app.websocket("/v1/stream")
async def stream_ws(ws: WebSocket) -> None:
    await ws.accept()
    session_id = str(uuid.uuid4())
    sample_rate = 8000
    chunk_buf = bytearray()

    try:
        await ws.send_text(json.dumps({"type": "ready", "session_id": session_id}))
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "start":
                    sample_rate = int(data.get("sample_rate", 8000))
                    await ws.send_text(json.dumps({"type": "ack", "sample_rate": sample_rate}))
                    continue
                if data.get("type") == "stop":
                    audio = pcm16_bytes_to_float32(bytes(chunk_buf))
                    out = ENGINE.transcribe(audio, sample_rate=sample_rate)
                    await ws.send_text(json.dumps({"type": "final", **out}))
                    await ws.send_text(json.dumps({"type": "done"}))
                    break
                continue

            if "bytes" in msg and msg["bytes"]:
                chunk = msg["bytes"]
                chunk_buf.extend(chunk)
                # Emit partial on each chunk for debugging.
                audio = pcm16_bytes_to_float32(bytes(chunk_buf))
                out = ENGINE.transcribe(audio, sample_rate=sample_rate)
                await ws.send_text(json.dumps({"type": "partial", **out}))

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# gRPC service stub intentionally not implemented in MVP. Expose HTTP+WS first.
