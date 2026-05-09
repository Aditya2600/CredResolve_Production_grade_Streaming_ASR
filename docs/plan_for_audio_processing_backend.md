plan for Audio processing backend                                                                       
     
                                   
● Here's a plan for the audio-processing backend, grounded in what's there today
  (worker/app/audio_processing.py, plus the gateway hooks).
                                                                                          
  Current state                                                                                           
                                                                                                          
  - AudioPreprocessor does Silero VAD → segment selection (`concat` default, `loudest` legacy) → optional
  RNNoise denoise (48 kHz framed loop via audioop).                                                       
  - Wired into worker/app/main.py through the cached get_audio_preprocessor() singleton, with             
  main_v2.py also using the module-level helper.                                                          
  - Gateway plumbs two flags end-to-end: vad_enabled, denoise_enabled. The historical apm_enabled flag
  has been removed (see docs/audio/apm-decision.md); the gateway intentionally does not perform APM.
  See docs/audio/noise-cancellation-deep-dive.md for the current gateway VAD + worker RNNoise/Silero flow.
  - Unit tests cover segment selection and singleton behavior; real-model integration/load coverage is
  still missing.
                                                                                                          
  Gaps worth fixing

  1. (Resolved.) VAD segment concatenation is now the default; legacy `loudest` mode still exists and can
  drop legitimate speech in multi-segment utterances.                                                      
  2. (Resolved.) apm_enabled has been removed from the public schema; the gateway does not run APM.
  See docs/audio/apm-decision.md for the rationale.
  3. audioop is removed in Python 3.13. Resampling will break on upgrade. Replace with
  scipy.signal.resample_poly or soxr.                                                                     
  4. Denoise frame loop is pure Python. ~480-byte slices in a tight loop is slow; batch via numpy or call
  into a vectorized denoiser (DeepFilterNet / Demucs-lite) instead of RNNoise.                            
  5. torch.hub.load at import time needs network on first run. Vendor Silero weights (pip install 
  silero-vad) or pin to a local snapshot.                                                                 
  6. No streaming variant. Current process() operates on a complete utterance. Streaming WS path (gateway
  → worker chunks) currently bypasses any per-chunk APM/denoise.                                          
  7. Single instance is not thread-safe (Silero state is fine; RNNoise C wrapper typically isn't). main.py
   calls it via asyncio.to_thread — needs either per-thread instances or a lock.                          
                  
  Proposed phases                                                                                         
                  
  - P0 — Hygiene (small, no behavior change): keep get_audio_preprocessor as the cached singleton; keep
  segment concatenation as the default behind VAD_SELECT_MODE=loudest|concat; expand tests from unit
  coverage into real-model integration/load coverage.         
  - P1 — APM: **dropped.** The gateway no longer exposes or implements APM (see
  docs/audio/apm-decision.md). NS/AGC capabilities, if ever needed server-side, would land in the worker
  AudioPreprocessor next to denoise rather than as a new gateway-side native dependency.
  - P2 — Replace audioop: swap to soxr (fastest C resampler) before Py 3.13.                              
  - P3 — Streaming pipeline: introduce a per-session StreamingAudioProcessor holding VAD/denoise state    
  across chunks, used by the WS path.                                                                     
  - P4 — Denoiser upgrade (optional): evaluate DeepFilterNet vs RNNoise on Hindi/Indian-English samples;  
  only swap if WER improves.                                                                              
                  
  Main tradeoff                                                                                           
                  
  P0 + P2 are pure tech-debt wins and safe. P4 (DeepFilterNet) adds latency and a real native dependency
  — only worth doing if you have measurements showing the current VAD-only path is leaving WER on the
  table. I'd start with P0 and benchmark before committing to P4. P1 is closed.                               
                  
