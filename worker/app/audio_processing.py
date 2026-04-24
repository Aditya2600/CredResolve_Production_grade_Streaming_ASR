import numpy as np
import logging
import torch
import io
import os
import audioop

log = logging.getLogger("worker.audio_processing")

class AudioPreprocessor:
    def __init__(self, vad_threshold=0.5):
        self.vad_threshold = vad_threshold
        self.vad_model = None
        self.utils = None
        self.rnnoise = None
        self._load_models()

    def _load_models(self):
        try:
            # Lazy load silero VAD
            self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                        model='silero_vad',
                                                        force_reload=False)
            log.info("Silero VAD model loaded successfully")
        except Exception as e:
            log.warning(f"Failed to load Silero VAD: {e}")

        try:
            # Check for rnnoise-wrapper or similar
            # In a real scenario, we'd use a C-wrapper for RNNoise
            # For this demo/impl, we'll prepare the hook
            from rnnoise_wrapper import RNNoise
            self.rnnoise = RNNoise()
            log.info("RNNoise denoiser loaded successfully")
        except ImportError:
            log.warning("RNNoise wrapper not found. Denoising will be skipped.")

    def get_rms(self, audio_data):
        """Calculate Root Mean Square (RMS) volume."""
        return np.sqrt(np.mean(audio_data.astype(np.float32)**2))

    def process(self, pcm_bytes, sample_rate, vad_enabled=True, denoise_enabled=True):
        if not pcm_bytes:
            return pcm_bytes

        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # 1. VAD Layer
        if vad_enabled and self.vad_model is not None:
            # Silero expects 16kHz
            if sample_rate != 16000:
                # Basic resample for VAD if needed (though ASR usually handles this)
                pass 
            
            # Convert to float32 for Silero
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32)
            
            # Get speech timestamps
            get_speech_timestamps = self.utils[0]
            speech_timestamps = get_speech_timestamps(audio_tensor, self.vad_model, sampling_rate=16000)
            
            if not speech_timestamps:
                return b"" # Silence

            # Max Volume Logic: If multiple segments, find the loudest one
            loudest_segment = None
            max_rms = -1
            
            for ts in speech_timestamps:
                segment = audio_int16[ts['start']:ts['end']]
                rms = self.get_rms(segment)
                if rms > max_rms:
                    max_rms = rms
                    loudest_segment = segment
            
            audio_int16 = loudest_segment

        # 2. Denoise Layer (RNNoise)
        if denoise_enabled and self.rnnoise is not None:
            try:
                # RNNoise REQUIRES 48kHz mono PCM, 16-bit
                target_sr = 48000
                frame_size = 480 # 10ms at 48kHz
                
                # High-speed Resample to 48kHz using audioop (C-backend)
                if sample_rate != target_sr:
                    pcm_48k, _ = audioop.ratecv(pcm_bytes, 2, 1, sample_rate, target_sr, None)
                else:
                    pcm_48k = pcm_bytes

                # Process in 10ms frames
                # Optimization: pre-convert to bytearray and slice
                cleaned_48k = bytearray()
                num_frames = len(pcm_48k) // (frame_size * 2)
                
                for i in range(num_frames):
                    start = i * frame_size * 2
                    end = start + frame_size * 2
                    frame_bytes = pcm_48k[start:end]
                    
                    # Process frame via C-wrapper
                    cleaned_48k.extend(self.rnnoise.process_frame(frame_bytes))
                
                # Resample back to original sample_rate for ASR
                if sample_rate != target_sr:
                    pcm_bytes, _ = audioop.ratecv(bytes(cleaned_48k), 2, 1, target_sr, sample_rate, None)
                else:
                    pcm_bytes = bytes(cleaned_48k)

            except Exception as e:
                log.error(f"RNNoise processing failed: {e}")

        return pcm_bytes
