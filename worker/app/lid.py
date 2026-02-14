import logging
import torch
import torchaudio
import numpy as np
from speechbrain.inference.classifiers import EncoderClassifier

log = logging.getLogger("worker.lid")

class SpeechBrainLID:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = None
        # Using a small, fast model for LID
        self.source = "speechbrain/lang-id-voxlingua107-ecapa"
        self.savedir = "models/lid_model"

    def load_model(self):
        log.info("Loading LID model from %s...", self.source)
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source=self.source,
                savedir=self.savedir,
                run_opts={"device": self.device}
            )
            log.info("LID model loaded successfully on %s", self.device)
        except Exception as e:
            log.error("Failed to load LID model: %s", e)
            self.classifier = None

    def identify_language(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if not self.classifier:
            return "hi" # Fallback

        try:
            # Convert bytes to tensor
            # audio_bytes is PCM16 LE
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            signal = torch.from_numpy(audio_np).to(self.device).unsqueeze(0) # [1, T]

            # SpeechBrain expects 16k, single channel
            # We assume input is already 16k mono
            
            prediction = self.classifier.classify_batch(signal)
            # prediction is (out_prob, score, index, text_lab)
            # text_lab is list of labels
            lang = prediction[3][0] # e.g., "hi"
            
            log.info("LID Detected: %s", lang)
            return lang
        except Exception as e:
            log.error("LID Inference failed: %s", e)
            return "hi"
