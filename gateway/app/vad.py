import webrtcvad

class VADSegmenter:
    def __init__(self, sample_rate=16000, frame_ms=20, mode=2, end_silence_ms=700, max_utt_ms=12000):
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10/20/30 for WebRTC VAD")
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
        self.end_silence_frames = max(1, end_silence_ms // frame_ms)
        self.max_frames = max(1, max_utt_ms // frame_ms)
        self.reset()

    def reset(self):
        self.in_speech = False
        self.silence_frames = 0
        self.frames = 0
        self.buffer = bytearray()

    def push(self, frame: bytes):
        if len(frame) != self.frame_bytes:
            return [], None

        self.frames += 1
        is_speech = self.vad.is_speech(frame, self.sample_rate)
        events, audio_ready = [], None

        if is_speech:
            if not self.in_speech:
                self.in_speech = True
                events.append("speech_start")
            self.silence_frames = 0
            self.buffer.extend(frame)
        else:
            if self.in_speech:
                self.silence_frames += 1
                self.buffer.extend(frame)
                if self.silence_frames >= self.end_silence_frames:
                    self.in_speech = False
                    events.append("speech_end")
                    audio_ready = bytes(self.buffer)
                    self.reset()

        if self.frames >= self.max_frames and self.in_speech and self.buffer:
            events.append("max_utt")
            audio_ready = bytes(self.buffer)
            self.reset()

        return events, audio_ready
