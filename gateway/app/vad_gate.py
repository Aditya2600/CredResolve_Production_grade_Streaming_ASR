from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class GateState(str, Enum):
    CLOSED = "closed"
    CANDIDATE = "candidate"
    OPEN = "open"
    HANGOVER = "hangover"


@dataclass(frozen=True)
class VADGateConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    vad_mode: int = 3
    open_window_frames: int = 3
    open_required_voiced_frames: int = 2
    close_window_frames: int = 5
    close_required_unvoiced_frames: int = 4
    hangover_ms: int = 200

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise ValueError("VADGateConfig.sample_rate must be 16000")
        if self.frame_ms != 20:
            raise ValueError("VADGateConfig.frame_ms must be 20")
        if self.vad_mode != 3:
            raise ValueError("VADGateConfig.vad_mode must be 3")
        if not 0 < self.open_required_voiced_frames <= self.open_window_frames:
            raise ValueError("open_required_voiced_frames must be within open_window_frames")
        if not 0 < self.close_required_unvoiced_frames <= self.close_window_frames:
            raise ValueError("close_required_unvoiced_frames must be within close_window_frames")
        if self.hangover_ms <= 0 or self.hangover_ms % self.frame_ms != 0:
            raise ValueError("hangover_ms must be a positive multiple of frame_ms")

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * (self.frame_ms / 1000.0) * 2)

    @property
    def hangover_frames(self) -> int:
        return self.hangover_ms // self.frame_ms


@dataclass(frozen=True)
class GateUpdate:
    previous_state: GateState
    state: GateState
    reason: str | None = None

    @property
    def transitioned(self) -> bool:
        return self.previous_state != self.state

    @property
    def opened(self) -> bool:
        return self.previous_state != GateState.OPEN and self.state == GateState.OPEN

    @property
    def closed(self) -> bool:
        return self.previous_state != GateState.CLOSED and self.state == GateState.CLOSED


class VADGateStateMachine:
    def __init__(self, config: VADGateConfig):
        self.config = config
        self._open_window = deque(maxlen=config.open_window_frames)
        self._close_window = deque(maxlen=config.close_window_frames)
        self.reset()

    @property
    def state(self) -> GateState:
        return self._state

    def reset(self) -> None:
        self._state = GateState.CLOSED
        self._hangover_frames_left = 0
        self._open_window.clear()
        self._close_window.clear()

    def process_frame(self, *, is_speech: bool, can_open: bool) -> GateUpdate:
        self._open_window.append(is_speech)
        self._close_window.append(is_speech)

        previous_state = self._state
        reason: str | None = None

        open_ready = (
            len(self._open_window) == self.config.open_window_frames
            and sum(1 for value in self._open_window if value)
            >= self.config.open_required_voiced_frames
        )
        close_ready = (
            len(self._close_window) == self.config.close_window_frames
            and sum(1 for value in self._close_window if not value)
            >= self.config.close_required_unvoiced_frames
        )

        if self._state == GateState.CLOSED:
            if open_ready:
                if can_open:
                    self._state = GateState.OPEN
                    reason = "vad_open"
                else:
                    self._state = GateState.CANDIDATE
                    reason = "awaiting_speaker_accept"
        elif self._state == GateState.CANDIDATE:
            if open_ready and can_open:
                self._state = GateState.OPEN
                reason = "speaker_accept"
            elif close_ready:
                self._state = GateState.CLOSED
                reason = "candidate_timeout"
        elif self._state == GateState.OPEN:
            if close_ready:
                self._state = GateState.HANGOVER
                self._hangover_frames_left = self.config.hangover_frames
                reason = "vad_close_pending"
        elif self._state == GateState.HANGOVER:
            if open_ready:
                self._state = GateState.OPEN
                self._hangover_frames_left = 0
                reason = "speech_resumed"
            else:
                self._hangover_frames_left = max(0, self._hangover_frames_left - 1)
                if self._hangover_frames_left == 0:
                    self._state = GateState.CLOSED
                    reason = "hangover_elapsed"

        return GateUpdate(previous_state=previous_state, state=self._state, reason=reason)
