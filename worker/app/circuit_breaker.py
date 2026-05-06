from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

from .metrics import (
    CIRCUIT_BREAKER_CALLS,
    CIRCUIT_BREAKER_STATE,
    CIRCUIT_BREAKER_TRANSITIONS,
)

log = logging.getLogger("worker.circuit_breaker")

T = TypeVar("T")


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(RuntimeError):
    pass


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    name: str
    state: CircuitBreakerState
    failure_count: int
    opened_at: float | None
    half_open_in_flight: bool
    half_open_success_count: int
    last_failure: str


class CircuitBreaker:
    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 30.0,
        half_open_success_threshold: int = 1,
        clock: Callable[[], float] | None = None,
        enabled: bool = True,
    ) -> None:
        self.name = (name or "default").strip()
        self.failure_threshold = max(int(failure_threshold), 1)
        self.recovery_timeout_sec = max(float(recovery_timeout_sec), 0.0)
        self.half_open_success_threshold = max(int(half_open_success_threshold), 1)
        self.enabled = bool(enabled)
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = False
        self._half_open_success_count = 0
        self._last_failure = ""
        self._publish_state_locked()

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            return self._state

    def snapshot(self) -> CircuitBreakerSnapshot:
        with self._lock:
            return CircuitBreakerSnapshot(
                name=self.name,
                state=self._state,
                failure_count=self._failure_count,
                opened_at=self._opened_at,
                half_open_in_flight=self._half_open_in_flight,
                half_open_success_count=self._half_open_success_count,
                last_failure=self._last_failure,
            )

    def call(self, fn: Callable[[], T]) -> T:
        if not self.enabled:
            return fn()

        state_at_call = self._before_call()
        try:
            result = fn()
        except Exception as exc:
            CIRCUIT_BREAKER_CALLS.labels(
                name=self.name,
                state=state_at_call.value,
                result="failure",
            ).inc()
            self._after_failure(exc)
            raise

        CIRCUIT_BREAKER_CALLS.labels(
            name=self.name,
            state=state_at_call.value,
            result="success",
        ).inc()
        self._after_success()
        return result

    def _before_call(self) -> CircuitBreakerState:
        now = self._clock()
        with self._lock:
            if (
                self._state == CircuitBreakerState.OPEN
                and self._opened_at is not None
                and now - self._opened_at >= self.recovery_timeout_sec
            ):
                self._transition_locked(CircuitBreakerState.HALF_OPEN, "recovery_timeout")

            if self._state == CircuitBreakerState.OPEN:
                CIRCUIT_BREAKER_CALLS.labels(
                    name=self.name,
                    state=self._state.value,
                    result="blocked",
                ).inc()
                raise CircuitBreakerOpenError(self._blocked_message_locked(now))

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_in_flight:
                    CIRCUIT_BREAKER_CALLS.labels(
                        name=self.name,
                        state=self._state.value,
                        result="blocked",
                    ).inc()
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker `{self.name}` is HALF_OPEN; recovery probe already in flight"
                    )
                self._half_open_in_flight = True

            return self._state

    def _after_success(self) -> None:
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_in_flight = False
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.half_open_success_threshold:
                    self._failure_count = 0
                    self._last_failure = ""
                    self._opened_at = None
                    self._transition_locked(CircuitBreakerState.CLOSED, "probe_success")
                return

            if self._state == CircuitBreakerState.CLOSED:
                self._failure_count = 0
                self._last_failure = ""

    def _after_failure(self, exc: Exception) -> None:
        with self._lock:
            self._last_failure = str(exc)
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_in_flight = False
                self._failure_count = self.failure_threshold
                self._opened_at = self._clock()
                self._transition_locked(CircuitBreakerState.OPEN, "probe_failure")
                return

            if self._state == CircuitBreakerState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._opened_at = self._clock()
                    self._transition_locked(CircuitBreakerState.OPEN, "failure_threshold")

    def _blocked_message_locked(self, now: float) -> str:
        retry_after = None
        if self._opened_at is not None:
            retry_after = max(self.recovery_timeout_sec - (now - self._opened_at), 0.0)
        if retry_after is None:
            return f"Circuit breaker `{self.name}` is OPEN"
        return f"Circuit breaker `{self.name}` is OPEN; retry after {retry_after:.2f}s"

    def _transition_locked(self, new_state: CircuitBreakerState, reason: str) -> None:
        old_state = self._state
        if old_state == new_state:
            return
        self._state = new_state
        if new_state != CircuitBreakerState.HALF_OPEN:
            self._half_open_in_flight = False
            self._half_open_success_count = 0
        CIRCUIT_BREAKER_TRANSITIONS.labels(
            name=self.name,
            from_state=old_state.value,
            to_state=new_state.value,
            reason=reason,
        ).inc()
        self._publish_state_locked()
        log.warning(
            "Circuit breaker transition name=%s from_state=%s to_state=%s reason=%s failure_count=%s last_failure=%s",
            self.name,
            old_state.value,
            new_state.value,
            reason,
            self._failure_count,
            self._last_failure or "-",
        )

    def _publish_state_locked(self) -> None:
        for state in CircuitBreakerState:
            CIRCUIT_BREAKER_STATE.labels(name=self.name, state=state.value).set(
                1 if state == self._state else 0
            )
