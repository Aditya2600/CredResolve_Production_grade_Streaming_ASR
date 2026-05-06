from __future__ import annotations

import pytest

from worker.app.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_circuit_breaker_opens_blocks_then_closes_after_successful_probe():
    clock = _FakeClock()
    breaker = CircuitBreaker(
        name="test-breaker",
        failure_threshold=2,
        recovery_timeout_sec=5.0,
        clock=clock,
    )

    assert breaker.state == CircuitBreakerState.CLOSED

    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("first failure")))
    assert breaker.state == CircuitBreakerState.CLOSED

    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("second failure")))
    assert breaker.state == CircuitBreakerState.OPEN

    with pytest.raises(CircuitBreakerOpenError):
        breaker.call(lambda: "blocked")

    clock.advance(5.0)
    assert breaker.call(lambda: "probe ok") == "probe ok"
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.snapshot().failure_count == 0


def test_half_open_probe_failure_reopens_circuit():
    clock = _FakeClock()
    breaker = CircuitBreaker(
        name="test-reopen",
        failure_threshold=1,
        recovery_timeout_sec=10.0,
        clock=clock,
    )

    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert breaker.state == CircuitBreakerState.OPEN

    clock.advance(10.0)
    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("still down")))

    snapshot = breaker.snapshot()
    assert snapshot.state == CircuitBreakerState.OPEN
    assert snapshot.failure_count == 1
    assert snapshot.last_failure == "still down"


def test_half_open_allows_only_one_recovery_probe_at_a_time():
    clock = _FakeClock()
    breaker = CircuitBreaker(
        name="test-half-open-probe",
        failure_threshold=1,
        recovery_timeout_sec=1.0,
        clock=clock,
    )

    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))

    clock.advance(1.0)

    def probe() -> str:
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(lambda: "nested probe")
        return "recovered"

    assert breaker.call(probe) == "recovered"
    assert breaker.state == CircuitBreakerState.CLOSED
