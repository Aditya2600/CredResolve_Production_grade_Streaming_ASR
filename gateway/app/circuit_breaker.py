import time

class CircuitBreaker:
    def __init__(self, fail_threshold: int, reset_ms: int):
        self.fail_threshold = fail_threshold
        self.reset_ms = reset_ms
        self.fails = 0
        self.open_until = 0

    def allow(self) -> bool:
        now = int(time.time() * 1000)
        return now >= self.open_until

    def on_success(self) -> None:
        self.fails = 0
        self.open_until = 0

    def on_failure(self) -> None:
        self.fails += 1
        if self.fails >= self.fail_threshold:
            self.open_until = int(time.time() * 1000) + self.reset_ms
