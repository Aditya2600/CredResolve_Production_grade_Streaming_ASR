import time
from collections import defaultdict

class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: float):
        self.rate = rate_per_sec
        self.burst = burst
        self.tokens = burst
        self.last = time.time()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.time()
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

class FallbackLimiter:
    def __init__(self, max_conns_per_key: int, new_conn_per_min: int, burst: int):
        self.active = defaultdict(int)
        self.bucket = defaultdict(lambda: TokenBucket(new_conn_per_min/60.0, burst))
        self.max_conns = max_conns_per_key

    def admit(self, api_key: str) -> tuple[bool, str]:
        if self.active[api_key] >= self.max_conns:
            return False, "TOO_MANY_CONNECTIONS"
        if not self.bucket[api_key].allow(1.0):
            return False, "RATE_LIMITED"
        self.active[api_key] += 1
        return True, "OK"

    def release(self, api_key: str) -> None:
        self.active[api_key] = max(0, self.active[api_key] - 1)
