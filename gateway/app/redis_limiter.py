import time
from dataclasses import dataclass
from redis.asyncio.client import Redis as RedisType

@dataclass
class AdmitResult:
    ok: bool
    reason: str

class RedisLimiter:
    """Redis-backed limiter:
    - Active WS connections per api_key (INCR/DECR)
    - Token bucket for new connections per minute per key (Lua)
    """
    def __init__(self, redis: RedisType, max_conns_per_key: int, new_conn_per_min: int, burst: int):
        self.r = redis
        self.max_conns = max_conns_per_key
        self.rate = float(new_conn_per_min) / 60.0  # tokens/sec
        self.burst = float(burst)

        self._lua = r'''
local key = KEYS[1]
local now = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local burst = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local t_key = key .. ":t"
local ts_key = key .. ":ts"

local tokens = tonumber(redis.call("GET", t_key) or burst)
local last = tonumber(redis.call("GET", ts_key) or now)

local delta = math.max(0, now - last)
tokens = math.min(burst, tokens + delta * rate)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call("SET", t_key, tokens, "PX", 600000)
redis.call("SET", ts_key, now, "PX", 600000)
return allowed
'''
    async def _token_allow(self, api_key: str, cost: float = 1.0) -> bool:
        now_ms = int(time.time() * 1000)
        allowed = await self.r.eval(self._lua, 1, f"tb:{api_key}", now_ms, self.rate/1000.0, self.burst, cost)
        return bool(allowed)

    async def admit(self, api_key: str) -> AdmitResult:
        active_key = f"active:{api_key}"
        active = int(await self.r.get(active_key) or 0)
        if active >= self.max_conns:
            return AdmitResult(False, "TOO_MANY_CONNECTIONS")

        if not await self._token_allow(api_key, 1.0):
            return AdmitResult(False, "RATE_LIMITED")

        pipe = self.r.pipeline()
        pipe.incr(active_key, 1)
        pipe.pexpire(active_key, 3600000)
        await pipe.execute()
        return AdmitResult(True, "OK")

    async def release(self, api_key: str) -> None:
        if not api_key:
            return
        try:
            await self.r.decr(f"active:{api_key}", 1)
        except Exception:
            pass
