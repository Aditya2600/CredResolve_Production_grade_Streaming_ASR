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

        self._admit_lua = r'''
local active_key = KEYS[1]
local tokens_key = KEYS[2]
local ts_key = KEYS[3]

local max_conns = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local rate = tonumber(ARGV[3])
local burst = tonumber(ARGV[4])
local cost = tonumber(ARGV[5])

local active = tonumber(redis.call("GET", active_key) or "0")
if active >= max_conns then
  return "TOO_MANY_CONNECTIONS"
end

local tokens = tonumber(redis.call("GET", tokens_key) or burst)
local last = tonumber(redis.call("GET", ts_key) or now)
local delta = math.max(0, now - last)
tokens = math.min(burst, tokens + delta * rate)

if tokens < cost then
  redis.call("SET", tokens_key, tokens, "PX", 600000)
  redis.call("SET", ts_key, now, "PX", 600000)
  return "RATE_LIMITED"
end

tokens = tokens - cost
redis.call("SET", tokens_key, tokens, "PX", 600000)
redis.call("SET", ts_key, now, "PX", 600000)
redis.call("INCR", active_key)
redis.call("PEXPIRE", active_key, 3600000)
return "OK"
'''

        self._release_lua = r'''
local active_key = KEYS[1]
local active = tonumber(redis.call("GET", active_key) or "0")

if active <= 1 then
  redis.call("DEL", active_key)
  return 0
end

return redis.call("DECR", active_key)
'''

    async def admit(self, api_key: str) -> AdmitResult:
        active_key = f"active:{api_key}"
        result = await self.r.eval(
            self._admit_lua,
            3,
            active_key,
            f"tb:{api_key}:t",
            f"tb:{api_key}:ts",
            self.max_conns,
            int(time.time() * 1000),
            self.rate / 1000.0,
            self.burst,
            1.0,
        )
        if isinstance(result, bytes):
            result = result.decode("utf-8")
        reason = str(result)
        return AdmitResult(reason == "OK", reason)

    async def release(self, api_key: str) -> None:
        if not api_key:
            return
        try:
            await self.r.eval(self._release_lua, 1, f"active:{api_key}")
        except Exception:
            pass
