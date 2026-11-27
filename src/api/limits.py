"""Atomic per-user, per-IP and daily counters shared by API processes."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256

from redis.exceptions import RedisError

from .errors import ApiError

_TAKE = """
for i, key in ipairs(KEYS) do
    local used = tonumber(redis.call('GET', key) or '0')
    if used >= tonumber(ARGV[2*i-1]) then return i end
end
for i, key in ipairs(KEYS) do
    if redis.call('INCR', key) == 1 then redis.call('EXPIRE', key, ARGV[2*i]) end
end
return 0
"""


async def take(redis, *buckets: tuple[str, str, int, int]) -> None:
    keys = [f"rag:limit:{name}:{sha256(subject.encode()).hexdigest()}" for name, subject, _, _ in buckets]
    arguments = [value for _, _, limit, ttl in buckets for value in (limit, ttl)]
    try:
        blocked = await redis.eval(_TAKE, len(keys), *keys, *arguments)
    except RedisError as exc:
        raise ApiError(503, "rate_limiter_unavailable", "Please try again shortly.") from exc
    if blocked:
        daily = buckets[blocked - 1][0] == "questions-per-day"
        message = "The daily question quota has been reached." if daily else "Too many requests. Try again later."
        raise ApiError(429, "daily_quota" if daily else "rate_limited", message)


def today() -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return now.date().isoformat(), max(1, int((midnight - now).total_seconds()) + 1)
