"""Short-lived Redis cache for verified, tenant-scoped answers."""

import asyncio
import json
import logging
from dataclasses import astuple
from hashlib import sha256

from redis.exceptions import RedisError

logger = logging.getLogger(__name__)
TTL_SECONDS = 300
IO_TIMEOUT_SECONDS = 0.25


def answer_cache_key(owner, revision, versions, question, turns, configuration) -> str:
    payload = json.dumps([
        str(owner), str(revision), sorted(str(version) for version in versions), question,
        [astuple(turn) for turn in turns], configuration,
    ], separators=(",", ":"))
    return "rag:answer:v1:" + sha256(payload.encode()).hexdigest()


async def read_answer(redis, key: str) -> dict | None:
    try:
        raw = await asyncio.wait_for(redis.get(key), IO_TIMEOUT_SECONDS)
        if raw is None:
            return None
        value = json.loads(raw)
        if (isinstance(value, dict) and isinstance(value.get("answer"), str)
                and value.get("abstained") is False and isinstance(value.get("citations"), list)
                and value["citations"]):
            return value
    except (RedisError, asyncio.TimeoutError, ValueError, RecursionError) as exc:
        logger.warning("Answer cache read failed (%s)", type(exc).__name__)
    return None


async def write_answer(redis, key: str, answer: dict) -> None:
    try:
        await asyncio.wait_for(redis.set(key, json.dumps(answer), ex=TTL_SECONDS), IO_TIMEOUT_SECONDS)
    except (RedisError, asyncio.TimeoutError) as exc:
        logger.warning("Answer cache write failed (%s)", type(exc).__name__)
