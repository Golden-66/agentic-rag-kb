import hashlib
import json
import logging

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class Cache:
    """Redis 缓存: Embedding 缓存 / 热点 Query 结果缓存 / 会话历史。
    Redis 不可用时自动降级 (所有操作吞掉异常), 不影响主流程。"""

    def __init__(self, s):
        self.s = s
        self.r = aioredis.from_url(s.redis_url, decode_responses=True)

    async def _safe(self, coro, default=None):
        try:
            return await coro
        except Exception as e:  # noqa: BLE001
            logger.warning("redis unavailable: %s", e)
            return default

    # ---- 通用 JSON ----
    async def get_json(self, key: str):
        raw = await self._safe(self.r.get(key))
        return json.loads(raw) if raw else None

    async def set_json(self, key: str, value, ttl: int):
        await self._safe(self.r.set(key, json.dumps(value, ensure_ascii=False), ex=ttl))

    async def mget_json(self, keys: list[str]) -> list:
        if not keys:
            return []
        raws = await self._safe(self.r.mget(keys), [None] * len(keys))
        return [json.loads(x) if x else None for x in raws]

    # ---- 命名空间版本号: 文档变更后让旧的问答缓存整体失效 ----
    async def ns_version(self, ns: str) -> str:
        return (await self._safe(self.r.get(f"ver:{ns}"))) or "0"

    async def bump_ns(self, ns: str):
        await self._safe(self.r.incr(f"ver:{ns}"))

    @staticmethod
    def query_key(ns: str, version: str, question: str) -> str:
        return f"qa:{ns}:{version}:{sha1(' '.join(question.lower().split()))}"

    # ---- 会话历史 (保留最近 N 轮) ----
    async def get_history(self, session_id: str) -> list[dict]:
        return (await self.get_json(f"hist:{session_id}")) or []

    async def add_turn(self, session_id: str, question: str, answer: str):
        hist = await self.get_history(session_id)
        hist += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        hist = hist[-self.s.history_turns * 2 :]
        await self.set_json(f"hist:{session_id}", hist, 24 * 3600)
