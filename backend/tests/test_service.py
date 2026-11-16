"""用假依赖验证 SSE 事件顺序、热点 Query 缓存命中与会话历史写入。"""
import asyncio

from app.agent.service import AgentService
from app.config import Settings


class MemCache:
    def __init__(self):
        self.kv, self.hist = {}, {}

    async def get_history(self, sid):
        return self.hist.get(sid, [])

    async def add_turn(self, sid, q, a):
        self.hist.setdefault(sid, []).extend([{"role": "user", "content": q}, {"role": "assistant", "content": a}])

    async def ns_version(self, ns):
        return "0"

    def query_key(self, ns, ver, q):
        return f"{ns}:{ver}:{q}"

    async def get_json(self, k):
        return self.kv.get(k)

    async def set_json(self, k, v, ttl):
        self.kv[k] = v


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def stream(self, messages):
        self.calls += 1
        for t in ["你", "好", "[1]"]:
            yield t


class FakeGraph:
    async def astream(self, state, stream_mode="updates"):
        yield {"analyze": {"intent": "knowledge", "standalone_question": state["question"], "sub_questions": [state["question"]]}}
        yield {"rerank": {"contexts": [{"filename": "f.pdf", "heading": "H", "score": 0.9, "parent_text": "父块"}], "best_score": 0.9}}
        yield {"build": {"context_text": "[1] 父块", "sufficient": True}}


async def collect(svc, q, sid=None):
    return [ev async for ev in svc.stream(q, "default", sid)]


def test_stream_event_order_cache_and_history():
    cache, llm = MemCache(), FakeLLM()
    svc = AgentService(Settings(), cache, llm, FakeGraph())

    events = asyncio.run(collect(svc, "问题", "s1"))
    kinds = [e["event"] for e in events]
    assert kinds[:3] == ["status", "status", "status"] and kinds[-1] == "done" and "sources" in kinds
    assert "".join(e["data"]["text"] for e in events if e["event"] == "token") == "你好[1]"
    assert cache.hist["s1"][-1]["content"] == "你好[1]"  # 写入会话历史

    events2 = asyncio.run(collect(svc, "问题"))  # 无历史的相同问题 -> 命中缓存, 不再调用 LLM
    assert events2[0]["data"]["node"] == "cache" and events2[-1]["data"]["cached"] is True
    assert llm.calls == 1
