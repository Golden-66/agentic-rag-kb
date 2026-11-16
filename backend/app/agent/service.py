import logging
import time
from typing import AsyncIterator

from .. import prompts as P

logger = logging.getLogger(__name__)

NODE_LABELS = {
    "analyze": "理解问题 / 意图识别 / 子问题拆解",
    "expand": "查询改写与多查询扩展 (HyDE)",
    "retrieve": "Dense + Sparse + BM25 混合检索",
    "rerank": "BGE-Reranker 重排序",
    "grade": "评估召回质量",
    "refine": "召回不足, 改写查询纠错重试",
    "build": "整理上下文",
    "direct": "闲聊, 直接回答",
}


def _detail(node: str, upd: dict) -> str:
    if node == "analyze":
        return f"intent={upd.get('intent')} | 子问题: {upd.get('sub_questions')}"
    if node == "retrieve":
        return f"候选子块 {len(upd.get('candidates', []))} 个"
    if node == "rerank":
        return f"保留 {len(upd.get('contexts', []))} 个父块, 最高分 {upd.get('best_score', 0):.3f}"
    if node == "grade":
        return "召回充分" if upd.get("sufficient") else "召回不足"
    if node == "refine":
        return f"第 {upd.get('retries')} 次重试"
    return ""


class AgentService:
    def __init__(self, s, cache, llm, graph):
        self.s, self.cache, self.llm, self.graph = s, cache, llm, graph

    # ---------- helpers ----------
    @staticmethod
    def sources(state: dict, full: bool = False) -> list[dict]:
        return [
            {
                "index": i,
                "filename": c["filename"],
                "heading": c["heading"],
                "score": round(c["score"], 4),
                "text": c["parent_text"] if full else c["parent_text"][:500],
            }
            for i, c in enumerate(state.get("contexts", []), 1)
        ]

    def _messages(self, state: dict, history: list[dict]) -> list[dict]:
        if state.get("intent") == "chitchat":
            return [{"role": "system", "content": P.CHITCHAT_SYSTEM}, *history, {"role": "user", "content": state["question"]}]
        ctx = state.get("context_text") or "（未检索到相关资料）"
        hint = "" if state.get("sufficient") else "\n（提示：检索到的资料可能不足以完整回答，请如实说明。）"
        user = f"参考资料：\n{ctx}\n\n问题：{state['standalone_question']}{hint}"
        return [{"role": "system", "content": P.ANSWER_SYSTEM}, *history, {"role": "user", "content": user}]

    async def _finish(self, session_id, question, answer):
        if session_id and answer:
            await self.cache.add_turn(session_id, question, answer)

    # ---------- main flow ----------
    async def stream(self, question: str, namespace: str, session_id: str | None) -> AsyncIterator[dict]:
        t0 = time.perf_counter()
        history = await self.cache.get_history(session_id) if session_id else []

        # 热点 Query 缓存: 仅对无历史上下文的独立问题生效
        cache_key = None
        if not history:
            ver = await self.cache.ns_version(namespace)
            cache_key = self.cache.query_key(namespace, ver, question)
            hit = await self.cache.get_json(cache_key)
            if hit:
                yield {"event": "status", "data": {"node": "cache", "label": "命中热点问答缓存", "detail": ""}}
                ans = hit["answer"]
                for i in range(0, len(ans), 24):
                    yield {"event": "token", "data": {"text": ans[i : i + 24]}}
                yield {"event": "sources", "data": {"sources": hit.get("sources", [])}}
                await self._finish(session_id, question, ans)
                ms = int((time.perf_counter() - t0) * 1000)
                yield {"event": "done", "data": {"cached": True, "latency_ms": ms, "first_token_ms": ms}}
                return

        state: dict = {"question": question, "namespace": namespace, "history": history}
        async for update in self.graph.astream(state, stream_mode="updates"):
            for node, upd in update.items():
                state.update(upd or {})
                yield {
                    "event": "status",
                    "data": {"node": node, "label": NODE_LABELS.get(node, node), "detail": _detail(node, upd or {})},
                }

        parts: list[str] = []
        first_ms = None
        async for tok in self.llm.stream(self._messages(state, history)):
            if first_ms is None:
                first_ms = int((time.perf_counter() - t0) * 1000)
            parts.append(tok)
            yield {"event": "token", "data": {"text": tok}}
        answer = "".join(parts)

        sources = self.sources(state)
        yield {"event": "sources", "data": {"sources": sources}}

        if cache_key and state.get("intent") == "knowledge" and state.get("contexts") and answer:
            await self.cache.set_json(cache_key, {"answer": answer, "sources": sources}, self.s.query_cache_ttl)
        await self._finish(session_id, question, answer)
        yield {
            "event": "done",
            "data": {"cached": False, "latency_ms": int((time.perf_counter() - t0) * 1000), "first_token_ms": first_ms},
        }

    async def answer(self, question: str, namespace: str, session_id: str | None) -> dict:
        text, sources, meta = [], [], {}
        async for ev in self.stream(question, namespace, session_id):
            if ev["event"] == "token":
                text.append(ev["data"]["text"])
            elif ev["event"] == "sources":
                sources = ev["data"]["sources"]
            elif ev["event"] == "done":
                meta = ev["data"]
        return {"answer": "".join(text), "sources": sources, **meta}

    async def retrieve_only(self, question: str, namespace: str) -> dict:
        """只跑检索图, 不生成答案 (用于评测)。"""
        state = await self.graph.ainvoke({"question": question, "namespace": namespace, "history": []})
        return {
            "intent": state.get("intent"),
            "sub_questions": state.get("sub_questions", []),
            "contexts": self.sources(state, full=True),
        }
