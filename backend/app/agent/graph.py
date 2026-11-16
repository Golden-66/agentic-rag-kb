"""LangGraph 编排的 Agentic RAG 检索流程 (生成阶段由 AgentService 流式完成)。

  START -> analyze ──chitchat──> direct ──────────────────────────────> END
              │
              └knowledge─> expand -> retrieve -> rerank -> grade ──足够/达到重试上限──> build -> END
                                        ▲                     │
                                        └──────── refine <────┘ (召回不足: 改写查询纠错重试)
"""
import asyncio
import logging

from langgraph.graph import END, START, StateGraph

from .. import prompts as P
from .state import AgentState

logger = logging.getLogger(__name__)


def _clean_list(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for x in value:
        if isinstance(x, str) and x.strip() and x.strip() not in out:
            out.append(x.strip())
    return out[:limit]


def format_history(history: list[dict]) -> str:
    if not history:
        return "（无）"
    return "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'][:300]}" for m in history[-6:]
    )


def _sys(content: str, user: str) -> list[dict]:
    return [{"role": "system", "content": content}, {"role": "user", "content": user}]


def build_graph(s, llm, retriever, reranker):
    # ---------- 1. 意图识别 + 指代消解 + 子问题拆解 ----------
    async def analyze(state: AgentState):
        q = state["question"]
        user = f"对话历史：\n{format_history(state.get('history', []))}\n\n当前问题：{q}"
        default = {"intent": "knowledge", "standalone_question": q, "sub_questions": [q]}
        data = await llm.chat_json(_sys(P.ANALYZE_SYSTEM, user), default)
        intent = data.get("intent") if data.get("intent") in ("knowledge", "chitchat") else "knowledge"
        standalone = str(data.get("standalone_question") or q).strip() or q
        subs = _clean_list(data.get("sub_questions"), s.max_sub_questions) or [standalone]
        return {"intent": intent, "standalone_question": standalone, "sub_questions": subs, "retries": 0}

    # ---------- 2. Query 改写 + 多查询扩展 + HyDE ----------
    async def expand(state: AgentState):
        async def one(q: str) -> dict:
            user = f"问题：{q}\n请生成 {s.multi_query_n} 条改写查询。"
            data = await llm.chat_json(_sys(P.EXPAND_SYSTEM, user), {"rewrites": [], "hyde": ""})
            rewrites = _clean_list(data.get("rewrites"), s.multi_query_n)
            hyde = str(data.get("hyde") or "").strip() if s.enable_hyde else ""
            return {"sub_question": q, "queries": [q] + [r for r in rewrites if r != q], "hyde": hyde or None}

        plans = await asyncio.gather(*[one(q) for q in state["sub_questions"]])
        return {"plans": list(plans)}

    # ---------- 3. 混合召回 ----------
    async def retrieve(state: AgentState):
        candidates = await retriever.retrieve(state["namespace"], state["plans"])
        return {"candidates": candidates}

    # ---------- 4. Cross-Encoder 重排 + 父块去重 ----------
    async def rerank(state: AgentState):
        cands = state.get("candidates", [])
        if not cands:
            return {"contexts": [], "best_score": 0.0}
        # 对原问题和每个子问题分别打分, 取最大值, 避免拆解后的子问题被漏掉
        queries = list(dict.fromkeys([state["standalone_question"], *state["sub_questions"]]))[:4]
        passages = [c["child_text"] for c in cands]
        all_scores = await asyncio.gather(*[reranker.rerank(q, passages) for q in queries])
        scored = []
        for i, c in enumerate(cands):
            scored.append({**c, "score": max(sc[i] for sc in all_scores)})
        scored.sort(key=lambda x: x["score"], reverse=True)

        contexts, seen = [], set()
        for c in scored:  # 同一父块只保留最高分的子块
            if c["parent_id"] in seen:
                continue
            seen.add(c["parent_id"])
            contexts.append(c)
            if len(contexts) >= s.rerank_top_k:
                break
        return {"contexts": contexts, "best_score": contexts[0]["score"] if contexts else 0.0}

    # ---------- 5. 召回质量评估 (重排分数快速通道 / LLM 裁判) ----------
    async def grade(state: AgentState):
        ctxs = state.get("contexts", [])
        if not ctxs:
            return {"sufficient": False}
        if state.get("best_score", 0.0) >= s.rerank_fast_threshold:
            return {"sufficient": True}  # 快速通道: 跳过一次 LLM 调用
        snippet = "\n\n".join(f"[{i}] {c['parent_text'][:600]}" for i, c in enumerate(ctxs, 1))
        user = f"问题：{state['standalone_question']}\n\n参考资料：\n{snippet}"
        data = await llm.chat_json(_sys(P.GRADE_SYSTEM, user), {"sufficient": True})
        return {"sufficient": bool(data.get("sufficient", True))}

    def route_after_grade(state: AgentState) -> str:
        if state.get("sufficient") or state.get("retries", 0) >= s.max_retries:
            return "build"
        return "refine"

    # ---------- 6. 迭代查询纠错 ----------
    async def refine(state: AgentState):
        tried = [q for p in state.get("plans", []) for q in p["queries"]]
        user = f"原问题：{state['standalone_question']}\n已尝试的查询：\n" + "\n".join(f"- {q}" for q in tried)
        data = await llm.chat_json(_sys(P.REFINE_SYSTEM, user), {"queries": []})
        qs = _clean_list(data.get("queries"), s.multi_query_n + 1) or [state["standalone_question"] + " 详细说明"]
        return {
            "plans": [{"sub_question": state["standalone_question"], "queries": qs, "hyde": None}],
            "retries": state.get("retries", 0) + 1,
        }

    # ---------- 7. 整理上下文 ----------
    async def build(state: AgentState):
        blocks = []
        for i, c in enumerate(state.get("contexts", []), 1):
            blocks.append(f"[{i}] 来源：{c['filename']} | {c['heading'] or '正文'}\n{c['parent_text']}")
        return {"context_text": "\n\n".join(blocks)}

    async def direct(state: AgentState):
        return {"contexts": [], "context_text": "", "sufficient": True}

    g = StateGraph(AgentState)
    for name, fn in [
        ("analyze", analyze), ("expand", expand), ("retrieve", retrieve), ("rerank", rerank),
        ("grade", grade), ("refine", refine), ("build", build), ("direct", direct),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", lambda st: st["intent"], {"knowledge": "expand", "chitchat": "direct"})
    g.add_edge("expand", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade")
    g.add_conditional_edges("grade", route_after_grade, {"build": "build", "refine": "refine"})
    g.add_edge("refine", "retrieve")
    g.add_edge("build", END)
    g.add_edge("direct", END)
    return g.compile()
