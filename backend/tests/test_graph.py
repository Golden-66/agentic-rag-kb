"""用假的 LLM / 检索器 / 重排器验证 LangGraph 的路由与纠错重试逻辑 (无需任何外部服务)。"""
import asyncio

from app import prompts as P
from app.agent.graph import build_graph
from app.config import Settings


class FakeLLM:
    def __init__(self, analyze, grade=None):
        self.analyze, self.grade_value = analyze, grade

    async def chat_json(self, messages, default):
        sys = messages[0]["content"]
        if sys == P.ANALYZE_SYSTEM:
            return self.analyze
        if sys == P.EXPAND_SYSTEM:
            return {"rewrites": ["改写A", "改写B"], "hyde": "假设文档"}
        if sys == P.GRADE_SYSTEM:
            return {"sufficient": self.grade_value}
        if sys == P.REFINE_SYSTEM:
            return {"queries": ["新查询"]}
        return default


class FakeRetriever:
    def __init__(self):
        self.calls = []

    async def retrieve(self, namespace, plans):
        self.calls.append(plans)
        return [
            {"child_id": "c1", "parent_id": "p1", "doc_id": "d", "filename": "f.pdf", "heading": "H",
             "child_text": "子块1", "parent_text": "父块1", "rrf_score": 0.1},
            {"child_id": "c2", "parent_id": "p1", "doc_id": "d", "filename": "f.pdf", "heading": "H",
             "child_text": "子块2", "parent_text": "父块1", "rrf_score": 0.09},
            {"child_id": "c3", "parent_id": "p2", "doc_id": "d", "filename": "f.pdf", "heading": "H2",
             "child_text": "子块3", "parent_text": "父块2", "rrf_score": 0.08},
        ]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    async def rerank(self, query, passages):
        return self.scores[: len(passages)]


def run(graph, q="问题"):
    return asyncio.run(graph.ainvoke({"question": q, "namespace": "default", "history": []}))


def test_chitchat_skips_retrieval():
    llm = FakeLLM({"intent": "chitchat", "standalone_question": "你好", "sub_questions": ["你好"]})
    retriever = FakeRetriever()
    out = run(build_graph(Settings(), llm, retriever, FakeReranker([0.9, 0.8, 0.7])), "你好")
    assert out["intent"] == "chitchat" and retriever.calls == []


def test_fast_path_and_parent_dedup():
    llm = FakeLLM({"intent": "knowledge", "standalone_question": "问题", "sub_questions": ["问题"]}, grade=False)
    retriever = FakeRetriever()
    out = run(build_graph(Settings(), llm, retriever, FakeReranker([0.9, 0.8, 0.7])))
    assert out["sufficient"] is True  # 最高分 >= 阈值, 不需要 LLM 裁判
    assert len(retriever.calls) == 1
    assert [c["parent_id"] for c in out["contexts"]] == ["p1", "p2"]  # c1/c2 同属父块 p1, 只保留一个
    assert "[1] 来源：f.pdf" in out["context_text"]


def test_low_recall_triggers_refine_until_max_retries():
    llm = FakeLLM({"intent": "knowledge", "standalone_question": "问题", "sub_questions": ["问题"]}, grade=False)
    retriever = FakeRetriever()
    s = Settings(max_retries=2)
    out = run(build_graph(s, llm, retriever, FakeReranker([0.1, 0.1, 0.1])))
    assert len(retriever.calls) == 3  # 首次 + 2 次纠错重试
    assert out["retries"] == 2 and out["sufficient"] is False
    assert retriever.calls[-1][0]["queries"] == ["新查询"]


def test_sub_questions_are_planned_separately():
    llm = FakeLLM({"intent": "knowledge", "standalone_question": "A和B区别", "sub_questions": ["A是什么", "B是什么"]})
    retriever = FakeRetriever()
    run(build_graph(Settings(), llm, retriever, FakeReranker([0.9, 0.8, 0.7])))
    plans = retriever.calls[0]
    assert [p["sub_question"] for p in plans] == ["A是什么", "B是什么"]
    assert plans[0]["queries"][0] == "A是什么" and plans[0]["hyde"] == "假设文档"
