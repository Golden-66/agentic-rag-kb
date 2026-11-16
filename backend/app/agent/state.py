from typing import TypedDict


class AgentState(TypedDict, total=False):
    # 输入
    question: str
    namespace: str
    history: list[dict]
    # 路由 / 规划
    intent: str  # knowledge | chitchat
    standalone_question: str
    sub_questions: list[str]
    plans: list[dict]  # [{"sub_question", "queries", "hyde"}]
    # 检索结果
    candidates: list[dict]
    contexts: list[dict]
    best_score: float
    sufficient: bool
    retries: int
    # 输出
    context_text: str
