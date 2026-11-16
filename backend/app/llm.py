import base64
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import AsyncIterator

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from . import prompts as P

logger = logging.getLogger(__name__)
_JSON_RE = re.compile(r"\{.*\}", re.S)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


class LLMClient:
    """基于 LangChain ChatOpenAI 的统一封装, 兼容 DeepSeek / Qwen / OpenAI 等 OpenAI 协议服务。"""

    def __init__(self, s):
        key = s.llm_api_key or "EMPTY"
        common = dict(base_url=s.llm_base_url, api_key=key, timeout=120, max_retries=2)
        self._fast = ChatOpenAI(model=s.llm_model, temperature=0.0, **common)
        self._answer = ChatOpenAI(model=s.llm_model, temperature=0.2, **common)
        self._vlm = ChatOpenAI(
            model=s.vlm_model,
            temperature=0.0,
            base_url=s.vlm_base_url or s.llm_base_url,
            api_key=s.vlm_api_key or key,
            timeout=120,
            max_retries=2,
        )

    async def chat(self, messages: list[dict]) -> str:
        resp = await self._fast.ainvoke(messages)
        return _text(resp.content)

    async def chat_json(self, messages: list[dict], default: dict) -> dict:
        """调用 LLM 并解析 JSON, 任何失败都回退到 default, 保证 Agent 流程不中断。"""
        try:
            text = await self.chat(messages)
            m = _JSON_RE.search(text)
            data = json.loads(m.group(0)) if m else None
            return data if isinstance(data, dict) else default
        except Exception as e:  # noqa: BLE001
            logger.warning("chat_json failed, use default: %s", e)
            return default

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        async for chunk in self._answer.astream(messages):
            tok = _text(chunk.content)
            if tok:
                yield tok

    async def describe_image(self, path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode()
        msg = HumanMessage(
            content=[
                {"type": "text", "text": P.IMAGE_CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        )
        resp = await self._vlm.ainvoke([msg])
        return _text(resp.content).strip()
