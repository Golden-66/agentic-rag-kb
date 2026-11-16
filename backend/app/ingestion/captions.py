"""用 VLM 为文档插图生成文字描述, 让图表/截图里的信息也能被检索到。"""
import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)[^)]*\)")
MIN_IMAGE_BYTES = 3 * 1024  # 过滤图标/分隔线等小图


async def caption_images(md: str, base_dir: Path, llm, s) -> str:
    paths = list(dict.fromkeys(IMG_RE.findall(md)))[: s.max_images]
    sem = asyncio.Semaphore(4)

    async def one(rel: str) -> tuple[str, str]:
        p = (base_dir / rel).resolve()
        if not p.is_file() or p.stat().st_size < MIN_IMAGE_BYTES:
            return rel, ""
        async with sem:
            try:
                return rel, await llm.describe_image(p)
            except Exception as e:  # noqa: BLE001
                logger.warning("图片描述失败 %s: %s", rel, e)
                return rel, ""

    captions = dict(await asyncio.gather(*[one(p) for p in paths]))

    def repl(m: re.Match) -> str:
        cap = captions.get(m.group(1), "")
        return f"[图片描述：{cap}]" if cap else ""  # 图片链接对检索无意义, 直接去掉

    return IMG_RE.sub(repl, md)
