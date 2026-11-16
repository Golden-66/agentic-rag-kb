import asyncio
import logging
import shutil
from pathlib import Path

from .captions import caption_images
from .chunker import chunk_markdown
from .parser import parse_to_markdown

logger = logging.getLogger(__name__)


class IngestPipeline:
    """解析 -> (图片描述) -> 标题层级父子分块 -> BGE-M3 向量化 -> Milvus + SQLite。"""

    def __init__(self, s, meta, llm, embedder, milvus, bm25, cache):
        self.s, self.meta, self.llm = s, meta, llm
        self.embedder, self.milvus, self.bm25, self.cache = embedder, milvus, bm25, cache

    async def run(self, doc_id: str, path: Path, filename: str, namespace: str):
        workdir = self.s.data_path / "work" / doc_id
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            md, img_dir = await asyncio.to_thread(parse_to_markdown, path, workdir, self.s)
            if self.s.vlm_enabled and img_dir is not None:
                md = await caption_images(md, img_dir, self.llm, self.s)

            parents, children = chunk_markdown(
                md, self.s.parent_max_chars, self.s.child_max_chars, self.s.child_overlap
            )
            if not children:
                raise ValueError("文档解析后没有可用文本")

            embs = await self.embedder.encode([c["text"] for c in children], use_cache=False)
            rows = [
                {
                    "id": c["child_id"],
                    "dense": e["dense"],
                    "sparse": e["sparse"],
                    "text": c["text"],
                    "parent_id": c["parent_id"],
                    "doc_id": doc_id,
                    "namespace": namespace,
                }
                for c, e in zip(children, embs)
            ]
            await asyncio.to_thread(self.milvus.insert, rows)
            await asyncio.to_thread(self.meta.insert_chunks, doc_id, namespace, parents, children)

            self.bm25.invalidate(namespace)
            await self.cache.bump_ns(namespace)  # 让该命名空间的旧问答缓存失效
            self.meta.set_status(doc_id, "ready", n_chunks=len(children))
            logger.info("文档 %s 入库完成: %d 父块 / %d 子块", filename, len(parents), len(children))
        except Exception as e:  # noqa: BLE001
            logger.exception("文档 %s 处理失败", filename)
            self.meta.set_status(doc_id, "failed", error=str(e)[:500])
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
