import asyncio

from .fusion import rrf_fuse


class Retriever:
    """多路召回: 每个查询(原问题/改写/HyDE) 同时走
       - Milvus 稠密+稀疏原生混合检索 (库内 RRF)
       - BM25 关键词检索
    再把所有结果列表做一次应用层 RRF, 并把命中的子块映射回父块。"""

    def __init__(self, s, embedder, milvus, bm25, meta):
        self.s, self.embedder, self.milvus, self.bm25, self.meta = s, embedder, milvus, bm25, meta

    async def retrieve(self, namespace: str, plans: list[dict]) -> list[dict]:
        texts: list[str] = []
        is_query: list[bool] = []
        for plan in plans:
            for q in plan["queries"]:
                texts.append(q)
                is_query.append(True)
            if plan.get("hyde"):
                texts.append(plan["hyde"])
                is_query.append(False)  # HyDE 文档只用于向量/稀疏召回, 不走 BM25
        if not texts:
            return []

        embs = await self.embedder.encode(texts)  # 一次批量编码, 命中缓存的直接跳过

        tasks = []
        for text, q_flag, emb in zip(texts, is_query, embs):
            tasks.append(
                asyncio.to_thread(self.milvus.hybrid_search, emb["dense"], emb["sparse"], namespace, self.s.recall_top_k)
            )
            if q_flag:
                tasks.append(asyncio.to_thread(self.bm25.search, namespace, text, self.s.recall_top_k))
        results = await asyncio.gather(*tasks)

        fused = rrf_fuse([[h["id"] for h in hits] for hits in results], k=self.s.rrf_k)
        ids = [i for i, _ in fused[: self.s.recall_top_k]]
        rrf_score = dict(fused)

        children = await asyncio.to_thread(self.meta.get_children, ids)
        parents = await asyncio.to_thread(self.meta.get_parents, [c["parent_id"] for c in children.values()])
        names = await asyncio.to_thread(self.meta.get_filenames, [c["doc_id"] for c in children.values()])

        candidates = []
        for cid in ids:
            c = children.get(cid)
            if not c or c["parent_id"] not in parents:
                continue
            p = parents[c["parent_id"]]
            candidates.append(
                {
                    "child_id": cid,
                    "parent_id": c["parent_id"],
                    "doc_id": c["doc_id"],
                    "filename": names.get(c["doc_id"], ""),
                    "heading": p["heading"],
                    "child_text": c["text"],
                    "parent_text": p["text"],
                    "rrf_score": rrf_score[cid],
                }
            )
        return candidates
