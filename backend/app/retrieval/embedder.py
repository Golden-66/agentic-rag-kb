import asyncio
import logging
import threading

from ..cache import sha1

logger = logging.getLogger(__name__)


class Embedder:
    """BGE-M3: 一次前向同时产出稠密向量与稀疏(词权重)向量。带 Redis Embedding 缓存。"""

    def __init__(self, s, cache):
        self.s, self.cache = s, cache
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from FlagEmbedding import BGEM3FlagModel  # 延迟导入, 加快启动/测试

                    logger.info("loading embedding model %s", self.s.embedding_model)
                    self._model = BGEM3FlagModel(self.s.embedding_model, use_fp16=self.s.use_fp16)
        return self._model

    def warmup(self):
        self._load()

    def _encode_sync(self, texts: list[str]) -> list[dict]:
        out = self._load().encode(
            texts,
            batch_size=self.s.embed_batch_size,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        res = []
        for dense, sparse in zip(out["dense_vecs"], out["lexical_weights"]):
            sp = {int(k): float(v) for k, v in sparse.items()} or {0: 1e-6}  # Milvus 不允许空稀疏向量
            res.append({"dense": [float(x) for x in dense], "sparse": sp})
        return res

    async def encode(self, texts: list[str], use_cache: bool = True) -> list[dict]:
        if not texts:
            return []
        results: list[dict | None] = [None] * len(texts)
        keys = [f"emb:{sha1(t)}" for t in texts]
        missing = list(range(len(texts)))

        if use_cache:
            missing = []
            for i, c in enumerate(await self.cache.mget_json(keys)):
                if c:
                    results[i] = {"dense": c["dense"], "sparse": {int(k): v for k, v in c["sparse"].items()}}
                else:
                    missing.append(i)

        if missing:
            encoded = await asyncio.to_thread(self._encode_sync, [texts[i] for i in missing])
            for i, e in zip(missing, encoded):
                results[i] = e
                if use_cache:
                    payload = {"dense": e["dense"], "sparse": {str(k): v for k, v in e["sparse"].items()}}
                    await self.cache.set_json(keys[i], payload, self.s.embedding_cache_ttl)
        return results  # type: ignore[return-value]
