import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


class Reranker:
    """BGE-Reranker (cross-encoder) 精排, 返回 0~1 的相关度分数。"""

    def __init__(self, s):
        self.s = s
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from FlagEmbedding import FlagReranker

                    logger.info("loading reranker model %s", self.s.reranker_model)
                    self._model = FlagReranker(self.s.reranker_model, use_fp16=self.s.use_fp16)
        return self._model

    def warmup(self):
        self._load()

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        scores = self._load().compute_score(
            [[query, p] for p in passages], batch_size=16, max_length=1024, normalize=True
        )
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(x) for x in scores]

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._score_sync, query, passages)
