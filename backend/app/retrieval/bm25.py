import logging
import threading

import jieba
from rank_bm25 import BM25Okapi

jieba.setLogLevel(logging.WARNING)


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip() and not all(not ch.isalnum() for ch in t)]


class BM25Index:
    """按命名空间懒加载的 BM25 稀疏检索 (jieba 中文分词)。文档变更后调用 invalidate 重建。"""

    def __init__(self, meta):
        self.meta = meta
        self._idx: dict[str, tuple[BM25Okapi, list[str]] | None] = {}
        self._lock = threading.Lock()

    def invalidate(self, namespace: str):
        with self._lock:
            self._idx.pop(namespace, None)

    def _get(self, namespace: str):
        with self._lock:
            if namespace not in self._idx:
                rows = self.meta.iter_children(namespace)
                if rows:
                    ids = [r[0] for r in rows]
                    self._idx[namespace] = (BM25Okapi([tokenize(r[1]) for r in rows]), ids)
                else:
                    self._idx[namespace] = None
            return self._idx[namespace]

    def search(self, namespace: str, query: str, limit: int) -> list[dict]:
        entry = self._get(namespace)
        tokens = tokenize(query)
        if not entry or not tokens:
            return []
        bm25, ids = entry
        scores = bm25.get_scores(tokens)
        top = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:limit]
        return [{"id": ids[i], "score": float(scores[i])} for i in top if scores[i] > 0]
