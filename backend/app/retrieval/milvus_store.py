import logging
import time

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

logger = logging.getLogger(__name__)


class MilvusStore:
    """Milvus 集合: 稠密向量(HNSW/COSINE) + 稀疏向量(SPARSE_INVERTED_INDEX/IP), 原生 hybrid_search + RRF。"""

    def __init__(self, s):
        self.s = s
        self.client: MilvusClient | None = None

    def connect(self):
        kwargs = {"uri": self.s.milvus_uri}
        if self.s.milvus_token:
            kwargs["token"] = self.s.milvus_token
        last: Exception | None = None
        for _ in range(30):  # Milvus 启动较慢, 重试等待
            try:
                self.client = MilvusClient(**kwargs)
                self.client.list_collections()
                break
            except Exception as e:  # noqa: BLE001
                last = e
                logger.info("waiting for milvus ... (%s)", e)
                time.sleep(2)
        else:
            raise RuntimeError(f"无法连接 Milvus: {self.s.milvus_uri}") from last
        self._ensure_collection()

    def _ensure_collection(self):
        name = self.s.milvus_collection
        if self.client.has_collection(name):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.s.embedding_dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("namespace", DataType.VARCHAR, max_length=64)

        idx = self.client.prepare_index_params()
        idx.add_index("dense", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
        idx.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        self.client.create_collection(collection_name=name, schema=schema, index_params=idx)
        logger.info("created milvus collection %s", name)

    def insert(self, rows: list[dict], batch: int = 200):
        for i in range(0, len(rows), batch):
            self.client.insert(collection_name=self.s.milvus_collection, data=rows[i : i + batch])

    def hybrid_search(self, dense: list[float], sparse: dict, namespace: str, limit: int) -> list[dict]:
        expr = f'namespace == "{namespace}"'  # namespace 已在 API 层做白名单校验
        reqs = [
            AnnSearchRequest(
                data=[dense],
                anns_field="dense",
                param={"metric_type": "COSINE", "params": {"ef": max(64, limit)}},
                limit=limit,
                expr=expr,
            ),
            AnnSearchRequest(
                data=[sparse],
                anns_field="sparse",
                param={"metric_type": "IP", "params": {}},
                limit=limit,
                expr=expr,
            ),
        ]
        res = self.client.hybrid_search(
            collection_name=self.s.milvus_collection,
            reqs=reqs,
            ranker=RRFRanker(self.s.rrf_k),
            limit=limit,
            output_fields=["parent_id", "doc_id", "text"],
        )
        return [
            {"id": h["id"], "score": float(h["distance"]), "text": h["entity"]["text"]}
            for h in (res[0] if res else [])
        ]

    def delete_doc(self, doc_id: str):
        self.client.delete(collection_name=self.s.milvus_collection, filter=f'doc_id == "{doc_id}"')
