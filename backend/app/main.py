import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent.graph import build_graph
from .agent.service import AgentService
from .cache import Cache
from .config import get_settings
from .ingestion.parser import SUPPORTED_EXTS
from .ingestion.pipeline import IngestPipeline
from .llm import LLMClient
from .retrieval.bm25 import BM25Index
from .retrieval.embedder import Embedder
from .retrieval.milvus_store import MilvusStore
from .retrieval.reranker import Reranker
from .retrieval.retriever import Retriever
from .schemas import NAMESPACE_RE, QueryRequest
from .store import MetaStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    s.data_path.mkdir(parents=True, exist_ok=True)
    (s.data_path / "uploads").mkdir(exist_ok=True)

    meta = MetaStore(str(s.data_path / "meta.db"))
    cache = Cache(s)
    llm = LLMClient(s)
    embedder, reranker = Embedder(s, cache), Reranker(s)
    milvus = MilvusStore(s)
    await asyncio.to_thread(milvus.connect)
    bm25 = BM25Index(meta)
    retriever = Retriever(s, embedder, milvus, bm25, meta)

    app.state.meta, app.state.milvus, app.state.bm25, app.state.cache = meta, milvus, bm25, cache
    app.state.agent = AgentService(s, cache, llm, build_graph(s, llm, retriever, reranker))
    app.state.pipeline = IngestPipeline(s, meta, llm, embedder, milvus, bm25, cache)

    if s.preload_models:  # 后台预热模型, 避免第一个请求等待加载

        def _warm():
            try:
                embedder.warmup()
                reranker.warmup()
                logger.info("models warmed up")
            except Exception:  # noqa: BLE001
                logger.exception("model warmup failed")

        asyncio.create_task(asyncio.to_thread(_warm))
    yield


app = FastAPI(title="Agentic RAG Knowledge Base", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def auth(x_api_key: str | None = Header(default=None)):
    key = get_settings().api_key
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="invalid api key")


def _check_namespace(ns: str) -> str:
    if not NAMESPACE_RE.match(ns):
        raise HTTPException(status_code=422, detail="namespace 只允许字母/数字/下划线/连字符, 最长 64")
    return ns


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------- documents ----------------
@app.post("/api/v1/documents", dependencies=[Depends(auth)])
async def upload_document(
    background: BackgroundTasks, file: UploadFile = File(...), namespace: str = Form("default")
):
    s = get_settings()
    namespace = _check_namespace(namespace)
    filename = file.filename or "unnamed"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=415, detail=f"仅支持 {sorted(SUPPORTED_EXTS)}")
    content = await file.read()
    if len(content) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {s.max_upload_mb}MB")

    doc_id = uuid.uuid4().hex
    path = s.data_path / "uploads" / f"{doc_id}{ext}"
    await asyncio.to_thread(path.write_bytes, content)
    app.state.meta.add_document(doc_id, filename, namespace)
    background.add_task(app.state.pipeline.run, doc_id, path, filename, namespace)  # 上传即返回, 异步处理
    return {"doc_id": doc_id, "filename": filename, "namespace": namespace, "status": "processing"}


@app.get("/api/v1/documents", dependencies=[Depends(auth)])
async def list_documents(namespace: str | None = None):
    if namespace:
        _check_namespace(namespace)
    return {"documents": await asyncio.to_thread(app.state.meta.list_documents, namespace)}


@app.delete("/api/v1/documents/{doc_id}", dependencies=[Depends(auth)])
async def delete_document(doc_id: str):
    doc = app.state.meta.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    await asyncio.to_thread(app.state.milvus.delete_doc, doc_id)
    await asyncio.to_thread(app.state.meta.delete_document, doc_id)
    app.state.bm25.invalidate(doc["namespace"])
    await app.state.cache.bump_ns(doc["namespace"])
    for p in (get_settings().data_path / "uploads").glob(f"{doc_id}.*"):
        p.unlink(missing_ok=True)
    return {"deleted": doc_id}


# ---------------- query ----------------
@app.post("/api/v1/query", dependencies=[Depends(auth)])
async def query(req: QueryRequest):
    return await app.state.agent.answer(req.query, req.namespace, req.session_id)


@app.post("/api/v1/query/stream", dependencies=[Depends(auth)])
async def query_stream(req: QueryRequest):
    async def gen():
        try:
            async for ev in app.state.agent.stream(req.query, req.namespace, req.session_id):
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            logger.exception("stream failed")
            yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/api/v1/retrieve", dependencies=[Depends(auth)])
async def retrieve(req: QueryRequest):
    """只检索不生成, 返回重排后的上下文 (供评测脚本使用)。"""
    return await app.state.agent.retrieve_only(req.query, req.namespace)
