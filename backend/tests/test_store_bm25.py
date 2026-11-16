from app.ingestion.chunker import chunk_markdown
from app.retrieval.bm25 import BM25Index
from app.store import MetaStore


def _ingest(store, doc_id, ns, md):
    parents, children = chunk_markdown(md, child_max=100)
    store.add_document(doc_id, f"{doc_id}.md", ns)
    store.insert_chunks(doc_id, ns, parents, children)
    return parents, children


def test_meta_store_roundtrip_and_delete(tmp_path):
    store = MetaStore(str(tmp_path / "m.db"))
    parents, children = _ingest(store, "d1", "ns", "# 标题\n\n这是一段关于差旅报销的说明。")
    got = store.get_children([c["child_id"] for c in children])
    assert len(got) == len(children)
    assert store.get_filenames(["d1"]) == {"d1": "d1.md"}
    store.set_status("d1", "ready", n_chunks=len(children))
    assert store.list_documents("ns")[0]["status"] == "ready"
    store.delete_document("d1")
    assert store.iter_children("ns") == [] and store.get_document("d1") is None


def test_bm25_search_and_invalidate(tmp_path):
    store = MetaStore(str(tmp_path / "m.db"))
    index = BM25Index(store)
    assert index.search("ns", "报销", 5) == []  # 空库
    _ingest(store, "d1", "ns", "# 报销\n\n差旅报销需要提交发票。\n\n# 请假\n\n年假需要提前一周申请。\n\n# 采购\n\n采购流程需要三家比价。")
    index.invalidate("ns")
    hits = index.search("ns", "年假 申请", 3)
    assert hits
    top = store.get_children([hits[0]["id"]])[hits[0]["id"]]
    assert "年假" in top["text"]
