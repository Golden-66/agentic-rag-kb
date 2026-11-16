import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, filename TEXT, namespace TEXT, status TEXT,
    error TEXT, n_chunks INTEGER DEFAULT 0, created_at REAL
);
CREATE TABLE IF NOT EXISTS parents (
    parent_id TEXT PRIMARY KEY, doc_id TEXT, namespace TEXT, heading TEXT, text TEXT
);
CREATE TABLE IF NOT EXISTS children (
    child_id TEXT PRIMARY KEY, parent_id TEXT, doc_id TEXT, namespace TEXT, text TEXT
);
CREATE INDEX IF NOT EXISTS idx_parents_doc ON parents(doc_id);
CREATE INDEX IF NOT EXISTS idx_children_doc ON children(doc_id);
CREATE INDEX IF NOT EXISTS idx_children_ns ON children(namespace);
"""


class MetaStore:
    """SQLite 元数据库: 文档状态 + 父/子块文本 (子块用于检索, 父块作为生成上下文)。"""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _exec(self, sql: str, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ---- documents ----
    def add_document(self, doc_id: str, filename: str, namespace: str):
        self._exec(
            "INSERT INTO documents(doc_id, filename, namespace, status, created_at) VALUES (?,?,?,?,?)",
            (doc_id, filename, namespace, "processing", time.time()),
        )

    def set_status(self, doc_id: str, status: str, error: str | None = None, n_chunks: int | None = None):
        if n_chunks is None:
            self._exec("UPDATE documents SET status=?, error=? WHERE doc_id=?", (status, error, doc_id))
        else:
            self._exec(
                "UPDATE documents SET status=?, error=?, n_chunks=? WHERE doc_id=?",
                (status, error, n_chunks, doc_id),
            )

    def list_documents(self, namespace: str | None = None) -> list[dict]:
        if namespace:
            rows = self._query("SELECT * FROM documents WHERE namespace=? ORDER BY created_at DESC", (namespace,))
        else:
            rows = self._query("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> dict | None:
        rows = self._query("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
        return dict(rows[0]) if rows else None

    def delete_document(self, doc_id: str):
        with self._lock:
            for t in ("children", "parents", "documents"):
                self._conn.execute(f"DELETE FROM {t} WHERE doc_id=?", (doc_id,))
            self._conn.commit()

    # ---- chunks ----
    def insert_chunks(self, doc_id: str, namespace: str, parents: list[dict], children: list[dict]):
        with self._lock:
            self._conn.executemany(
                "INSERT INTO parents(parent_id, doc_id, namespace, heading, text) VALUES (?,?,?,?,?)",
                [(p["parent_id"], doc_id, namespace, p["heading"], p["text"]) for p in parents],
            )
            self._conn.executemany(
                "INSERT INTO children(child_id, parent_id, doc_id, namespace, text) VALUES (?,?,?,?,?)",
                [(c["child_id"], c["parent_id"], doc_id, namespace, c["text"]) for c in children],
            )
            self._conn.commit()

    @staticmethod
    def _batches(items: list[str], size: int = 500):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _get_by_ids(self, table: str, key: str, ids: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for batch in self._batches(list(dict.fromkeys(ids))):
            marks = ",".join("?" * len(batch))
            for r in self._query(f"SELECT * FROM {table} WHERE {key} IN ({marks})", batch):
                out[r[key]] = dict(r)
        return out

    def get_children(self, ids: list[str]) -> dict[str, dict]:
        return self._get_by_ids("children", "child_id", ids)

    def get_parents(self, ids: list[str]) -> dict[str, dict]:
        return self._get_by_ids("parents", "parent_id", ids)

    def get_filenames(self, doc_ids: list[str]) -> dict[str, str]:
        docs = self._get_by_ids("documents", "doc_id", doc_ids)
        return {k: v["filename"] for k, v in docs.items()}

    def iter_children(self, namespace: str) -> list[tuple[str, str]]:
        rows = self._query("SELECT child_id, text FROM children WHERE namespace=?", (namespace,))
        return [(r["child_id"], r["text"]) for r in rows]
