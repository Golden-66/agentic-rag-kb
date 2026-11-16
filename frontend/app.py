import json
import os
import uuid

import httpx
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
HEADERS = {"X-API-Key": os.getenv("API_KEY", "")} if os.getenv("API_KEY") else {}

st.set_page_config(page_title="企业智能知识库", page_icon="📚", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []


def sse_events(resp: httpx.Response):
    """解析 SSE: 逐个产出 (event, data)。"""
    event, data = "message", []
    for line in resp.iter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and data:
            yield event, json.loads("\n".join(data))
            event, data = "message", []


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📎 引用来源 ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['index']}] {s['filename']}** · {s['heading'] or '正文'} · 相关度 {s['score']}")
            st.caption(s["text"])


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("知识库")
    namespace = st.text_input("命名空间", value="default", help="不同命名空间的知识完全隔离")
    files = st.file_uploader("上传文档 (PDF / MD / TXT)", type=["pdf", "md", "markdown", "txt"], accept_multiple_files=True)
    if files and st.button("上传并入库", use_container_width=True):
        for f in files:
            try:
                r = httpx.post(
                    f"{API}/api/v1/documents",
                    files={"file": (f.name, f.getvalue())},
                    data={"namespace": namespace},
                    headers=HEADERS,
                    timeout=120,
                )
                r.raise_for_status()
                st.success(f"{f.name} 已提交处理")
            except Exception as e:  # noqa: BLE001
                st.error(f"{f.name} 上传失败: {e}")

    st.subheader("文档列表")
    if st.button("刷新", use_container_width=True):
        st.rerun()
    try:
        docs = httpx.get(f"{API}/api/v1/documents", params={"namespace": namespace}, headers=HEADERS, timeout=10).json()["documents"]
    except Exception as e:  # noqa: BLE001
        docs = []
        st.warning(f"后端不可用: {e}")
    icons = {"ready": "✅", "processing": "⏳", "failed": "❌"}
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c1.write(f"{icons.get(d['status'], '•')} {d['filename']}")
        if d["status"] == "failed":
            c1.caption(d.get("error") or "")
        if c2.button("🗑", key=d["doc_id"]):
            httpx.delete(f"{API}/api/v1/documents/{d['doc_id']}", headers=HEADERS, timeout=30)
            st.rerun()

    if st.button("新会话", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()

# ---------------- chat ----------------
st.title("📚 企业级智能知识库")
st.caption("Agentic RAG · 混合检索 · 重排序 · 子问题拆解 · 迭代纠错")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources", []))

if prompt := st.chat_input("向知识库提问…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_slot, answer_slot = st.empty(), st.empty()
        text, sources, done = "", [], {}
        try:
            with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
                with client.stream(
                    "POST",
                    f"{API}/api/v1/query/stream",
                    json={"query": prompt, "namespace": namespace, "session_id": st.session_state.session_id},
                    headers=HEADERS,
                ) as resp:
                    resp.raise_for_status()
                    for event, data in sse_events(resp):
                        if event == "status":
                            status_slot.caption(f"⚙️ {data['label']} {data.get('detail', '')}")
                        elif event == "token":
                            text += data["text"]
                            answer_slot.markdown(text + "▌")
                        elif event == "sources":
                            sources = data["sources"]
                        elif event == "done":
                            done = data
                        elif event == "error":
                            st.error(data["message"])
        except Exception as e:  # noqa: BLE001
            st.error(f"请求失败: {e}")

        answer_slot.markdown(text)
        if done:
            tag = "缓存命中 · " if done.get("cached") else ""
            status_slot.caption(f"✅ {tag}首字 {done.get('first_token_ms')} ms · 总耗时 {done.get('latency_ms')} ms")
        else:
            status_slot.empty()
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})
