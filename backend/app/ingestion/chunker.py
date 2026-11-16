"""标题层级感知 + 表格保护 + 父子分块的 Markdown 切分器 (纯 Python, 无重依赖)。

策略:
1. 按 Markdown 标题切成章节, 每个章节携带完整标题路径 ("A > B > C");
2. 章节内按段落/表格块装包成「父块」(用于给 LLM 提供完整上下文);
3. 父块再切成更小的「子块」(用于向量/稀疏检索), 子块文本前拼接标题路径;
4. 表格作为整体块处理, 超长表格按行拆分且每片重复表头, 避免表格结构被破坏。
"""
from __future__ import annotations

import re
import uuid

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def split_sections(md: str) -> list[tuple[str, str]]:
    """返回 [(标题路径, 正文)], 忽略代码块内的 # 行。"""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    heading = ""
    in_fence = False

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((heading, text))

    for line in md.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            flush()
            buf.clear()
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            heading = " > ".join(t for _, t in stack)
        else:
            buf.append(line)
    flush()
    return sections


def to_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def is_table(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    return len(lines) >= 2 and all(ln.strip().startswith("|") for ln in lines)


def _split_table(block: str, max_chars: int) -> list[str]:
    lines = block.splitlines()
    if len(lines) >= 2 and TABLE_SEP_RE.match(lines[1]):
        header, rows = lines[:2], lines[2:]
    else:
        header, rows = [], lines
    head_len = len("\n".join(header))
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = head_len
    for row in rows:
        if cur and cur_len + len(row) + 1 > max_chars:
            pieces.append("\n".join(header + cur))
            cur, cur_len = [], head_len
        cur.append(row)
        cur_len += len(row) + 1
    if cur:
        pieces.append("\n".join(header + cur))
    return pieces or [block]


def _split_text(block: str, max_chars: int, overlap: int) -> list[str]:
    overlap = min(overlap, max_chars // 2)
    out: list[str] = []
    cur = ""
    for piece in (p for p in SENT_SPLIT_RE.split(block) if p):
        while len(piece) > max_chars:  # 单句超长, 硬切
            if cur:
                out.append(cur)
                cur = ""
            out.append(piece[:max_chars])
            piece = piece[max_chars - overlap :]
        if cur and len(cur) + len(piece) > max_chars:
            out.append(cur)
            cur = (cur[-overlap:] if overlap else "") + piece
        else:
            cur += piece
    if cur.strip():
        out.append(cur)
    return [o.strip() for o in out if o.strip()]


def split_long_block(block: str, max_chars: int, overlap: int) -> list[str]:
    return _split_table(block, max_chars) if is_table(block) else _split_text(block, max_chars, overlap)


def pack_blocks(blocks: list[str], max_chars: int, overlap: int = 0) -> list[str]:
    out: list[str] = []
    cur = ""
    for b in blocks:
        if len(b) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(split_long_block(b, max_chars, overlap))
            continue
        if cur and len(cur) + 2 + len(b) > max_chars:
            out.append(cur)
            cur = b
        else:
            cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        out.append(cur)
    return out


def chunk_markdown(
    md: str, parent_max: int = 1500, child_max: int = 400, overlap: int = 60
) -> tuple[list[dict], list[dict]]:
    parents: list[dict] = []
    children: list[dict] = []
    for heading, body in split_sections(md):
        for ptext in pack_blocks(to_blocks(body), parent_max, overlap):
            pid = uuid.uuid4().hex
            parents.append({"parent_id": pid, "heading": heading, "text": ptext})
            for piece in pack_blocks(to_blocks(ptext), child_max, overlap):
                ctext = f"{heading}\n{piece}" if heading else piece
                children.append({"child_id": uuid.uuid4().hex, "parent_id": pid, "text": ctext})
    return parents, children
