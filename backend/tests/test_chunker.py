from app.ingestion.chunker import chunk_markdown, is_table, split_sections


def test_heading_path():
    md = "# A\n\n简介。\n\n## B\n\n正文内容。\n\n### C\n\n更深一层。\n"
    heads = [h for h, _ in split_sections(md)]
    assert heads == ["A", "A > B", "A > B > C"]


def test_code_fence_heading_ignored():
    md = "# A\n\n```\n# not a heading\n```\n\n文字\n"
    assert [h for h, _ in split_sections(md)] == ["A"]


def test_parent_child_links_and_heading_prefix():
    md = "# 报销制度\n\n" + "差旅费用需要提前申请。" * 80
    parents, children = chunk_markdown(md, parent_max=500, child_max=120, overlap=20)
    assert parents and children
    pids = {p["parent_id"] for p in parents}
    assert all(c["parent_id"] in pids for c in children)
    assert all(c["text"].startswith("报销制度\n") for c in children)
    assert all(len(p["text"]) <= 500 + 20 for p in parents)


def test_small_table_kept_whole():
    table = "| 项目 | 金额 |\n| --- | --- |\n| 机票 | 800 |\n| 酒店 | 400 |"
    parents, children = chunk_markdown(f"# 标准\n\n{table}\n", child_max=400)
    assert is_table(table)
    assert any(table in c["text"] for c in children)


def test_long_table_split_repeats_header():
    header = "| 项目 | 金额 |\n| --- | --- |"
    rows = "\n".join(f"| 项目{i} | {i * 10} |" for i in range(120))
    parents, children = chunk_markdown(f"# 标准\n\n{header}\n{rows}\n", parent_max=300, child_max=300)
    assert len(children) > 1
    assert all("| 项目 | 金额 |" in c["text"] for c in children)
