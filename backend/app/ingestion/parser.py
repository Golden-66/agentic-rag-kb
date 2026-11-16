"""文档解析: PDF -> Markdown。

优先使用 MinerU (版式分析 + 表格识别 + 扫描件 OCR) 的命令行, 失败时可回退到 pymupdf4llm。
MinerU 的输出目录结构随版本变化, 这里递归查找生成的 .md 文件, 不依赖固定子目录名。
"""
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_EXTS = {".md", ".markdown", ".txt"}
SUPPORTED_EXTS = TEXT_EXTS | {".pdf"}


def _parse_mineru(path: Path, workdir: Path, s) -> tuple[str, Path]:
    out = workdir / "mineru_out"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [s.mineru_bin, "-p", str(path), "-o", str(out)]
    logger.info("run MinerU: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, timeout=s.mineru_timeout)
    mds = sorted(out.rglob("*.md"), key=lambda p: p.stat().st_size, reverse=True)
    if not mds:
        raise RuntimeError("MinerU 没有产出 markdown 文件")
    md_path = mds[0]
    return md_path.read_text(encoding="utf-8", errors="ignore"), md_path.parent


def _parse_pymupdf(path: Path) -> str:
    import pymupdf4llm  # 延迟导入

    return pymupdf4llm.to_markdown(str(path))


def parse_to_markdown(path: Path, workdir: Path, s) -> tuple[str, Path | None]:
    """返回 (markdown 文本, 图片所在目录 或 None)。"""
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="ignore"), path.parent
    if ext != ".pdf":
        raise ValueError(f"不支持的文件类型: {ext}")

    if s.parser_backend == "mineru":
        try:
            return _parse_mineru(path, workdir, s)
        except Exception as e:  # noqa: BLE001
            if not s.parser_fallback:
                raise
            logger.warning("MinerU 解析失败 (%s), 回退到 pymupdf4llm (不支持扫描件 OCR)", e)
    return _parse_pymupdf(path), None
