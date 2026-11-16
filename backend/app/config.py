from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- service ----
    api_key: str = ""  # 为空则不启用 X-API-Key 鉴权
    max_upload_mb: int = 50
    preload_models: bool = True
    data_dir: str = "./data"

    # ---- LLM (任意 OpenAI 兼容接口) ----
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # ---- VLM (文档图片描述, 可选) ----
    vlm_enabled: bool = False
    vlm_base_url: str = ""  # 为空则复用 llm_base_url
    vlm_api_key: str = ""  # 为空则复用 llm_api_key
    vlm_model: str = "qwen-vl-max"
    max_images: int = 20

    # ---- 本地模型 ----
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_fp16: bool = True
    embedding_dim: int = 1024
    embed_batch_size: int = 16

    # ---- Milvus / Redis ----
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "kb_chunks"
    redis_url: str = "redis://localhost:6379/0"

    # ---- 文档解析 ----
    parser_backend: str = "mineru"  # mineru | pymupdf
    parser_fallback: bool = True  # MinerU 失败时回退到 pymupdf4llm
    mineru_bin: str = "mineru"
    mineru_timeout: int = 1800

    # ---- 分块 (按字符数, 对中文更直观) ----
    parent_max_chars: int = 1500
    child_max_chars: int = 400
    child_overlap: int = 60

    # ---- 检索 / Agent ----
    recall_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60
    rerank_fast_threshold: float = 0.5  # 重排最高分超过该值则跳过 LLM 评估
    max_retries: int = 2
    multi_query_n: int = 2
    enable_hyde: bool = True
    max_sub_questions: int = 3

    # ---- 缓存 ----
    query_cache_ttl: int = 3600
    embedding_cache_ttl: int = 7 * 24 * 3600
    history_turns: int = 5

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
