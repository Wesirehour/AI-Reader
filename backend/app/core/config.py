import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    app_name: str = os.getenv("APP_NAME", "AI Reader MVP")
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    cors_origins: list = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]

    # Stage-4 model routing + config center (OpenAI SDK compatible gateway)
    llm_provider: str = os.getenv("LLM_PROVIDER", "scnet-openai")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.scnet.cn/api/llm")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_default_model: str = os.getenv("LLM_DEFAULT_MODEL", "DeepSeek-V3.2")
    llm_model_list: list = [
        x.strip()
        for x in os.getenv(
            "LLM_MODEL_LIST",
            "MiniMax-M2.5,DeepSeek-V3.2,Qwen3-235B-A22B-Thinking-2507,DeepSeek-R1-0528",
        ).split(",")
        if x.strip()
    ]
    llm_timeout_sec: int = int(os.getenv("LLM_TIMEOUT_SEC", "45"))

    mineru_cmd: str = os.getenv("MINERU_CMD", "mineru")
    mineru_output_dir: str = os.getenv("MINERU_OUTPUT_DIR", "./mineru_output")
    auto_process_on_upload: bool = os.getenv("AUTO_PROCESS_ON_UPLOAD", "false").lower() in ("1", "true", "yes")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    mineru_parse_mode: str = os.getenv("MINERU_PARSE_MODE", "api_file")
    mineru_api_base_url: str = os.getenv("MINERU_API_BASE_URL", "https://mineru.net/api/v1/agent")
    mineru_api_timeout_sec: int = int(os.getenv("MINERU_API_TIMEOUT_SEC", "300"))
    mineru_api_poll_interval_sec: int = int(os.getenv("MINERU_API_POLL_INTERVAL_SEC", "3"))
    mineru_language: str = os.getenv("MINERU_LANGUAGE", "ch")
    mineru_enable_table: bool = os.getenv("MINERU_ENABLE_TABLE", "false").lower() in ("1", "true", "yes")
    mineru_enable_formula: bool = os.getenv("MINERU_ENABLE_FORMULA", "true").lower() in ("1", "true", "yes")
    mineru_is_ocr: bool = os.getenv("MINERU_IS_OCR", "false").lower() in ("1", "true", "yes")
    mineru_loader_mode: str = os.getenv("MINERU_LOADER_MODE", "flash")
    embeddings_model_name: str = os.getenv("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
    embeddings_device: str = os.getenv("EMBEDDINGS_DEVICE", "cpu")
    hf_endpoint: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    allow_insecure_dev: bool = os.getenv("ALLOW_INSECURE_DEV", "false").lower() in ("1", "true", "yes")

    # Retrieval switches (hybrid BM25+dense, optional rerank)
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")  # dense | bm25 | hybrid
    retrieval_recall_topn: int = int(os.getenv("RETRIEVAL_RECALL_TOPN", "30"))
    reranker_model: str = os.getenv("RERANKER_MODEL", "")  # e.g. BAAI/bge-reranker-base

    def validate(self):
        errors = []
        warnings = []
        weak_keys = {"", "change-me-in-production", "secret", "changeme"}
        if self.secret_key in weak_keys:
            msg = "SECRET_KEY is unset or using a weak default"
            (warnings if self.allow_insecure_dev else errors).append(msg)
        if not self.llm_api_key:
            warnings.append("LLM_API_KEY is empty; /api/chat will fail at runtime")
        if self.retrieval_mode not in ("dense", "bm25", "hybrid"):
            warnings.append(
                "RETRIEVAL_MODE=%s is not one of dense|bm25|hybrid; falling back to hybrid"
                % self.retrieval_mode
            )
        if errors:
            raise RuntimeError(
                "Insecure startup configuration: %s. "
                "Fix it, or set ALLOW_INSECURE_DEV=true for local dev only."
                % "; ".join(errors)
            )
        for w in warnings:
            print("[config] WARN:", w)


settings = Settings()
settings.validate()
