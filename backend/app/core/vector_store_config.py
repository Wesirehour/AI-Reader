import json
import os
from pathlib import Path


class VectorStoreConfig:
    def __init__(self):
        config_path = Path(__file__).resolve().parents[2] / "config" / "vector_store.json"
        data = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        backend = str(data.get("backend", "faiss")).strip().lower()
        if backend not in ("faiss", "milvus"):
            backend = "faiss"
        env_backend = os.getenv("VECTOR_STORE_BACKEND", "").strip().lower()
        if env_backend in ("faiss", "milvus"):
            backend = env_backend
        self.backend = backend

        faiss = data.get("faiss") or {}
        self.faiss_cache_enabled = bool(faiss.get("cache_enabled", True))

        milvus = data.get("milvus") or {}
        self.milvus_uri = str(milvus.get("uri", "http://localhost:19530")).strip()
        self.milvus_token = str(milvus.get("token", "")).strip()
        self.milvus_user = str(milvus.get("user", "")).strip()
        self.milvus_password = str(milvus.get("password", "")).strip()
        self.milvus_db_name = str(milvus.get("db_name", "default")).strip() or "default"
        self.milvus_collection_prefix = str(milvus.get("collection_prefix", "rag_doc_")).strip() or "rag_doc_"
        self.milvus_metric_type = str(milvus.get("metric_type", "COSINE")).strip().upper() or "COSINE"
        self.milvus_index_params = milvus.get("index_params") or {
            "index_type": "AUTOINDEX",
            "metric_type": self.milvus_metric_type,
            "params": {},
        }
        self.milvus_search_params = milvus.get("search_params") or {"params": {"nprobe": 10}}

    @property
    def use_faiss(self):
        return self.backend == "faiss"

    @property
    def use_milvus(self):
        return self.backend == "milvus"

    def milvus_connection_args(self):
        args = {"uri": self.milvus_uri, "db_name": self.milvus_db_name}
        if self.milvus_token:
            args["token"] = self.milvus_token
        elif self.milvus_user and self.milvus_password:
            args["user"] = self.milvus_user
            args["password"] = self.milvus_password
        return args


vector_store_config = VectorStoreConfig()
