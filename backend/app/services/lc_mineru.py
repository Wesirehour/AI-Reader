import json
import os

from app.core.config import settings
from app.core.vector_store_config import vector_store_config
from app.services.structured_chunking import structured_chunks_from_markdown

if settings.hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

_FAISS_CACHE = {}


def _require_faiss_stack():
    try:
        from langchain_mineru import MinerULoader  # noqa: F401
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: F401
        from langchain_community.vectorstores import FAISS  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "LangChain MinerU stack is not installed. "
            "Please install dependencies from requirements.txt"
        ) from exc


def _require_milvus_stack():
    try:
        from langchain_mineru import MinerULoader  # noqa: F401
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: F401
        from langchain_community.vectorstores import Milvus  # noqa: F401
        import pymilvus  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Milvus stack is not installed. "
            "Please install dependencies from requirements.txt"
        ) from exc


def get_hf_embeddings(model_name="all-MiniLM-L6-v2", device="cpu"):
    _require_faiss_stack()
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )


def _strategy():
    mode = (os.getenv("CHUNK_STRATEGY") or "structured").strip().lower()
    return mode if mode in ("structured", "recursive") else "structured"


def _build_chunks(docs, chunk_size, chunk_overlap, tokenizer_model_name):
    """Produce LangChain Documents using the configured chunking strategy.

    Strategy "structured" (default): heading-aware, token-bounded, preserves
    tables/formulas/code, emits child chunks that carry parent_text in metadata.

    Strategy "recursive": legacy character-based RecursiveCharacterTextSplitter.
    Kept as a fallback and for A/B comparison via eval_retrieval.py.
    """
    from langchain_core.documents import Document as LCDocument
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    strategy = _strategy()
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        for i, d in enumerate(chunks):
            meta = dict(d.metadata or {})
            meta["chunk_index"] = i
            d.metadata = meta
        return chunks

    # Structured: process each loader document, preserving per-doc metadata
    # (typically `page`), and reassign chunk_index globally at the end.
    lc_chunks = []
    parent_offset = 0
    for d in docs:
        md = d.page_content or ""
        base_meta = dict(d.metadata or {})
        produced = structured_chunks_from_markdown(
            md,
            tokenizer_model_name=tokenizer_model_name,
            child_max_tokens=int(os.getenv("CHILD_MAX_TOKENS", "220")),
            child_overlap_tokens=int(os.getenv("CHILD_OVERLAP_TOKENS", "40")),
            parent_max_tokens=int(os.getenv("PARENT_MAX_TOKENS", "900")),
        )
        local_parents = 0
        for c in produced:
            meta = dict(base_meta)
            meta.update(c["metadata"])
            meta["parent_id"] = int(meta.get("parent_id", 0)) + parent_offset
            local_parents = max(local_parents, int(meta["parent_id"]) - parent_offset + 1)
            lc_chunks.append(LCDocument(page_content=c["text"], metadata=meta))
        parent_offset += local_parents

    for i, d in enumerate(lc_chunks):
        d.metadata["chunk_index"] = i
    return lc_chunks


def build_faiss_from_pdf(
    pdf_path,
    output_dir,
    chunk_size=1200,
    chunk_overlap=200,
    mineru_mode="flash",
    embeddings_model="all-MiniLM-L6-v2",
    embeddings_device="cpu",
):
    _require_faiss_stack()
    from langchain_mineru import MinerULoader
    from langchain_community.vectorstores import FAISS

    os.makedirs(output_dir, exist_ok=True)

    loader = MinerULoader(source=pdf_path, mode=mineru_mode)
    docs = loader.load()
    chunks = _build_chunks(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer_model_name="sentence-transformers/" + embeddings_model
        if "/" not in embeddings_model
        else embeddings_model,
    )

    embeddings = get_hf_embeddings(model_name=embeddings_model, device=embeddings_device)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(output_dir)

    chunks_json_path = os.path.join(output_dir, "chunks.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {"chunk_index": d.metadata.get("chunk_index", i), "text": d.page_content, "metadata": d.metadata or {}}
                for i, d in enumerate(chunks)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {"chunk_count": len(chunks), "index_dir": output_dir, "chunks_json_path": chunks_json_path}


def load_faiss_index(index_dir, embeddings_model="all-MiniLM-L6-v2", embeddings_device="cpu", use_cache=None):
    _require_faiss_stack()
    from langchain_community.vectorstores import FAISS

    if use_cache is None:
        use_cache = vector_store_config.faiss_cache_enabled
    cache_key = (index_dir, embeddings_model, embeddings_device)
    if use_cache and cache_key in _FAISS_CACHE:
        return _FAISS_CACHE[cache_key]

    embeddings = get_hf_embeddings(model_name=embeddings_model, device=embeddings_device)
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    if use_cache:
        _FAISS_CACHE[cache_key] = vs
    return vs


def _sanitize_collection_name(name):
    safe = "".join([c if c.isalnum() else "_" for c in (name or "")]).strip("_").lower()
    return safe[:200] if safe else "rag_doc_default"


def build_milvus_from_pdf(
    pdf_path,
    collection_name,
    chunk_size=1200,
    chunk_overlap=200,
    mineru_mode="flash",
    embeddings_model="all-MiniLM-L6-v2",
    embeddings_device="cpu",
    connection_args=None,
    index_params=None,
):
    _require_milvus_stack()
    from langchain_mineru import MinerULoader
    from langchain_community.vectorstores import Milvus

    connection_args = connection_args or {}
    index_params = index_params or {}

    loader = MinerULoader(source=pdf_path, mode=mineru_mode)
    docs = loader.load()
    chunks = _build_chunks(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer_model_name="sentence-transformers/" + embeddings_model
        if "/" not in embeddings_model
        else embeddings_model,
    )

    embeddings = get_hf_embeddings(model_name=embeddings_model, device=embeddings_device)
    Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=_sanitize_collection_name(collection_name),
        connection_args=connection_args,
        index_params=index_params,
        drop_old=True,
        auto_id=True,
    )

    return {
        "chunk_count": len(chunks),
        "chunks": [
            {"chunk_index": d.metadata.get("chunk_index", i), "text": d.page_content, "metadata": d.metadata or {}}
            for i, d in enumerate(chunks)
        ],
    }


def load_milvus_index(
    collection_name,
    embeddings_model="all-MiniLM-L6-v2",
    embeddings_device="cpu",
    connection_args=None,
    search_params=None,
):
    _require_milvus_stack()
    from langchain_community.vectorstores import Milvus

    embeddings = get_hf_embeddings(model_name=embeddings_model, device=embeddings_device)
    return Milvus(
        embedding_function=embeddings,
        collection_name=_sanitize_collection_name(collection_name),
        connection_args=connection_args or {},
        search_params=search_params or {},
        auto_id=True,
    )
