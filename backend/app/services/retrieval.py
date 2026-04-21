import json
import os
from datetime import datetime

from app.core.vector_store_config import vector_store_config
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.hybrid import bm25_search, rrf_fuse
from app.services.lc_mineru import (
    build_faiss_from_pdf,
    build_milvus_from_pdf,
    load_faiss_index,
    load_milvus_index,
)
from app.services.mineru_parser import compute_file_sha256
from app.services.rerank import rerank as cross_rerank


def _retrieval_mode():
    return os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower() or "hybrid"


def _recall_topn():
    try:
        return max(5, int(os.getenv("RETRIEVAL_RECALL_TOPN", "30")))
    except Exception:
        return 30


def _doc_index_dir(mineru_output_dir, file_hash):
    return os.path.join(mineru_output_dir, "lc_cache", file_hash, "faiss")


def _doc_chunks_json(mineru_output_dir, file_hash):
    return os.path.join(mineru_output_dir, "lc_cache", file_hash, "faiss", "chunks.json")


def _doc_milvus_chunks_json(mineru_output_dir, file_hash):
    return os.path.join(mineru_output_dir, "lc_cache", file_hash, "milvus", "chunks.json")


def _doc_milvus_collection(file_hash):
    return "%s%s" % (vector_store_config.milvus_collection_prefix, file_hash)


def _extract_page_from_meta(meta):
    if not isinstance(meta, dict):
        return None
    candidates = ["page", "page_num", "page_number", "source_page", "pageno"]
    for key in candidates:
        val = meta.get(key)
        if val is None:
            continue
        try:
            n = int(val)
            if n >= 0:
                return n
        except Exception:
            continue
    return None


def process_document(
    db,
    document,
    mineru_output_dir,
    mineru_cmd,  # kept for API compatibility, not used in LangChain path
    chunk_size,
    chunk_overlap,
    parse_mode,  # kept for API compatibility, not used in LangChain path
    api_base_url,  # kept for API compatibility, not used in LangChain path
    api_timeout_sec,  # kept for API compatibility, not used in LangChain path
    api_poll_interval_sec,  # kept for API compatibility, not used in LangChain path
    language,  # kept for API compatibility, not used in LangChain path
    enable_table,  # kept for API compatibility, not used in LangChain path
    enable_formula,  # kept for API compatibility, not used in LangChain path
    is_ocr,  # kept for API compatibility, not used in LangChain path
    mineru_loader_mode,
    embeddings_model_name,
    embeddings_device,
):
    document.process_status = "processing"
    document.process_error = ""
    if not document.file_hash:
        document.file_hash = compute_file_sha256(document.file_path)
    db.add(document)
    db.commit()

    file_hash = document.file_hash
    index_dir = _doc_index_dir(mineru_output_dir, file_hash)
    chunks_json_path = _doc_chunks_json(mineru_output_dir, file_hash)
    milvus_chunks_json = _doc_milvus_chunks_json(mineru_output_dir, file_hash)

    try:
        if vector_store_config.use_milvus:
            collection_name = _doc_milvus_collection(file_hash)
            result = build_milvus_from_pdf(
                pdf_path=document.file_path,
                collection_name=collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                mineru_mode=mineru_loader_mode,
                embeddings_model=embeddings_model_name,
                embeddings_device=embeddings_device,
                connection_args=vector_store_config.milvus_connection_args(),
                index_params=vector_store_config.milvus_index_params,
            )
            chunk_count = int(result["chunk_count"])
            os.makedirs(os.path.dirname(milvus_chunks_json), exist_ok=True)
            with open(milvus_chunks_json, "w", encoding="utf-8") as f:
                json.dump(result.get("chunks") or [], f, ensure_ascii=False, indent=2)
            chunks_source = milvus_chunks_json
        else:
            if not os.path.exists(os.path.join(index_dir, "index.faiss")):
                result = build_faiss_from_pdf(
                    pdf_path=document.file_path,
                    output_dir=index_dir,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    mineru_mode=mineru_loader_mode,
                    embeddings_model=embeddings_model_name,
                    embeddings_device=embeddings_device,
                )
                chunk_count = int(result["chunk_count"])
            else:
                if os.path.exists(chunks_json_path):
                    with open(chunks_json_path, "r", encoding="utf-8") as f:
                        chunk_count = len(json.load(f))
                else:
                    chunk_count = 0
            chunks_source = chunks_json_path

        db.query(Chunk).filter(Chunk.document_id == document.id).delete()
        if os.path.exists(chunks_source):
            with open(chunks_source, "r", encoding="utf-8") as f:
                chunk_items = json.load(f)
            for item in chunk_items:
                db.add(
                    Chunk(
                        document_id=document.id,
                        chunk_index=int(item.get("chunk_index", 0)),
                        text=item.get("text", ""),
                        metadata_json=json.dumps(item.get("metadata", {}), ensure_ascii=False),
                        embedding_json="[]",
                    )
                )

        document.markdown_path = ""
        document.markdown_url = ""
        document.process_status = "ready"
        document.chunk_count = chunk_count
        document.processed_at = datetime.utcnow()
        db.add(document)
        db.commit()
        if vector_store_config.use_milvus:
            return {"chunk_count": chunk_count, "output_dir": _doc_milvus_collection(file_hash)}
        return {"chunk_count": chunk_count, "output_dir": index_dir}
    except Exception as exc:
        document.process_status = "failed"
        document.process_error = str(exc)
        db.add(document)
        db.commit()
        raise


def search_chunks(
    db,
    owner_id,
    query,
    top_k=5,
    document_id=None,
    mineru_output_dir="./mineru_output",
    embeddings_model_name="all-MiniLM-L6-v2",
    embeddings_device="cpu",
):
    q = db.query(Document).filter(Document.owner_id == owner_id, Document.process_status == "ready")
    if document_id is not None:
        q = q.filter(Document.id == document_id)
    docs = q.all()

    mode = _retrieval_mode()
    recall_n = _recall_topn()
    per_doc_k = max(top_k, min(recall_n, 50))

    all_hits = []
    for doc in docs:
        if not doc.file_hash:
            continue

        dense_hits = []
        if mode in ("dense", "hybrid"):
            try:
                if vector_store_config.use_milvus:
                    collection_name = _doc_milvus_collection(doc.file_hash)
                    vs = load_milvus_index(
                        collection_name=collection_name,
                        embeddings_model=embeddings_model_name,
                        embeddings_device=embeddings_device,
                        connection_args=vector_store_config.milvus_connection_args(),
                        search_params=vector_store_config.milvus_search_params,
                    )
                    raw = vs.similarity_search_with_score(query, k=per_doc_k)
                else:
                    index_dir = _doc_index_dir(mineru_output_dir, doc.file_hash)
                    if not os.path.exists(os.path.join(index_dir, "index.faiss")):
                        raw = []
                    else:
                        vs = load_faiss_index(
                            index_dir=index_dir,
                            embeddings_model=embeddings_model_name,
                            embeddings_device=embeddings_device,
                        )
                        raw = vs.similarity_search_with_score(query, k=per_doc_k)
                for lc_doc, score in raw:
                    meta = lc_doc.metadata or {}
                    similarity = 1.0 / (1.0 + float(score))
                    dense_hits.append(
                        {
                            "score": similarity,
                            "document_id": doc.id,
                            "document_name": doc.filename,
                            "chunk_id": int(meta.get("chunk_index", 0)),
                            "chunk_index": int(meta.get("chunk_index", 0)),
                            "page": _extract_page_from_meta(meta),
                            "text": lc_doc.page_content,
                            "parent_text": meta.get("parent_text") or "",
                            "heading_path": meta.get("heading_path") or "",
                            "parent_id": meta.get("parent_id"),
                        }
                    )
            except Exception:
                dense_hits = []

        sparse_hits = []
        if mode in ("bm25", "hybrid"):
            chunks_json_path = (
                _doc_milvus_chunks_json(mineru_output_dir, doc.file_hash)
                if vector_store_config.use_milvus
                else _doc_chunks_json(mineru_output_dir, doc.file_hash)
            )
            try:
                raw = bm25_search(doc.file_hash, chunks_json_path, query, top_k=per_doc_k)
            except Exception:
                raw = []
            for item in raw:
                meta = item.get("metadata") or {}
                sparse_hits.append(
                    {
                        "score": float(item.get("score", 0.0)),
                        "document_id": doc.id,
                        "document_name": doc.filename,
                        "chunk_id": int(item.get("chunk_index", 0)),
                        "chunk_index": int(item.get("chunk_index", 0)),
                        "page": _extract_page_from_meta(meta),
                        "text": item.get("text", ""),
                        "parent_text": meta.get("parent_text") or "",
                        "heading_path": meta.get("heading_path") or "",
                        "parent_id": meta.get("parent_id"),
                    }
                )

        if mode == "dense":
            doc_fused = dense_hits
        elif mode == "bm25":
            doc_fused = sparse_hits
        else:
            doc_fused = rrf_fuse([dense_hits, sparse_hits], k=60, top_k=per_doc_k)

        all_hits.extend(doc_fused)

    if mode == "hybrid":
        # re-fuse across documents so RRF ranks are global
        all_hits.sort(key=lambda x: x.get("rrf_score", x.get("score", 0.0)), reverse=True)
    else:
        all_hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Optional cross-encoder rerank on the fused candidate set. Give rerank
    # a slightly wider window so parent-dedup (below) still leaves us top_k.
    all_hits = cross_rerank(query, all_hits[: max(top_k * 3, recall_n)], top_k=top_k * 3)

    # Small-to-Big: collapse multiple children hitting the same parent so the
    # LLM isn't handed duplicate context. We keep the first (highest-scoring)
    # child per (document_id, parent_id).
    deduped = []
    seen_parents = set()
    for h in all_hits:
        pid = h.get("parent_id")
        if pid is None:
            deduped.append(h)
            continue
        key = (h.get("document_id"), pid)
        if key in seen_parents:
            continue
        seen_parents.add(key)
        deduped.append(h)
    all_hits = deduped

    # Normalize an exposed `score` for the UI: prefer rerank > rrf > raw
    for h in all_hits:
        if "rerank_score" in h:
            h["score"] = float(h["rerank_score"])
        elif "rrf_score" in h:
            h["score"] = float(h["rrf_score"])
    return all_hits[:top_k]


def list_document_chunks(db, owner_id, document_id, limit=2000):
    doc = db.query(Document).filter(Document.id == document_id, Document.owner_id == owner_id).first()
    if not doc:
        return None
    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index.asc())
        .limit(max(1, min(limit, 5000)))
        .all()
    )
    items = []
    for r in rows:
        meta = {}
        try:
            meta = json.loads(r.metadata_json or "{}")
        except Exception:
            meta = {}
        items.append({"chunk_index": r.chunk_index, "page": _extract_page_from_meta(meta), "text": r.text})
    return items
