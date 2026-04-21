import os
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_optional_current_user
from app.core.config import settings
from app.core.security import decode_access_token
from app.core.vector_store_config import vector_store_config
from app.db.session import get_db
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.user import User
from app.schemas.document import DocumentOut
from app.services.mineru_parser import compute_file_sha256
from app.services.retrieval import process_document

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".pdf"}


def _doc_out(doc: Document):
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        file_url=f"/api/documents/file/{doc.id}",
        file_hash=doc.file_hash,
        markdown_path=doc.markdown_path,
        markdown_url=doc.markdown_url,
        process_status=doc.process_status,
        process_error=doc.process_error,
        chunk_count=doc.chunk_count,
        processed_at=doc.processed_at,
        created_at=doc.created_at,
    )


@router.post("/upload", response_model=DocumentOut)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _allowed(file.filename):
        raise HTTPException(status_code=400, detail="Only PDF is supported in MVP stage")

    os.makedirs(settings.upload_dir, exist_ok=True)
    ext = Path(file.filename).suffix.lower()
    stored_name = f"{uuid4().hex}{ext}"
    abs_path = str(Path(settings.upload_dir) / stored_name)

    with open(abs_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file_hash = compute_file_sha256(abs_path)

    doc = Document(
        owner_id=current_user.id,
        filename=file.filename,
        stored_filename=stored_name,
        file_path=abs_path,
        file_hash=file_hash,
        markdown_path="",
        markdown_url="",
        process_status="uploaded",
        process_error="",
        chunk_count=0,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    if settings.auto_process_on_upload:
        try:
            process_document(
                db=db,
                document=doc,
                mineru_output_dir=settings.mineru_output_dir,
                mineru_cmd=settings.mineru_cmd,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                parse_mode=settings.mineru_parse_mode,
                api_base_url=settings.mineru_api_base_url,
                api_timeout_sec=settings.mineru_api_timeout_sec,
                api_poll_interval_sec=settings.mineru_api_poll_interval_sec,
                language=settings.mineru_language,
                enable_table=settings.mineru_enable_table,
                enable_formula=settings.mineru_enable_formula,
                is_ocr=settings.mineru_is_ocr,
                mineru_loader_mode=settings.mineru_loader_mode,
                embeddings_model_name=settings.embeddings_model_name,
                embeddings_device=settings.embeddings_device,
            )
            db.refresh(doc)
        except Exception:
            db.refresh(doc)

    return _doc_out(doc)


@router.get("", response_model=list)
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    docs = (
        db.query(Document)
        .filter(Document.owner_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return [_doc_out(d) for d in docs]


@router.get("/file/{document_id}")
def get_document_file(
    document_id: int,
    token: str = Query(default="", description="JWT token for iframe/pdf access"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_current_user),
):
    user = current_user
    if not user and token:
        username = decode_access_token(token)
        if username:
            user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc = db.query(Document).filter(Document.id == document_id, Document.owner_id == user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(
        path=doc.file_path,
        media_type="application/pdf",
        filename=doc.filename,
        content_disposition_type="inline",
    )


def _safe_rmtree(path):
    if not path:
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _safe_remove(path):
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner_id == current_user.id)
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_hash = doc.file_hash
    file_path = doc.file_path

    # Only drop the shared cache/index if no other doc row references the same hash.
    other_same_hash = 0
    if file_hash:
        other_same_hash = (
            db.query(Document)
            .filter(Document.file_hash == file_hash, Document.id != doc.id)
            .count()
        )

    db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
    db.delete(doc)
    db.commit()

    _safe_remove(file_path)

    if file_hash and other_same_hash == 0:
        cache_root = os.path.join(settings.mineru_output_dir, "lc_cache", file_hash)
        _safe_rmtree(cache_root)
        if vector_store_config.use_milvus:
            try:
                from pymilvus import connections, utility  # type: ignore

                conn_args = vector_store_config.milvus_connection_args() or {}
                connections.connect(alias="_delete", **conn_args)
                collection = "%s%s" % (vector_store_config.milvus_collection_prefix, file_hash)
                if utility.has_collection(collection, using="_delete"):
                    utility.drop_collection(collection, using="_delete")
                connections.disconnect("_delete")
            except Exception:
                pass

    return {"ok": True, "deleted_id": document_id}
