from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.document import Document
from app.models.user import User
from app.schemas.retrieval import ChunkListResponse, ProcessResult, SearchRequest, SearchResponse
from app.services.retrieval import list_document_chunks, process_document, search_chunks

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/process/{document_id}", response_model=ProcessResult)
def process_doc_endpoint(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(Document.id == document_id, Document.owner_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    result = process_document(
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
    return ProcessResult(
        document_id=doc.id,
        process_status=doc.process_status,
        chunk_count=result["chunk_count"],
        output_dir=result["output_dir"],
    )


@router.post("/search", response_model=SearchResponse)
def search_endpoint(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")
    hits = search_chunks(
        db=db,
        owner_id=current_user.id,
        query=payload.query,
        top_k=min(payload.top_k, 20),
        document_id=payload.document_id,
        mineru_output_dir=settings.mineru_output_dir,
        embeddings_model_name=settings.embeddings_model_name,
        embeddings_device=settings.embeddings_device,
    )
    return SearchResponse(query=payload.query, hits=hits)


@router.get("/chunks/{document_id}", response_model=ChunkListResponse)
def chunks_endpoint(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = list_document_chunks(
        db=db,
        owner_id=current_user.id,
        document_id=document_id,
        limit=3000,
    )
    if items is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return ChunkListResponse(document_id=document_id, chunks=items)
