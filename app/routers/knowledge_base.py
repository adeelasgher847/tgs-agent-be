from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.core.config import settings
from app.core.logger import logger
from app.models.knowledge_base_document import KnowledgeBase
from app.models.knowledge_base_chunk import KbChunk
from app.models.kb_file import KbFile
from app.schemas.base import SuccessResponse
from app.schemas.knowledge_base import (
    KbCreate,
    KbFileStatusOut,
    KbFileUploadResponse,
    KbList,
    KbOut,
    KbTextIngestRequest,
    KbTextIngestResponse,
    KnowledgeBaseDocumentList,
    KnowledgeBaseDocumentOut,
    KnowledgeBaseIngestTextRequest,
    KnowledgeBaseIngestTextResponse,
    KnowledgeBaseRetrievePreviewRequest,
    KnowledgeBaseRetrievePreviewResponse,
    KnowledgeBaseRetrievedChunkOut,
)
from app.services.embedding_service import embed_text_for_rag
from app.services.kb_ingestion_service import (
    gcs_kb_path,
    run_text_ingestion,
    upload_kb_file_to_gcs,
)
from app.services.rag_service import rag_service
from app.utils.response import create_success_response
from app.utils.arq_pool import get_arq_pool

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

router = APIRouter()


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_kb_or_404(db: Session, kb_id: uuid.UUID, workspace_id: uuid.UUID) -> KnowledgeBase:
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.workspace_id == workspace_id)
        .first()
    )
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


# ── Knowledge base CRUD ───────────────────────────────────────────────────────

@router.post("/", response_model=SuccessResponse[KbOut], status_code=201)
def create_knowledge_base(
    payload: KbCreate,
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    kb = KnowledgeBase(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        name=payload.name,
        description=payload.description,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return create_success_response(KbOut.model_validate(kb), "Knowledge base created")


@router.get("/", response_model=SuccessResponse[KbList])
def list_knowledge_bases(
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    kbs = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.workspace_id == workspace_id)
        .order_by(KnowledgeBase.created_at.desc())
        .all()
    )
    return create_success_response(
        KbList(knowledge_bases=[KbOut.model_validate(k) for k in kbs], total=len(kbs)),
        "Knowledge bases fetched",
    )


@router.delete("/{kb_id}", response_model=SuccessResponse[dict])
def delete_knowledge_base(
    kb_id: uuid.UUID,
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    kb = _get_kb_or_404(db, kb_id, workspace_id)
    db.delete(kb)
    db.commit()
    return create_success_response({"kb_id": str(kb_id)}, "Knowledge base deleted")


# ── File upload → async ingestion ─────────────────────────────────────────────

@router.post("/{kb_id}/file", response_model=SuccessResponse[KbFileUploadResponse], status_code=202)
async def upload_kb_file(
    kb_id: uuid.UUID,
    file: UploadFile = File(...),
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Accept a PDF, DOCX, or TXT file (≤50 MB).
    Streams it to GCS, creates a KbFile record, enqueues the ARQ ingestion job.
    Returns 202 {job_id, file_id}.
    """
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    _get_kb_or_404(db, kb_id, workspace_id)

    # Validate extension
    filename = file.filename or ""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read content — enforce 50 MB cap
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
        )

    file_id = uuid.uuid4()
    file_type = ext.lstrip(".")

    # Persist KbFile record (status=processing)
    kb_file = KbFile(
        id=file_id,
        kb_id=kb_id,
        original_filename=filename,
        size_bytes=len(content),
        file_type=file_type,
        status="processing",
    )

    # Upload to GCS if bucket is configured
    if settings.GCS_KB_BUCKET:
        gcs_path = gcs_kb_path(workspace_id, kb_id, file_id, filename)
        try:
            import io
            upload_kb_file_to_gcs(
                settings.GCS_KB_BUCKET, gcs_path, io.BytesIO(content)
            )
            kb_file.gcs_path = gcs_path
        except Exception as e:
            logger.error("GCS upload failed for file_id=%s: %s", file_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail="File storage failed")

    db.add(kb_file)
    db.commit()

    # Enqueue ARQ ingestion job
    job_id = str(uuid.uuid4())
    pool = get_arq_pool()
    if pool:
        try:
            await pool.enqueue_job("kb_ingestion_task", str(file_id))
        except Exception as e:
            logger.warning("ARQ enqueue failed for file_id=%s: %s", file_id, e)
    else:
        logger.warning("ARQ pool unavailable; kb_ingestion_task not queued for file_id=%s", file_id)

    return create_success_response(
        KbFileUploadResponse(job_id=job_id, file_id=file_id),
        "File accepted for ingestion",
    )


# ── Text ingest → synchronous ─────────────────────────────────────────────────

@router.post("/{kb_id}/text", response_model=SuccessResponse[KbTextIngestResponse], status_code=201)
async def ingest_kb_text(
    kb_id: uuid.UUID,
    payload: KbTextIngestRequest,
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Ingest a raw text snippet (≤10 000 chars) synchronously.
    Chunks via tiktoken, embeds with OpenAI ada-002, inserts kb_chunks.
    Returns 201 {kb_id, chunk_count}.
    """
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    _get_kb_or_404(db, kb_id, workspace_id)

    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is required for text ingestion",
        )

    try:
        chunk_count = await run_text_ingestion(
            db=db,
            kb_id=kb_id,
            content=payload.content,
            api_key=settings.OPENAI_API_KEY,
        )
        db.commit()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error("Text ingestion failed for kb_id=%s: %s", kb_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Ingestion failed")

    return create_success_response(
        KbTextIngestResponse(kb_id=kb_id, chunk_count=chunk_count),
        "Text ingested successfully",
    )


# ── File status ───────────────────────────────────────────────────────────────

@router.get(
    "/{kb_id}/files/{file_id}/status",
    response_model=SuccessResponse[KbFileStatusOut],
)
def get_file_status(
    kb_id: uuid.UUID,
    file_id: uuid.UUID,
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    _get_kb_or_404(db, kb_id, workspace_id)

    kb_file = db.query(KbFile).filter(KbFile.id == file_id, KbFile.kb_id == kb_id).first()
    if not kb_file:
        raise HTTPException(status_code=404, detail="File not found")

    return create_success_response(
        KbFileStatusOut(
            file_id=kb_file.id,
            status=kb_file.status,
            chunk_count=kb_file.chunk_count,
            error_message=kb_file.error_message,
        ),
        "File status fetched",
    )


# ── Legacy: list knowledge bases as "documents" ───────────────────────────────

@router.get("/documents", response_model=SuccessResponse[KnowledgeBaseDocumentList])
def list_documents(
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    kbs = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.workspace_id == workspace_id)
        .order_by(KnowledgeBase.created_at.desc())
        .all()
    )
    return create_success_response(
        KnowledgeBaseDocumentList(
            documents=[KnowledgeBaseDocumentOut.model_validate(k) for k in kbs],
            total=len(kbs),
        ),
        "KB documents fetched successfully",
    )


# ── Legacy: ingest text via old interface ─────────────────────────────────────

@router.post(
    "/documents/ingest-text",
    response_model=SuccessResponse[KnowledgeBaseIngestTextResponse],
)
async def ingest_text_document(
    request: KnowledgeBaseIngestTextRequest,
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Legacy endpoint: ingest normalized text into the workspace's Auto-Ingest KB."""
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    if not settings.OPENAI_API_KEY and not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="At least one embedding provider key must be configured",
        )

    try:
        document_id = rag_service.ingest_document(
            tenant_id=workspace_id,
            agent_id=request.agent_id,
            title=request.title,
            source_type=request.source_type,
            source_ref=request.source_ref,
            full_text=request.full_text,
            embedding_func=embed_text_for_rag,
            version=request.version,
            db_session=db,
            replace_existing=True,
            max_chars=request.chunk_max_chars,
            overlap_chars=request.chunk_overlap_chars,
        )
    except Exception as e:
        logger.error("Legacy ingest-text failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Ingestion failed")

    return create_success_response(
        KnowledgeBaseIngestTextResponse(document_id=document_id),
        "Document ingested successfully",
    )


# ── Legacy: retrieve preview ──────────────────────────────────────────────────

@router.post(
    "/retrieve-preview",
    response_model=SuccessResponse[KnowledgeBaseRetrievePreviewResponse],
)
def retrieve_preview(
    request: KnowledgeBaseRetrievePreviewRequest,
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    rag_chunks = rag_service.retrieve(
        user_text=request.user_text,
        tenant_id=workspace_id,
        agent_id=request.agent_id,
        embedding_func=embed_text_for_rag,
        top_k=request.top_k,
        db_session=db,
    )

    filtered = [c for c in rag_chunks if (c.score or 0.0) >= settings.RAG_SCORE_THRESHOLD]
    context_block = rag_service.format_rag_context(filtered, max_chars=settings.RAG_MAX_CONTEXT_CHARS)

    retrieved_chunks_out = [
        KnowledgeBaseRetrievedChunkOut(
            chunk_n=i,
            score=c.score,
            source_title=c.source_title,
            source_ref=c.source_ref,
        )
        for i, c in enumerate(filtered, start=1)
    ]

    return create_success_response(
        KnowledgeBaseRetrievePreviewResponse(
            context_block=context_block,
            retrieved_chunks=retrieved_chunks_out,
        ),
        "RAG retrieve preview generated",
    )


# ── Legacy: delete document (now deletes all chunks for a KB) ────────────────

@router.delete("/documents/{document_id}", response_model=SuccessResponse[dict])
def delete_document(
    document_id: uuid.UUID,
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    # document_id maps to kb_id in the new schema
    kb = _get_kb_or_404(db, document_id, workspace_id)

    chunk_rows = db.query(KbChunk).filter(KbChunk.kb_id == kb.id).all()
    vector_ids = [str(c.id) for c in chunk_rows]
    rag_service.delete_vectors(vector_ids=vector_ids, db_session=db)

    db.delete(kb)
    db.commit()
    return create_success_response(
        {"document_id": str(document_id), "deleted_chunks": len(vector_ids)},
        "Document deleted successfully",
    )
