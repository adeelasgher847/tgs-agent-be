from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

import json as _json

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import String, cast, func, text as sa_text
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_owner, require_tenant
from app.core.config import settings
from app.core.logger import logger
from app.models.call_flow import CallFlow
from app.models.knowledge_base_document import KnowledgeBase
from app.models.knowledge_base_chunk import KbChunk
from app.models.kb_file import KbFile
from app.schemas.base import SuccessResponse
from app.schemas.knowledge_base import (
    KbCreate,
    KbDetail,
    KbFileOut,
    KbFileStatusOut,
    KbFileUploadResponse,
    KbList,
    KbListItem,
    KbOut,
    KbSearchResponse,
    KbSearchResultItem,
    KbTextIngestRequest,
    KbTextIngestResponse,
    KbUpdate,
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
from app.utils.redis_client import get_redis

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# In-process TTL cache used when Redis is unavailable.
# Structure: key -> (value: int, expires_at: float)
_MEM_CACHE: dict[str, tuple[int, float]] = {}

_FILE_COUNT_TTL = 60    # seconds
_CHUNK_COUNT_TTL = 120  # seconds

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_kb_or_404(db: Session, kb_id: uuid.UUID, workspace_id: uuid.UUID) -> KnowledgeBase:
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.workspace_id == workspace_id)
        .first()
    )
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def _file_count(db: Session, kb_id: uuid.UUID) -> int:
    return (
        db.query(func.count(KbFile.id))
        .filter(KbFile.kb_id == kb_id, KbFile.status == "ready")
        .scalar()
        or 0
    )


def _chunk_count(db: Session, kb_id: uuid.UUID) -> int:
    return (
        db.query(func.count(KbChunk.id))
        .filter(KbChunk.kb_id == kb_id)
        .scalar()
        or 0
    )


def _mem_cache_get(key: str) -> Optional[int]:
    entry = _MEM_CACHE.get(key)
    if entry is not None and time.monotonic() < entry[1]:
        return entry[0]
    _MEM_CACHE.pop(key, None)
    return None


def _mem_cache_set(key: str, value: int, ttl: int) -> None:
    _MEM_CACHE[key] = (value, time.monotonic() + ttl)


async def _cached_file_count(db: Session, kb_id: uuid.UUID) -> int:
    """COUNT of ready files for a KB; cached 60 s in Redis (fallback: in-process dict)."""
    key = f"kb:file_count:{kb_id}"
    redis = get_redis()
    if redis is not None:
        try:
            cached = await redis.get(key)
            if cached is not None:
                return int(cached)
        except Exception:
            pass
    else:
        mem = _mem_cache_get(key)
        if mem is not None:
            return mem

    count = _file_count(db, kb_id)

    if redis is not None:
        try:
            await redis.setex(key, _FILE_COUNT_TTL, count)
        except Exception:
            pass
    else:
        _mem_cache_set(key, count, _FILE_COUNT_TTL)

    return count


async def _cached_chunk_count(db: Session, kb_id: uuid.UUID) -> int:
    """COUNT of chunks for a KB; cached 120 s in Redis (fallback: in-process dict)."""
    key = f"kb:chunk_count:{kb_id}"
    redis = get_redis()
    if redis is not None:
        try:
            cached = await redis.get(key)
            if cached is not None:
                return int(cached)
        except Exception:
            pass
    else:
        mem = _mem_cache_get(key)
        if mem is not None:
            return mem

    count = _chunk_count(db, kb_id)

    if redis is not None:
        try:
            await redis.setex(key, _CHUNK_COUNT_TTL, count)
        except Exception:
            pass
    else:
        _mem_cache_set(key, count, _CHUNK_COUNT_TTL)

    return count


def _bytes_to_mb(size_bytes: Optional[int]) -> Optional[str]:
    if size_bytes is None:
        return None
    return f"{round(size_bytes / 1_048_576, 2):.2f} MB"


def _is_kb_linked_to_active_flow(db: Session, kb_id: uuid.UUID) -> bool:
    """Return True if the KB is referenced in any non-deleted call flow.

    Uses a String-cast LIKE search so it works on both PostgreSQL (JSONB) and
    SQLite (TEXT) test environments.
    """
    result = (
        db.query(CallFlow)
        .filter(
            CallFlow.is_deleted == False,  # noqa: E712
            cast(CallFlow.knowledge_base_ids, String).contains(str(kb_id)),
        )
        .first()
    )
    return result is not None


# ── Knowledge base CRUD ───────────────────────────────────────────────────────

@router.post("/", response_model=SuccessResponse[KbOut], status_code=201)
def create_knowledge_base(
    payload: KbCreate,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
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
async def list_knowledge_bases(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
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
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )
    total = (
        db.query(func.count(KnowledgeBase.id))
        .filter(KnowledgeBase.workspace_id == workspace_id)
        .scalar()
        or 0
    )

    items = []
    for kb in kbs:
        items.append(
            KbListItem(
                id=kb.id,
                name=kb.name,
                description=kb.description,
                file_count=await _cached_file_count(db, kb.id),
                total_chunk_count=await _cached_chunk_count(db, kb.id),
                created_at=kb.created_at,
            )
        )
    return create_success_response(
        KbList(knowledge_bases=items, total=total),
        "Knowledge bases fetched",
    )


@router.get("/{kb_id}", response_model=SuccessResponse[KbDetail])
def get_knowledge_base(
    kb_id: uuid.UUID,
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    kb = _get_kb_or_404(db, kb_id, workspace_id)
    files = (
        db.query(KbFile)
        .filter(KbFile.kb_id == kb_id)
        .order_by(KbFile.created_at.desc())
        .all()
    )
    file_out = [
        KbFileOut(
            id=f.id,
            filename=f.original_filename,
            size_bytes=f.size_bytes,
            size_mb=_bytes_to_mb(f.size_bytes),
            status=f.status,
            chunk_count=f.chunk_count,
            created_at=f.created_at,
        )
        for f in files
    ]
    detail = KbDetail(
        id=kb.id,
        workspace_id=kb.workspace_id,
        name=kb.name,
        description=kb.description,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
        files=file_out,
    )
    return create_success_response(detail, "Knowledge base fetched")


@router.put("/{kb_id}", response_model=SuccessResponse[KbOut])
def update_knowledge_base(
    kb_id: uuid.UUID,
    payload: KbUpdate,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    kb = _get_kb_or_404(db, kb_id, workspace_id)

    if payload.name is not None:
        kb.name = payload.name
    if payload.description is not None:
        kb.description = payload.description

    db.commit()
    db.refresh(kb)
    return create_success_response(KbOut.model_validate(kb), "Knowledge base updated")


@router.delete("/{kb_id}", response_model=SuccessResponse[dict])
def delete_knowledge_base(
    kb_id: uuid.UUID,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    kb = _get_kb_or_404(db, kb_id, workspace_id)

    if _is_kb_linked_to_active_flow(db, kb_id):
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a knowledge base that is linked to an active call flow",
        )

    db.delete(kb)
    db.commit()
    return create_success_response({"kb_id": str(kb_id)}, "Knowledge base deleted")


# ── File management ───────────────────────────────────────────────────────────

@router.delete("/{kb_id}/files/{file_id}", response_model=SuccessResponse[dict])
def delete_kb_file(
    kb_id: uuid.UUID,
    file_id: uuid.UUID,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    workspace_id = user.current_tenant_id
    _get_kb_or_404(db, kb_id, workspace_id)

    kb_file = (
        db.query(KbFile)
        .filter(KbFile.id == file_id, KbFile.kb_id == kb_id)
        .first()
    )
    if not kb_file:
        raise HTTPException(status_code=404, detail="File not found")

    # Explicitly delete associated chunks (FK is SET NULL, not CASCADE)
    db.query(KbChunk).filter(KbChunk.file_id == file_id).delete(synchronize_session=False)
    db.delete(kb_file)
    db.commit()
    return create_success_response(
        {"file_id": str(file_id), "kb_id": str(kb_id)},
        "File deleted",
    )


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

    filename = file.filename or ""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
        )

    file_id = uuid.uuid4()
    file_type = ext.lstrip(".")

    kb_file = KbFile(
        id=file_id,
        kb_id=kb_id,
        original_filename=filename,
        size_bytes=len(content),
        file_type=file_type,
        status="processing",
    )

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


# ── Similarity search ─────────────────────────────────────────────────────────

@router.get("/{kb_id}/search", response_model=SuccessResponse[KbSearchResponse])
async def search_knowledge_base(
    kb_id: uuid.UUID,
    q: str = Query(..., min_length=1, description="Search query text"),
    limit: int = Query(default=5, ge=1, le=25),
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Cosine-similarity search over a single KB's chunks.
    Returns [{content, score, metadata}] sorted by relevance descending.
    """
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    _get_kb_or_404(db, kb_id, workspace_id)

    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is required for KB search",
        )

    try:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, embed_text_for_rag, q)
        vec_str = "[" + ",".join(str(f) for f in embedding) + "]"

        stmt = sa_text(
            """
            SELECT content,
                   1 - (embedding::vector <=> CAST(:vec AS vector)) AS score,
                   metadata AS chunk_metadata
            FROM kbchunk
            WHERE kb_id = :kb_id
              AND embedding IS NOT NULL
            ORDER BY embedding::vector <=> CAST(:vec AS vector)
            LIMIT :limit
            """
        )
        rows = db.execute(
            stmt, {"vec": vec_str, "kb_id": str(kb_id), "limit": limit}
        ).fetchall()
    except Exception as e:
        logger.error("KB search failed for kb_id=%s: %s", kb_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")

    results = []
    for row in rows:
        meta = row.chunk_metadata or {}
        if isinstance(meta, str):
            try:
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        results.append(
            KbSearchResultItem(
                content=row.content,
                score=float(row.score or 0.0),
                metadata=meta,
            )
        )

    return create_success_response(
        KbSearchResponse(results=results),
        f"Search returned {len(results)} result(s)",
    )


# ── Legacy: list knowledge bases as "documents" ────────────────────────────────

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
