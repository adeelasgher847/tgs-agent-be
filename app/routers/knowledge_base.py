from __future__ import annotations

import asyncio
import time
import uuid

import json as _json
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import String, cast, func, text as sa_text
from sqlalchemy.orm import Session

from app.api.deps import (
    get_db,
    require_admin_or_owner,
    require_config_or_api_key,
    require_readonly_or_api_key,
)
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
    KbFileAccessResponse,
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
    upload_kb_file_to_s3,
)
from app.services.rag_service import rag_service
from app.services.s3_service import get_s3_client
from app.utils.response import create_success_response
from app.utils.arq_pool import get_arq_pool
from app.utils.redis_client import get_redis

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
KB_FILE_SIGNED_URL_EXPIRY_SECONDS = 300  # 5 minutes

_CONTENT_TYPE_BY_FILE_TYPE = {
    "pdf": "application/pdf",
    "txt": "text/plain",
}

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


def _content_disposition(disposition: str, filename: str) -> str:
    """Build a Content-Disposition header value, stripping characters that
    could break out of the quoted filename (e.g. a `"` in a user-supplied
    upload filename injecting extra header parameters)."""
    safe_filename = filename.replace('"', "").replace("\r", "").replace("\n", "")
    return f'{disposition}; filename="{safe_filename}"'


def _get_kb_file_by_id_or_404(
    db: Session, file_id: uuid.UUID, workspace_id: uuid.UUID
) -> KbFile:
    """Tenant-scoped KbFile lookup that isn't nested under a known kb_id.

    KbFile has no workspace_id column of its own, so tenant scoping is done
    by joining through KnowledgeBase.workspace_id — never trust file_id alone.
    """
    kb_file = (
        db.query(KbFile)
        .join(KnowledgeBase, KnowledgeBase.id == KbFile.kb_id)
        .filter(KbFile.id == file_id, KnowledgeBase.workspace_id == workspace_id)
        .first()
    )
    if not kb_file:
        raise HTTPException(status_code=404, detail="File not found")
    return kb_file


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


def _mem_cache_get(key: str) -> int | None:
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
        except Exception as exc:
            logger.debug("Redis file-count cache read failed for %s: %s", key, exc)
    else:
        mem = _mem_cache_get(key)
        if mem is not None:
            return mem

    count = _file_count(db, kb_id)

    if redis is not None:
        try:
            await redis.setex(key, _FILE_COUNT_TTL, count)
        except Exception as exc:
            logger.debug("Redis file-count cache write failed for %s: %s", key, exc)
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
        except Exception as exc:
            logger.debug("Redis chunk-count cache read failed for %s: %s", key, exc)
    else:
        mem = _mem_cache_get(key)
        if mem is not None:
            return mem

    count = _chunk_count(db, kb_id)

    if redis is not None:
        try:
            await redis.setex(key, _CHUNK_COUNT_TTL, count)
        except Exception as exc:
            logger.debug("Redis chunk-count cache write failed for %s: %s", key, exc)
    else:
        _mem_cache_set(key, count, _CHUNK_COUNT_TTL)

    return count


def _bytes_to_mb(size_bytes: int | None) -> str | None:
    if size_bytes is None:
        return None
    return f"{round(size_bytes / 1_048_576, 2):.2f} MB"


def _linked_call_flow_ids_by_kb(
    db: Session, workspace_id: uuid.UUID
) -> dict[str, list[uuid.UUID]]:
    """Non-deleted call flows in this workspace, grouped by linked KB id.

    One query for the whole workspace (used by both the list and detail
    endpoints) rather than one query per KB. Filters in Python rather than
    with a JSONB containment operator so this stays portable across the
    Postgres (JSONB) and SQLite (TEXT) test setups.
    """
    flows = (
        db.query(CallFlow.id, CallFlow.knowledge_base_ids)
        .filter(CallFlow.tenant_id == workspace_id, CallFlow.is_deleted == False)  # noqa: E712
        .all()
    )
    by_kb: dict[str, list[uuid.UUID]] = {}
    for flow_id, kb_ids in flows:
        # dict.fromkeys dedupes while preserving order, in case a flow's own
        # knowledge_base_ids array ever contains the same id twice.
        for kb_id_str in dict.fromkeys(kb_ids or []):
            by_kb.setdefault(kb_id_str, []).append(flow_id)
    return by_kb


def _linked_call_flow_ids(
    db: Session, kb_id: uuid.UUID, workspace_id: uuid.UUID
) -> list[uuid.UUID]:
    return _linked_call_flow_ids_by_kb(db, workspace_id).get(str(kb_id), [])


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

@router.post(
    "/",
    response_model=SuccessResponse[KbOut],
    status_code=201,
    summary="Create a knowledge base",
)
def create_knowledge_base(
    payload: KbCreate,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    """
    Create an empty knowledge base (KB) container, scoped to the caller's
    current workspace. This only creates the KB record — it has no content
    yet, so it won't affect any calls until you populate and attach it.

    **Auth:** requires `admin` or `owner` role (JWT). Workspace is taken from
    the authenticated user's `current_tenant_id` — there is no `workspaceId`
    in the request body.

    **Typical frontend flow:**
    1. `POST /api/v1/kb/` (this endpoint) → get back `data.id` (the `kb_id`).
    2. Add content to the KB, either or both:
       - `POST /api/v1/kb/{kb_id}/file` — upload a PDF/DOCX/TXT file (≤ 50 MB).
         Returns `202` immediately with a `job_id`; ingestion (chunking +
         embedding) runs asynchronously. Poll
         `GET /api/v1/kb/{kb_id}/files/{file_id}/status` to know when it's
         ready.
       - `POST /api/v1/kb/{kb_id}/text` — ingest raw text directly.
         Synchronous — chunks + embeddings are ready when this call returns
         `201`.
    3. Attach the KB to a call flow so the voice agent can retrieve from it:
       `PUT /api/v1/call-flows/{flow_id}/knowledge-bases` with
       `{"kb_ids": ["<kb_id>", ...]}`. A call flow can have multiple KBs;
       passing an empty list detaches all of them.

    **Request body:**
    - `name` (required) — display name, 1–255 characters.
    - `description` (optional) — free-text summary of the KB's contents.

    **Response (`201`):** `data` is the created KB —
    `{id, workspace_id, name, description, created_at, updated_at}`.
    """
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
    user=Depends(require_readonly_or_api_key),
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

    call_flow_ids_by_kb = _linked_call_flow_ids_by_kb(db, workspace_id)
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
                call_flow_ids=call_flow_ids_by_kb.get(str(kb.id), []),
            )
        )
    return create_success_response(
        KbList(knowledge_bases=items, total=total),
        "Knowledge bases fetched",
    )


@router.get("/{kb_id}", response_model=SuccessResponse[KbDetail])
def get_knowledge_base(
    kb_id: uuid.UUID,
    user=Depends(require_readonly_or_api_key),
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
        call_flow_ids=_linked_call_flow_ids(db, kb_id, workspace_id),
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


# ── File access (view/download) ─────────────────────────────────────────────────

@router.get("/files/{file_id}", response_model=SuccessResponse[KbFileAccessResponse])
def get_kb_file_access(
    file_id: uuid.UUID,
    action: Literal["view", "download"] = Query(
        ..., description="Whether to return a URL for inline viewing or forced download."
    ),
    user=Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
):
    """
    Return a short-lived presigned S3 URL to view or download a KB file.

    Note: unlike other file endpoints on this router, this is not nested
    under `/{kb_id}/` — tenant scoping is enforced by joining KbFile to its
    parent KnowledgeBase's workspace_id.
    """
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    kb_file = _get_kb_file_by_id_or_404(db, file_id, workspace_id)

    if kb_file.status != "ready" or not kb_file.s3_path:
        raise HTTPException(status_code=404, detail="File not available")

    if not settings.S3_KB_BUCKET:
        logger.error(
            "S3_KB_BUCKET is not configured; cannot generate signed URL for file_id=%s",
            file_id,
        )
        raise HTTPException(status_code=500, detail="File storage is not configured")

    file_type = (kb_file.file_type or "").lower()
    preview_supported = file_type in ("pdf", "txt")

    params = {
        "Bucket": settings.S3_KB_BUCKET,
        "Key": kb_file.s3_path,
    }

    if action == "download":
        params["ResponseContentDisposition"] = _content_disposition(
            "attachment", kb_file.original_filename
        )
    else:
        # action == "view"
        if preview_supported:
            params["ResponseContentDisposition"] = "inline"
            content_type = _CONTENT_TYPE_BY_FILE_TYPE.get(file_type)
            if content_type:
                params["ResponseContentType"] = content_type
        else:
            # DOCX (or any other non-previewable type): still return a
            # presigned URL for a uniform response shape, but force download
            # since inline viewing isn't meaningful in a browser.
            params["ResponseContentDisposition"] = _content_disposition(
                "attachment", kb_file.original_filename
            )

    try:
        client = get_s3_client()
        signed_url = client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=KB_FILE_SIGNED_URL_EXPIRY_SECONDS,
        )
    except Exception as exc:
        logger.error(
            "Failed to generate signed URL for kb file_id=%s: %s",
            file_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Could not generate file URL")

    return create_success_response(
        KbFileAccessResponse(
            file_id=kb_file.id,
            filename=kb_file.original_filename,
            file_type=kb_file.file_type,
            action=action,
            url=signed_url,
            preview_supported=preview_supported,
            expires_in=KB_FILE_SIGNED_URL_EXPIRY_SECONDS,
        ),
        "File access URL generated",
    )


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
    user=Depends(require_config_or_api_key),
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

    if settings.S3_KB_BUCKET:
        gcs_path = gcs_kb_path(workspace_id, kb_id, file_id, filename)
        try:
            import io
            upload_kb_file_to_s3(
                settings.S3_KB_BUCKET, gcs_path, io.BytesIO(content)
            )
            kb_file.s3_path = gcs_path
        except Exception as e:
            logger.error("S3 upload failed for file_id=%s: %s", file_id, e, exc_info=True)
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
    user=Depends(require_config_or_api_key),
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
    user=Depends(require_readonly_or_api_key),
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
    user=Depends(require_readonly_or_api_key),
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
    user=Depends(require_readonly_or_api_key),
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
    user=Depends(require_config_or_api_key),
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
    user=Depends(require_readonly_or_api_key),
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
    user=Depends(require_config_or_api_key),
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
