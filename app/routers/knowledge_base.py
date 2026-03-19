from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.core.logger import logger
from app.core.config import settings
from app.models.agent import Agent
from app.models.knowledge_base_document import KnowledgeBaseDocument
from app.models.knowledge_base_chunk import KnowledgeBaseChunk
from app.services.embedding_service import embed_text_for_rag
from app.services.rag_service import rag_service
from app.utils.response import create_success_response

from app.schemas.base import SuccessResponse
from app.schemas.knowledge_base import (
    KnowledgeBaseIngestTextRequest,
    KnowledgeBaseIngestTextResponse,
    KnowledgeBaseDocumentList,
    KnowledgeBaseRetrievePreviewRequest,
    KnowledgeBaseRetrievePreviewResponse,
    KnowledgeBaseRetrievedChunkOut,
)


router = APIRouter()


@router.post(
    "/documents/ingest-text",
    response_model=SuccessResponse[KnowledgeBaseIngestTextResponse],
)
def ingest_text_document(
    request: KnowledgeBaseIngestTextRequest,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Admin endpoint: ingest a normalized text document into Pinecone-backed RAG.
    """
    tenant_id = user.current_tenant_id
    if tenant_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    agent_id: Optional[uuid.UUID] = request.agent_id
    if agent_id is not None:
        agent = (
            db.query(Agent)
            .filter(Agent.id == agent_id, Agent.tenant_id == tenant_id, Agent.is_deleted == False)  # noqa: E712
            .first()
        )
        if not agent:
            raise HTTPException(
                status_code=400,
                detail="agent_id not found or does not belong to your tenant",
            )

    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured")
    if not settings.PINECONE_API_KEY:
        raise HTTPException(status_code=400, detail="PINECONE_API_KEY is not configured")

    def embedding_func(text: str):
        return embed_text_for_rag(text)

    document_id = rag_service.ingest_document(
        tenant_id=tenant_id,
        agent_id=agent_id,
        title=request.title,
        source_type=request.source_type,
        source_ref=request.source_ref,
        full_text=request.full_text,
        embedding_func=embedding_func,
        version=request.version,
        db_session=db,
        replace_existing=True,
        max_chars=request.chunk_max_chars,
        overlap_chars=request.chunk_overlap_chars,
    )

    return create_success_response(
        KnowledgeBaseIngestTextResponse(document_id=document_id),
        "Document ingested successfully",
    )


@router.get(
    "/documents",
    response_model=SuccessResponse[KnowledgeBaseDocumentList],
)
def list_documents(
    agent_id: Optional[uuid.UUID] = None,
    active_only: bool = True,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    tenant_id = user.current_tenant_id
    if tenant_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    q = db.query(KnowledgeBaseDocument).filter(KnowledgeBaseDocument.tenant_id == tenant_id)
    if agent_id is not None:
        q = q.filter(KnowledgeBaseDocument.agent_id == agent_id)
    if active_only:
        q = q.filter(KnowledgeBaseDocument.is_active == True)  # noqa: E712

    documents = q.order_by(KnowledgeBaseDocument.updated_at.desc().nullslast()).all()
    out_documents = [
        {
            "id": d.id,
            "tenant_id": d.tenant_id,
            "agent_id": d.agent_id,
            "title": d.title,
            "source_type": d.source_type,
            "source_ref": d.source_ref,
            "version": d.version,
            "is_active": d.is_active,
            "created_at": d.created_at,
            "updated_at": d.updated_at,
        }
        for d in documents
    ]

    return create_success_response(
        KnowledgeBaseDocumentList(documents=out_documents, total=len(out_documents)),
        "KB documents fetched successfully",
    )


@router.post(
    "/retrieve-preview",
    response_model=SuccessResponse[KnowledgeBaseRetrievePreviewResponse],
)
def retrieve_preview(
    request: KnowledgeBaseRetrievePreviewRequest,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Admin endpoint: runs retrieval and returns the context block + retrieved chunk previews.
    Useful for debugging prompt/KB issues.
    """
    tenant_id = user.current_tenant_id
    if tenant_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    def embedding_func(text: str):
        return embed_text_for_rag(text)

    rag_chunks = rag_service.retrieve(
        user_text=request.user_text,
        tenant_id=tenant_id,
        agent_id=request.agent_id,
        embedding_func=embedding_func,
        top_k=request.top_k,
    )

    # Apply the same "low confidence" gating as voice layer.
    filtered = []
    for c in rag_chunks:
        score = c.score or 0.0
        if score >= settings.RAG_SCORE_THRESHOLD:
            filtered.append(c)

    context_block = rag_service.format_rag_context(
        filtered,
        max_chars=settings.RAG_MAX_CONTEXT_CHARS,
    )

    retrieved_chunks_out = []
    for i, c in enumerate(filtered, start=1):
        retrieved_chunks_out.append(
            KnowledgeBaseRetrievedChunkOut(
                chunk_n=i,
                score=c.score,
                source_title=c.source_title,
                source_ref=c.source_ref,
            )
        )

    return create_success_response(
        KnowledgeBaseRetrievePreviewResponse(
            context_block=context_block,
            retrieved_chunks=retrieved_chunks_out,
        ),
        "RAG retrieve preview generated",
    )


@router.delete(
    "/documents/{document_id}",
    response_model=SuccessResponse[dict],
)
def delete_document(
    document_id: uuid.UUID,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Admin endpoint: delete a KB document's vectors from Pinecone and remove chunk inventory.
    """
    tenant_id = user.current_tenant_id
    if tenant_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    doc = (
        db.query(KnowledgeBaseDocument)
        .filter(KnowledgeBaseDocument.id == document_id, KnowledgeBaseDocument.tenant_id == tenant_id)
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_rows = db.query(KnowledgeBaseChunk).filter(KnowledgeBaseChunk.document_id == document_id).all()
    vector_ids = [c.vector_id for c in chunk_rows if c.vector_id]

    try:
        rag_service.delete_vectors(vector_ids=vector_ids)
    except Exception as e:
        logger.warning("Failed to delete vectors from Pinecone: %s", e, exc_info=True)

    # Remove chunk inventory rows.
    db.query(KnowledgeBaseChunk).filter(KnowledgeBaseChunk.document_id == document_id).delete(synchronize_session=False)
    # Mark inactive so it doesn't appear as active in listings.
    doc.is_active = False
    db.commit()

    return create_success_response(
        {"document_id": str(document_id), "deleted_chunks": len(vector_ids)},
        "Document deleted successfully",
    )

