from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


class KnowledgeBaseIngestTextRequest(BaseModel):
    title: str = Field(..., description="Human-readable title for the KB document")
    source_type: str = Field(..., description="Source type (e.g. html, faq, kb_text_dir, db_rows)")
    source_ref: str = Field(..., description="Source reference (URL/slug/path), used for de-duplication")
    full_text: str = Field(..., description="Normalized text content to embed")

    version: str = Field(default="v1", description="Version label for this document ingest")
    agent_id: Optional[uuid.UUID] = Field(default=None, description="If provided, document is scoped to an agent")

    # Baseline chunking controls (char-based).
    chunk_max_chars: int = Field(default=1000, ge=200, le=10000)
    chunk_overlap_chars: int = Field(default=100, ge=0, le=2000)


class KnowledgeBaseIngestTextResponse(BaseModel):
    document_id: uuid.UUID


class KnowledgeBaseDocumentOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    agent_id: Optional[uuid.UUID] = None

    title: str
    source_type: str
    source_ref: str
    version: str

    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None


class KnowledgeBaseDocumentList(BaseModel):
    documents: List[KnowledgeBaseDocumentOut]
    total: int


class KnowledgeBaseRetrievePreviewRequest(BaseModel):
    user_text: str
    agent_id: Optional[uuid.UUID] = None
    top_k: int = Field(default=5, ge=1, le=25)


class KnowledgeBaseRetrievedChunkOut(BaseModel):
    chunk_n: int
    score: Optional[float] = None
    source_title: Optional[str] = None
    source_ref: Optional[str] = None


class KnowledgeBaseRetrievePreviewResponse(BaseModel):
    context_block: str
    retrieved_chunks: List[KnowledgeBaseRetrievedChunkOut]

