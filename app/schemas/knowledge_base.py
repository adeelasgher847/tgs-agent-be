from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


# ── Knowledge Base CRUD ───────────────────────────────────────────────────────

class KbCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class KbOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class KbList(BaseModel):
    knowledge_bases: List[KbOut]
    total: int


# ── File upload ───────────────────────────────────────────────────────────────

class KbFileUploadResponse(BaseModel):
    job_id: str
    file_id: uuid.UUID


# ── Text ingest ───────────────────────────────────────────────────────────────

class KbTextIngestRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)


class KbTextIngestResponse(BaseModel):
    kb_id: uuid.UUID
    chunk_count: int


# ── File status ───────────────────────────────────────────────────────────────

class KbFileStatusOut(BaseModel):
    file_id: uuid.UUID
    status: str
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


# ── Legacy schemas (kept for backward compatibility with existing callers) ────

class KnowledgeBaseIngestTextRequest(BaseModel):
    title: str = Field(..., description="Human-readable title for the KB document")
    source_type: str = Field(..., description="Source type (e.g. html, faq, kb_text_dir)")
    source_ref: str = Field(..., description="Source reference used for de-duplication")
    full_text: str = Field(..., description="Normalized text content to embed")
    version: str = Field(default="v1")
    agent_id: Optional[uuid.UUID] = None
    chunk_max_chars: int = Field(default=1000, ge=200, le=10000)
    chunk_overlap_chars: int = Field(default=100, ge=0, le=2000)


class KnowledgeBaseIngestTextResponse(BaseModel):
    document_id: uuid.UUID


class KnowledgeBaseDocumentOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


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


# ── Search ────────────────────────────────────────────────────────────────────

class KbSearchResultItem(BaseModel):
    content: str
    score: float
    metadata: dict


class KbSearchResponse(BaseModel):
    results: List[KbSearchResultItem]
