from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


# ── Knowledge Base CRUD ───────────────────────────────────────────────────────

class KbCreate(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Display name for the knowledge base.",
        examples=["Product FAQ"],
    )
    description: str | None = Field(
        None,
        description="Optional human-readable summary of what this KB contains.",
        examples=["Answers to common pricing and refund questions"],
    )


class KbUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class KbOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class KbListItem(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None = None
    file_count: int
    total_chunk_count: int
    created_at: datetime
    call_flow_ids: List[uuid.UUID] = Field(
        default_factory=list,
        description="IDs of non-deleted call flows this KB is currently attached to.",
    )


class KbList(BaseModel):
    knowledge_bases: List[KbListItem]
    total: int


# ── KB detail with files ──────────────────────────────────────────────────────

class KbFileOut(BaseModel):
    id: uuid.UUID
    filename: str
    size_bytes: int | None = None
    size_mb: str | None = None
    status: str
    chunk_count: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class KbDetail(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    files: List[KbFileOut]
    call_flow_ids: List[uuid.UUID] = Field(
        default_factory=list,
        description="IDs of non-deleted call flows this KB is currently attached to.",
    )


# ── Call-flow KB linking ──────────────────────────────────────────────────────

class FlowKbUpdate(BaseModel):
    kb_ids: List[uuid.UUID]


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
    chunk_count: int | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}


# ── Legacy schemas (kept for backward compatibility with existing callers) ────

class KnowledgeBaseIngestTextRequest(BaseModel):
    title: str = Field(..., description="Human-readable title for the KB document")
    source_type: str = Field(..., description="Source type (e.g. html, faq, kb_text_dir)")
    source_ref: str = Field(..., description="Source reference used for de-duplication")
    full_text: str = Field(..., description="Normalized text content to embed")
    version: str = Field(default="v1")
    agent_id: uuid.UUID | None = None
    chunk_max_chars: int = Field(default=1000, ge=200, le=10000)
    chunk_overlap_chars: int = Field(default=100, ge=0, le=2000)


class KnowledgeBaseIngestTextResponse(BaseModel):
    document_id: uuid.UUID


class KnowledgeBaseDocumentOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class KnowledgeBaseDocumentList(BaseModel):
    documents: List[KnowledgeBaseDocumentOut]
    total: int


class KnowledgeBaseRetrievePreviewRequest(BaseModel):
    user_text: str
    agent_id: uuid.UUID | None = None
    top_k: int = Field(default=5, ge=1, le=25)


class KnowledgeBaseRetrievedChunkOut(BaseModel):
    chunk_n: int
    score: float | None = None
    source_title: str | None = None
    source_ref: str | None = None


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
