"""
Knowledge-base ingestion pipeline.

Responsibilities:
- Text extraction from PDF (PyMuPDF), DOCX (python-docx), TXT
- Tiktoken-based chunking: 800 tokens, 100-token overlap, min 50 tokens
- OpenAI text-embedding-ada-002 batch embeddings (100 per call, 1 s sleep, exp backoff on 429)
- GCS streaming upload (never buffers full file in memory)
- DB writes: insert kb_chunks, update kb_files.status / chunk_count
"""
from __future__ import annotations

import asyncio
import io
import json
import uuid
from typing import IO, List

from app.core.config import settings
from app.core.logger import logger

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
CHUNK_MIN_TOKENS = 50
EMBED_BATCH_SIZE = 100
EMBED_SLEEP_BETWEEN_BATCHES = 1.0  # seconds


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=content, filetype="pdf")
    pages: List[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def extract_text_from_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text(content: bytes, file_type: str) -> str:
    ft = (file_type or "").lower().lstrip(".")
    if ft == "pdf":
        return extract_text_from_pdf(content)
    if ft == "docx":
        return extract_text_from_docx(content)
    # TXT — decode best-effort
    return content.decode("utf-8", errors="replace")


# ── Tiktoken chunking ─────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
) -> List[str]:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if not tokens:
        return []

    chunks: List[str] = []
    start = 0
    total = len(tokens)

    while start < total:
        end = min(start + max_tokens, total)
        chunk_tokens = tokens[start:end]
        if len(chunk_tokens) >= min_tokens:
            chunks.append(enc.decode(chunk_tokens))
        elif chunks:
            # Append under-minimum tail to the previous chunk (avoids orphan slivers)
            chunks[-1] = chunks[-1] + " " + enc.decode(chunk_tokens)
        if end >= total:
            break
        start = end - overlap_tokens

    return chunks


# ── OpenAI ada-002 embeddings ─────────────────────────────────────────────────

async def embed_chunks(
    chunks: List[str],
    api_key: str,
    batch_size: int = EMBED_BATCH_SIZE,
) -> List[List[float]]:
    """
    Embed a list of text chunks using OpenAI text-embedding-ada-002.

    Rate-limit handling:
    - 1-second sleep between every batch of 100 chunks
    - Exponential back-off (1 s, 2 s, 4 s, 8 s, 16 s) on HTTP 429
    """
    from openai import AsyncOpenAI, RateLimitError

    client = AsyncOpenAI(api_key=api_key)
    all_embeddings: List[List[float]] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = await client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                all_embeddings.extend([d.embedding for d in resp.data])
                break
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "OpenAI 429 on embedding batch %d/%d; retrying in %ds (attempt %d/%d)",
                    batch_start // batch_size + 1,
                    (len(chunks) + batch_size - 1) // batch_size,
                    wait,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait)

        # Sleep between batches to stay within OpenAI rate limits
        if batch_start + batch_size < len(chunks):
            await asyncio.sleep(EMBED_SLEEP_BETWEEN_BATCHES)

    return all_embeddings


# ── GCS streaming upload ──────────────────────────────────────────────────────

def gcs_kb_path(
    workspace_id: uuid.UUID,
    kb_id: uuid.UUID,
    file_id: uuid.UUID,
    filename: str,
) -> str:
    return f"{settings.GCS_KB_PREFIX}/{workspace_id}/{kb_id}/{file_id}/{filename}"


def upload_kb_file_to_gcs(
    bucket_name: str,
    gcs_path: str,
    file_obj: IO[bytes],
    content_type: str = "application/octet-stream",
) -> None:
    """Stream-upload a file-like object to GCS. Never buffers the full file in memory."""
    from google.cloud import storage  # type: ignore

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_file(file_obj, content_type=content_type)


# ── Core ingestion logic (called by ARQ worker and sync text endpoint) ────────

async def run_file_ingestion(
    db,
    file_id: uuid.UUID,
    file_bytes: bytes,
    file_type: str,
    api_key: str,
) -> int:
    """
    Extract → chunk → embed → insert kb_chunks.
    Returns the number of chunks inserted.
    Raises on error; caller is responsible for updating kb_file.status.
    """
    from app.models.kb_file import KbFile
    from app.models.knowledge_base_chunk import KbChunk

    kb_file = db.get(KbFile, file_id)
    if kb_file is None:
        raise ValueError(f"KbFile {file_id} not found")

    text = extract_text(file_bytes, file_type)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text could be extracted from the file")

    embeddings = await embed_chunks(chunks, api_key)

    chunk_rows: List[KbChunk] = []
    for i, (chunk_content, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_rows.append(
            KbChunk(
                id=uuid.uuid4(),
                kb_id=kb_file.kb_id,
                file_id=file_id,
                content=chunk_content,
                # Store as JSON string for the TEXT column; Postgres sees it as vector-compatible string
                embedding=json.dumps(embedding),
                chunk_metadata={"chunk_index": i, "source": "file", "filename": kb_file.original_filename},
            )
        )

    db.add_all(chunk_rows)
    db.flush()  # get IDs without committing yet
    return len(chunk_rows)


async def run_text_ingestion(
    db,
    kb_id: uuid.UUID,
    content: str,
    api_key: str,
) -> int:
    """
    Chunk + embed raw text and insert into kb_chunks directly (no KbFile row).
    Returns the number of chunks inserted.
    """
    from app.models.knowledge_base_chunk import KbChunk

    chunks = chunk_text(content)
    if not chunks:
        raise ValueError("Text content produced no chunks after minimum-token filtering")

    embeddings = await embed_chunks(chunks, api_key)

    chunk_rows: List[KbChunk] = []
    for i, (chunk_content, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_rows.append(
            KbChunk(
                id=uuid.uuid4(),
                kb_id=kb_id,
                file_id=None,
                content=chunk_content,
                embedding=json.dumps(embedding),
                chunk_metadata={"chunk_index": i, "source": "text"},
            )
        )

    db.add_all(chunk_rows)
    db.flush()
    return len(chunk_rows)
