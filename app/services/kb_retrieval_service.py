"""
Real-time KB retrieval layer for per-call-turn context injection.

Called before every LLM invocation when flow.knowledge_base_ids is non-empty.
Embeddings and retrieved results are cached in Redis for 300 s.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger


@dataclass
class RetrievedChunk:
    content: str
    score: float
    metadata: dict


# ── Embedding cache ───────────────────────────────────────────────────────────

async def _get_embedding_cached(
    transcript: str,
    redis_client,
) -> list[float]:
    """Embed transcript text, caching the vector in Redis for 300 s."""
    cache_key = "kb:emb:" + hashlib.sha256(transcript.encode()).hexdigest()

    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug("Redis embedding cache read failed: %s", e)

    from app.services.embedding_service import embed_text_for_rag

    loop = asyncio.get_running_loop()
    embedding: list[float] = await loop.run_in_executor(None, embed_text_for_rag, transcript)

    if redis_client:
        try:
            await redis_client.set(cache_key, json.dumps(embedding), ex=300)
        except Exception as e:
            logger.debug("Redis embedding cache write failed: %s", e)

    return embedding


# ── Per-KB pgvector query ─────────────────────────────────────────────────────

async def _query_single_kb(
    db: Session,
    kb_id: uuid.UUID,
    vec_str: str,
    top_k: int,
) -> List[RetrievedChunk]:
    """
    Run cosine-similarity search against one KB.
    Returns empty list on failure so asyncio.gather partial failures are safe.
    """
    stmt = text(
        """
        SELECT content,
               1 - (embedding::vector <=> CAST(:vec AS vector)) AS score,
               metadata AS chunk_metadata
        FROM kbchunk
        WHERE kb_id = :kb_id
          AND embedding IS NOT NULL
        ORDER BY embedding::vector <=> CAST(:vec AS vector)
        LIMIT :top_k
        """
    )
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(
        None,
        lambda: db.execute(
            stmt,
            {"vec": vec_str, "kb_id": str(kb_id), "top_k": top_k},
        ).fetchall(),
    )

    results: List[RetrievedChunk] = []
    for row in rows:
        meta = row.chunk_metadata or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        results.append(
            RetrievedChunk(
                content=row.content,
                score=float(row.score or 0.0),
                metadata=meta,
            )
        )
    return results


# ── Prompt block formatter ────────────────────────────────────────────────────

def format_kb_context_block(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into the injection block required by the ticket spec."""
    if not chunks:
        return ""
    parts = ["--- KNOWLEDGE BASE CONTEXT ---"]
    for chunk in chunks:
        parts.append(chunk.content.strip())
        parts.append("---")
    parts[-1] = "--- END CONTEXT ---"
    return "\n".join(parts)


# ── Main entry point ──────────────────────────────────────────────────────────

async def retrieve_kb_context_for_turn(
    transcript: str,
    kb_ids: List[uuid.UUID],
    db: Session,
    redis_client=None,
) -> tuple[str, float]:
    """
    Embed transcript, query all attached KBs in parallel, return (context_block, latency_ms).

    Cache key: sha256(transcript + ':' + ':'.join(sorted(kb_ids))), TTL 300 s.
    Fails open: returns ("", latency_ms) on any error so the call is never blocked.
    """
    if not transcript or not kb_ids:
        return "", 0.0

    # Normalise: JSONB stores UUIDs as strings; ensure we always have uuid.UUID objects.
    normalised_ids: List[uuid.UUID] = []
    for k in kb_ids:
        try:
            normalised_ids.append(k if isinstance(k, uuid.UUID) else uuid.UUID(str(k)))
        except (ValueError, AttributeError):
            logger.warning("kb_retrieval: skipping invalid kb_id=%r", k)
    if not normalised_ids:
        return "", 0.0

    kb_ids = normalised_ids
    t0 = time.perf_counter()

    cache_key = (
        "kb:ctx:"
        + hashlib.sha256(
            (transcript + ":" + ":".join(sorted(str(k) for k in kb_ids))).encode()
        ).hexdigest()
    )

    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    "kb_retrieval cache_hit=true latency_ms=%.1f kb_count=%d",
                    latency_ms,
                    len(kb_ids),
                )
                return json.loads(cached), latency_ms
        except Exception as e:
            logger.debug("Redis result cache read failed: %s", e)

    try:
        embedding = await _get_embedding_cached(transcript, redis_client)
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "kb_retrieval embedding_failed=true latency_ms=%.1f error=%s",
            latency_ms,
            str(e)[:200],
        )
        return "", latency_ms

    vec_str = "[" + ",".join(str(f) for f in embedding) + "]"
    top_k = settings.RAG_TOP_K

    # Query all KBs in parallel; log and skip individual failures
    raw_results = await asyncio.gather(
        *[_query_single_kb(db, kb_id, vec_str, top_k) for kb_id in kb_ids],
        return_exceptions=True,
    )

    all_chunks: List[RetrievedChunk] = []
    for kb_id, result in zip(kb_ids, raw_results):
        if isinstance(result, Exception):
            logger.error(
                "kb_retrieval partial_failure kb_id=%s error=%s",
                kb_id,
                str(result)[:200],
            )
            continue
        all_chunks.extend(result)

    # Merge across KBs, sort by cosine score descending, keep global top 5
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    top_chunks = all_chunks[:5]

    context_block = format_kb_context_block(top_chunks)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "kb_retrieval latency_ms=%.1f chunks=%d kb_count=%d cache_hit=false",
        latency_ms,
        len(top_chunks),
        len(kb_ids),
    )

    if redis_client and context_block:
        try:
            await redis_client.set(cache_key, json.dumps(context_block), ex=300)
        except Exception as e:
            logger.debug("Redis result cache write failed: %s", e)

    return context_block, latency_ms
