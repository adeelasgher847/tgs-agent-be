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
from typing import List

from sqlalchemy import text

from app.core.config import settings
from app.core.logger import logger


@dataclass
class RetrievedChunk:
    content: str
    score: float
    metadata: dict


# ── Embedding cache ───────────────────────────────────────────────────────────

def build_embedding_cache_key(
    transcript: str,
    model_id: str | None = None,
    embedding_provider: str | None = None,
) -> str:
    """
    Build isolated cache key for text embedding vectors.
    Includes text, embedding provider, model version, and vector dimension.
    """
    norm_text = (transcript or "").strip().lower()
    provider_slug = embedding_provider or ("openai" if settings.OPENAI_API_KEY else "gemini")
    model = model_id or (
        settings.RAG_EMBEDDING_MODEL
        if provider_slug == "openai"
        else settings.RAG_FALLBACK_EMBEDDING_MODEL
    )
    dim = settings.VECTOR_DIMENSION
    raw = f"{provider_slug}:{model}:{dim}:{norm_text}"
    return "kb:emb:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _get_embedding_cached(
    transcript: str,
    redis_client,
    model_id: str | None = None,
    embedding_provider: str | None = None,
) -> list[float]:
    """Embed transcript text, caching the vector in Redis for 300 s."""
    cache_key = build_embedding_cache_key(
        transcript, model_id=model_id, embedding_provider=embedding_provider
    )

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
    kb_id: uuid.UUID,
    vec_str: str,
    top_k: int,
) -> List[RetrievedChunk]:
    """
    Run cosine-similarity search against one KB.
    Opens its own short-lived SessionLocal so concurrent gather() calls never
    share a Session across threads (SQLAlchemy Session is not thread-safe).
    Returns empty list on failure so asyncio.gather partial failures are safe.
    """
    from app.db.session import SessionLocal

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

    def _run() -> list:
        with SessionLocal() as session:
            return session.execute(
                stmt,
                {"vec": vec_str, "kb_id": str(kb_id), "top_k": top_k},
            ).fetchall()

    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, _run)

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

_KB_CONTEXT_INSTRUCTION = (
    "The following information comes from the business's uploaded Knowledge Base. "
    "Use this information when answering relevant factual questions. Do not ignore it. "
    "Do not add facts that are not supported by it."
)


def format_kb_context_block(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into the injection block required by the ticket spec."""
    if not chunks:
        return ""
    parts = ["--- KNOWLEDGE BASE CONTEXT ---", _KB_CONTEXT_INSTRUCTION]
    for chunk in chunks:
        parts.append(chunk.content.strip())
        parts.append("---")
    parts[-1] = "--- END CONTEXT ---"
    return "\n".join(parts)


# ── Retrieval result cache ───────────────────────────────────────────────────

def build_retrieval_cache_key(
    transcript: str,
    kb_ids: List[uuid.UUID | str],
    top_k: int | None = None,
    score_threshold: float | None = None,
    kb_revisions: dict[str, str] | None = None,
    tenant_id: str | uuid.UUID | None = None,
    agent_id: str | uuid.UUID | None = None,
    model_id: str | None = None,
    embedding_provider: str | None = None,
) -> str:
    """
    Build robust, fully-isolated cache key for RAG context retrieval results.
    Includes normalized query, embedding model version, sorted KB IDs, top_k,
    similarity threshold, sorted KB content revisions, and tenant/agent scope.
    """
    norm_text = (transcript or "").strip().lower()
    provider_slug = embedding_provider or ("openai" if settings.OPENAI_API_KEY else "gemini")
    model = model_id or (
        settings.RAG_EMBEDDING_MODEL
        if provider_slug == "openai"
        else settings.RAG_FALLBACK_EMBEDDING_MODEL
    )
    k_val = top_k if top_k is not None else settings.RAG_TOP_K
    thresh_val = score_threshold if score_threshold is not None else settings.RAG_SCORE_THRESHOLD
    sorted_kbs = sorted(str(k) for k in kb_ids)

    revs_str = ""
    if kb_revisions:
        revs_str = "|".join(f"{k}:{kb_revisions.get(k, 'v1')}" for k in sorted_kbs)

    key_components = [
        f"q:{norm_text}",
        f"emb:{provider_slug}:{model}",
        f"kbs:{','.join(sorted_kbs)}",
        f"top_k:{k_val}",
        f"thresh:{thresh_val:.4f}",
    ]
    if revs_str:
        key_components.append(f"revs:{revs_str}")
    if tenant_id:
        key_components.append(f"tenant:{tenant_id}")
    if agent_id:
        key_components.append(f"agent:{agent_id}")

    raw = "|".join(key_components)
    return "kb:ctx:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def get_kb_revision(kb_id: uuid.UUID | str, redis_client=None) -> str | None:
    """
    Fetch current content revision string for a given KB.
    Returns None if Redis is unavailable or on error (signaling cache bypass).
    """
    if not redis_client:
        return None
    try:
        rev = await redis_client.get(f"kb:rev:{kb_id}")
        if rev is None:
            return "v1"
        return rev if isinstance(rev, str) else rev.decode("utf-8")
    except Exception as e:
        logger.warning("Failed to read kb revision from Redis for kb_id=%s (bypassing cache): %s", kb_id, e)
        return None


async def invalidate_kb_cache(kb_id: uuid.UUID | str, redis_client=None) -> str | None:
    """
    Atomically update KB revision in Redis, invalidating all future retrieval
    lookups for any query hitting this KB across all tenants and workers.
    Returns the new revision string on success, or None on Redis failure.
    """
    if not redis_client:
        logger.warning("Cannot invalidate KB cache for kb_id=%s: Redis client is not available", kb_id)
        return None
    new_rev = str(time.time_ns())
    try:
        await redis_client.set(f"kb:rev:{kb_id}", new_rev)
        logger.info("Invalidated KB cache for kb_id=%s new_revision=%s", kb_id, new_rev)
        return new_rev
    except Exception as e:
        logger.error("Failed to invalidate KB cache in Redis for kb_id=%s: %s", kb_id, e)
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

async def retrieve_kb_context_for_turn(
    transcript: str,
    kb_ids: List[uuid.UUID],
    redis_client=None,
    tenant_id: uuid.UUID | str | None = None,
    agent_id: uuid.UUID | str | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
    kb_revisions: dict[str, str] | None = None,
) -> tuple[str, float]:
    """
    Embed transcript, query all attached KBs in parallel, return (context_block, latency_ms).

    Cache key: sha256 of all query, model, KB ID, threshold, and revision parameters (TTL 300s).
    Fails open on errors (returns ("", latency_ms) so the call is never blocked).
    Fails closed on Redis revision unavailability (bypasses cache and queries DB directly).
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

    effective_top_k = top_k if top_k is not None else settings.RAG_TOP_K
    effective_threshold = (
        score_threshold if score_threshold is not None else settings.RAG_SCORE_THRESHOLD
    )

    # Fetch KB revisions if not explicitly supplied
    if kb_revisions is None and redis_client:
        try:
            rev_tasks = [get_kb_revision(k, redis_client) for k in kb_ids]
            rev_list = await asyncio.gather(*rev_tasks)
            if all(r is not None for r in rev_list):
                kb_revisions = {str(k): str(r) for k, r in zip(kb_ids, rev_list)}
            else:
                # One or more KB revisions failed to resolve -> bypass cache
                kb_revisions = None
        except Exception:
            kb_revisions = None

    cache_key = None
    if redis_client and kb_revisions is not None:
        cache_key = build_retrieval_cache_key(
            transcript=transcript,
            kb_ids=kb_ids,
            top_k=effective_top_k,
            score_threshold=effective_threshold,
            kb_revisions=kb_revisions,
            tenant_id=tenant_id,
            agent_id=agent_id,
        )
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

    t_embed_start = time.perf_counter()
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
    embedding_latency_ms = (time.perf_counter() - t_embed_start) * 1000

    vec_str = "[" + ",".join(str(f) for f in embedding) + "]"

    # Query all KBs in parallel; each call opens its own session (thread-safe).
    t_search_start = time.perf_counter()
    raw_results = await asyncio.gather(
        *[_query_single_kb(kb_id, vec_str, effective_top_k) for kb_id in kb_ids],
        return_exceptions=True,
    )
    vector_search_latency_ms = (time.perf_counter() - t_search_start) * 1000

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

    # Merge across KBs, sort by cosine similarity score descending.
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    candidates_found = len(all_chunks)

    # Relevance floor
    relevant_chunks = [c for c in all_chunks if c.score >= effective_threshold]
    top_chunks = relevant_chunks[:5]

    context_block = format_kb_context_block(top_chunks)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "kb_retrieval latency_ms=%.1f embedding_latency_ms=%.1f vector_search_latency_ms=%.1f "
        "chunks=%d kb_count=%d cache_hit=false candidates_found=%d "
        "candidates_above_threshold=%d threshold=%.2f",
        latency_ms,
        embedding_latency_ms,
        vector_search_latency_ms,
        len(top_chunks),
        len(kb_ids),
        candidates_found,
        len(relevant_chunks),
        effective_threshold,
    )

    if redis_client and cache_key and context_block:
        try:
            await redis_client.set(cache_key, json.dumps(context_block), ex=300)
        except Exception as e:
            logger.debug("Redis result cache write failed: %s", e)

    return context_block, latency_ms
