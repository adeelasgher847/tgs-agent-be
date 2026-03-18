from __future__ import annotations

import uuid
from typing import Optional

from app.core.config import settings
from app.core.logger import logger
from app.services.openai_service import openai_service
from app.services.rag_service import rag_service


def build_rag_context_block_with_trace(
    user_text: str,
    tenant_id: Optional[uuid.UUID],
    agent_id: Optional[uuid.UUID],
) -> tuple[str, dict]:
    """
    Build a RAG context block string and a small trace payload for auditing/debugging.
    The trace intentionally excludes full chunk text to keep logs/metadata small.
    """
    raw_text = (user_text or "").strip()
    # Guardrail: prevent extremely long inputs from blowing up latency and prompt size.
    if settings.RAG_MAX_QUERY_CHARS and len(raw_text) > settings.RAG_MAX_QUERY_CHARS:
        text = raw_text[: settings.RAG_MAX_QUERY_CHARS].rstrip()
    else:
        text = raw_text

    trace: dict = {
        "status": None,
        "query_len": len(text),
        "query_truncated": settings.RAG_MAX_QUERY_CHARS is not None and len(raw_text) > settings.RAG_MAX_QUERY_CHARS,
        "top_k": settings.RAG_TOP_K,
        "score_threshold": settings.RAG_SCORE_THRESHOLD,
        "retrieved_preview": [],
        "filtered_count": 0,
        "initial_retrieved_count": 0,
        "timeout": False,
        "error": None,
    }

    if not text:
        trace["status"] = "empty_user_text"
        return """
# KNOWLEDGE BASE CONTEXT
No relevant knowledge base entries were found for this query.
If the user asks for specific factual, pricing, or policy details you do not see in the conversation history,
respond that this information is not available instead of guessing or inventing details.
""", trace

    # If RAG is not configured or we don't know the tenant, don't even try.
    if not tenant_id or not settings.PINECONE_API_KEY:
        trace["status"] = "missing_tenant_or_config"
        return """
# KNOWLEDGE BASE CONTEXT
No relevant knowledge base entries were found for this query.
If the user asks for specific factual, pricing, or policy details you do not see in the conversation history,
respond that this information is not available instead of guessing or inventing details.
""", trace

    try:
        def embedding_func(chunk_text: str):
            return openai_service.embed_text(
                text=chunk_text,
                model_name="text-embedding-3-small",
                api_key=None,  # use settings.OPENAI_API_KEY from config
            )

        rag_chunks = rag_service.retrieve(
            user_text=text,
            tenant_id=tenant_id,
            agent_id=agent_id,
            embedding_func=embedding_func,
            top_k=settings.RAG_TOP_K,
        )

        trace["initial_retrieved_count"] = len(rag_chunks)

        # Drop very low-confidence hits if score is present.
        filtered = []
        for c in rag_chunks:
            score = c.score or 0.0
            if score >= settings.RAG_SCORE_THRESHOLD:
                filtered.append(c)
        rag_chunks = filtered
        trace["filtered_count"] = len(rag_chunks)

        if not rag_chunks:
            trace["status"] = "low_confidence_or_empty_after_threshold"
            logger.info(
                "RAG retrieve: tenant_id=%s agent_id=%s text_len=%d results=0",
                tenant_id,
                agent_id,
                len(text),
            )
            return """
# KNOWLEDGE BASE CONTEXT
No relevant knowledge base entries were found for this query.
If the user asks for specific factual, pricing, or policy details you do not see in the conversation history,
respond that this information is not available instead of guessing or inventing details.
""", trace

        # Build rendered context
        rag_context = rag_service.format_rag_context(
            rag_chunks,
            max_chars=settings.RAG_MAX_CONTEXT_CHARS,
        )

        # Add trace preview (no chunk text)
        retrieved_preview = []
        for i, c in enumerate(rag_chunks, start=1):
            retrieved_preview.append(
                {
                    "chunk_n": i,
                    "title": c.source_title,
                    "ref": c.source_ref,
                    "score": c.score,
                    "vector_id": c.vector_id,
                    "chunk_index": c.chunk_index,
                }
            )
        trace["retrieved_preview"] = retrieved_preview
        trace["status"] = "high_confidence"

        logger.info(
            "RAG retrieve: tenant_id=%s agent_id=%s text_len=%d top_k=%d threshold=%.3f results=%d preview=%s",
            tenant_id,
            agent_id,
            len(text),
            settings.RAG_TOP_K,
            settings.RAG_SCORE_THRESHOLD,
            len(rag_chunks),
            retrieved_preview,
        )

        return f"""
# KNOWLEDGE BASE CONTEXT
You have access to company knowledge retrieved for this specific tenant and agent.
Use ONLY this context (plus conversation history) for factual, policy, and business details.
If you use information from a chunk, cite the chunk number like [1], [2], ...
If the answer is not clearly supported by this context, say that the information is not available (do not guess).

{rag_context}
""", trace

    except Exception as e:
        trace["status"] = "failure"
        trace["error"] = str(e)
        logger.error("RAG retrieval failed; continuing without context: %s", e, exc_info=True)
        return """
# KNOWLEDGE BASE CONTEXT
No relevant knowledge base entries were found for this query.
If the user asks for specific factual, pricing, or policy details you do not see in the conversation history,
respond that this information is not available instead of guessing or inventing details.
""", trace


def build_rag_context_block(
    user_text: str,
    tenant_id: Optional[uuid.UUID],
    agent_id: Optional[uuid.UUID],
) -> str:
    # Backwards compatible wrapper used by existing call paths.
    context_block, _trace = build_rag_context_block_with_trace(
        user_text=user_text,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )
    return context_block

