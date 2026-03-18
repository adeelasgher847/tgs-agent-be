from __future__ import annotations

import uuid
from typing import Optional

from app.core.config import settings
from app.core.logger import logger
from app.services.openai_service import openai_service
from app.services.rag_service import rag_service


def build_rag_context_block(
    user_text: str,
    tenant_id: Optional[uuid.UUID],
    agent_id: Optional[uuid.UUID],
) -> str:
    """
    Build a RAG context block string for use inside system prompts.

    This lives in the voice layer so that:
    - RAG infra (Pinecone, chunking, embeddings) stays in services.
    - Voice agents can control how/when RAG is called and how context
      is framed for the model.
    """
    text = (user_text or "").strip()
    if not text:
        # No user text to ground the query – be explicit that nothing is available.
        return """
# KNOWLEDGE BASE CONTEXT
No relevant knowledge base entries were found for this query.
If the user asks for specific factual, pricing, or policy details you do not see in the conversation history,
respond that this information is not available instead of guessing or inventing details.
"""

    # If RAG is not configured or we don't know the tenant, don't even try.
    if not tenant_id or not settings.PINECONE_API_KEY:
        return """
# KNOWLEDGE BASE CONTEXT
No relevant knowledge base entries were found for this query.
If the user asks for specific factual, pricing, or policy details you do not see in the conversation history,
respond that this information is not available instead of guessing or inventing details.
"""

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
            top_k=5,
        )

        # Drop very low-confidence hits if score is present.
        filtered = []
        for c in rag_chunks:
            score = c.score or 0.0
            if score >= 0.4:
                filtered.append(c)
        rag_chunks = filtered

        if not rag_chunks:
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
"""

        rag_context = rag_service.format_rag_context(rag_chunks)

        logger.info(
            "RAG retrieve: tenant_id=%s agent_id=%s text_len=%d results=%d",
            tenant_id,
            agent_id,
            len(text),
            len(rag_chunks),
        )

        return f"""
# KNOWLEDGE BASE CONTEXT
You have access to company knowledge retrieved for this specific tenant and agent.
Use ONLY this context (plus conversation history) for factual, policy, and business details.
If the answer is not clearly supported by this context, say that you are not sure or that the information is not available.

{rag_context}
"""

    except Exception as e:
        logger.error("RAG retrieval failed; continuing without context: %s", e, exc_info=True)
        return """
# KNOWLEDGE BASE CONTEXT
No relevant knowledge base entries were found for this query.
If the user asks for specific factual, pricing, or policy details you do not see in the conversation history,
respond that this information is not available instead of guessing or inventing details.
"""

