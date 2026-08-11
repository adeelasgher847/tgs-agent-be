from __future__ import annotations

from app.core.config import settings


def embed_text_for_rag(text: str) -> list[float]:
    """
    Generate a single embedding for RAG retrieval.

    Primary and ONLY provider when configured: OpenAI (settings.RAG_EMBEDDING_MODEL,
    1536 dims). This function is used for both ingesting KbChunk.embedding rows and
    embedding live queries against those same rows — every caller depends on the
    returned vector living in the same embedding space as what's already stored in
    the `kbchunk` table (OpenAI's), so silently switching providers on failure would
    return semantically meaningless nearest-neighbors instead of no results.

    Gemini is used ONLY as a fallback when OpenAI is not configured at all
    (no OPENAI_API_KEY) — a deliberate deployment choice, not a mid-call failover.
    If OPENAI_API_KEY *is* configured but the call fails (rate limit, outage, etc.),
    this raises instead of silently falling back to a different embedding space.
    Callers (e.g. kb_retrieval_service.retrieve_kb_context_for_turn) already treat
    any exception here as fail-open: KB context is omitted for that turn and the
    LLM call proceeds without it, rather than being blocked or served bad matches.
    """
    if settings.OPENAI_API_KEY:
        return _embed_openai_ada002(text)

    if settings.GEMINI_API_KEY:
        from app.services.gemini_service import gemini_service

        return gemini_service.embed_text(
            text=text,
            model_name=settings.RAG_FALLBACK_EMBEDDING_MODEL,
            output_dimensionality=settings.VECTOR_DIMENSION,
            api_key=None,
        )

    raise RuntimeError(
        "No embedding provider available. Set OPENAI_API_KEY or GEMINI_API_KEY."
    )


def _embed_openai_ada002(text: str) -> list[float]:
    from app.core.openai_client import get_openai_client

    client = get_openai_client()
    resp = client.embeddings.create(
        model=settings.RAG_EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding
