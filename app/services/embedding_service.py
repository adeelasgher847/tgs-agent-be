from __future__ import annotations

import asyncio
import time

from app.core.config import settings
from app.core.logger import logger


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
    # Route through openai_service's cached-by-api-key client dict (see
    # OpenAIService.get_client) instead of app.core.openai_client.get_openai_client()
    # directly. get_openai_client() constructs a brand-new OpenAI(...) client
    # (and therefore a brand-new httpx.Client / connection pool) on every call —
    # previously this function called it fresh on every single embedding, so no
    # TCP/TLS connection was ever kept warm across calls, and the very first
    # embedding call in a freshly started worker process additionally pays for
    # one-time httpx/certifi import + CA-bundle parsing + cold DNS resolution on
    # top of the handshake, which is what pushed the first-in-process retrieval
    # past RAG_KB_RETRIEVAL_TIMEOUT_SEC in production while every later, already-
    # warm call landed around ~200ms. Reusing the cached client lets the
    # underlying connection pool keep the TCP/TLS session alive across calls.
    from app.services.openai_service import openai_service

    client = openai_service.get_client()
    resp = client.embeddings.create(
        model=settings.RAG_EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


async def warm_embedding_client() -> None:
    """
    Fire a single, cheap embedding call to pre-establish the cached OpenAI
    client's TCP/TLS connection (and pay any one-time module-import /
    CA-bundle-parsing / DNS-resolution cost) before the first real call
    arrives on this worker process.

    Intended to be scheduled as a non-blocking, fire-and-forget background
    task from app startup (`asyncio.create_task`, never awaited inline) — see
    app/main.py's lifespan. Must never raise: this is best-effort latency
    warm-up, not a startup dependency, so any failure (no API key configured,
    transient network issue, etc.) is logged and swallowed rather than
    surfaced. Each Uvicorn worker process runs its own lifespan, so this
    naturally warms one client per worker rather than a single shared one.
    """
    if not settings.OPENAI_API_KEY:
        return
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, embed_text_for_rag, "warmup")
        logger.info(
            "kb_retrieval embedding client warm-up complete in %.1fms",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as exc:
        logger.warning(
            "kb_retrieval embedding client warm-up failed (non-fatal): %s", exc
        )
