from __future__ import annotations

from app.core.config import settings
from app.core.logger import logger


def embed_text_for_rag(text: str) -> list[float]:
    """
    Generate a single embedding for RAG retrieval.

    Primary: OpenAI text-embedding-ada-002 (1536 dims).
    Fallback: Gemini embedding model (when OPENAI_API_KEY is absent).
    """
    if settings.OPENAI_API_KEY:
        try:
            return _embed_openai_ada002(text)
        except Exception as openai_err:
            logger.warning(
                "OpenAI ada-002 embedding failed; falling back to Gemini: %s",
                str(openai_err)[:200],
            )

    if settings.GEMINI_API_KEY:
        try:
            from app.services.gemini_service import gemini_service

            return gemini_service.embed_text(
                text=text,
                model_name=settings.RAG_FALLBACK_EMBEDDING_MODEL,
                output_dimensionality=settings.VECTOR_DIMENSION,
                api_key=None,
            )
        except Exception as gemini_err:
            logger.error("Gemini embedding fallback also failed: %s", str(gemini_err)[:200])
            raise

    raise RuntimeError(
        "No embedding provider available. Set OPENAI_API_KEY or GEMINI_API_KEY."
    )


def _embed_openai_ada002(text: str) -> list[float]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return resp.data[0].embedding
