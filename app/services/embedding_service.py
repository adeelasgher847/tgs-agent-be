from __future__ import annotations

from app.core.config import settings
from app.core.logger import logger
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service


def embed_text_for_rag(text: str) -> list[float]:
    """
    Generate embeddings for RAG with provider fallback.

    Primary: OpenAI model from settings.RAG_EMBEDDING_MODEL
    Fallback: Gemini model from settings.RAG_FALLBACK_EMBEDDING_MODEL
    """
    try:
        return openai_service.embed_text(
            text=text,
            model_name=settings.RAG_EMBEDDING_MODEL,
            api_key=None,
        )
    except Exception as openai_error:
        fallback_provider = (settings.RAG_FALLBACK_EMBEDDING_PROVIDER or "").strip().lower()
        if fallback_provider != "gemini" or not settings.GEMINI_API_KEY:
            raise

        logger.warning(
            "OpenAI embedding failed for model '%s'; falling back to Gemini model '%s'. "
            "error=%s",
            settings.RAG_EMBEDDING_MODEL,
            settings.RAG_FALLBACK_EMBEDDING_MODEL,
            str(openai_error)[:200],
        )
        return gemini_service.embed_text(
            text=text,
            model_name=settings.RAG_FALLBACK_EMBEDDING_MODEL,
            output_dimensionality=settings.VECTOR_DIMENSION,
            api_key=None,
        )
