from __future__ import annotations

from app.core.config import settings
from app.core.logger import logger
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service


def embed_text_for_rag(text: str) -> list[float]:
    """
    Generate embeddings for RAG with provider fallback.

    Primary: Gemini model from settings.RAG_FALLBACK_EMBEDDING_MODEL
    Fallback: OpenAI model from settings.RAG_EMBEDDING_MODEL (optional)
    """
    try:
        return gemini_service.embed_text(
            text=text,
            model_name=settings.RAG_FALLBACK_EMBEDDING_MODEL,
            output_dimensionality=settings.VECTOR_DIMENSION,
            api_key=None,
        )
    except Exception as gemini_error:
        if not settings.OPENAI_API_KEY:
            raise

        logger.warning(
            "Gemini embedding failed for model '%s'; falling back to OpenAI model '%s'. "
            "error=%s",
            settings.RAG_FALLBACK_EMBEDDING_MODEL,
            settings.RAG_EMBEDDING_MODEL,
            str(gemini_error)[:200],
        )
        return openai_service.embed_text(
            text=text,
            model_name=settings.RAG_EMBEDDING_MODEL,
            api_key=None,
        )
