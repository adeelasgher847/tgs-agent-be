from __future__ import annotations

"""
Vector DB session helpers for RAG (pgvector on a separate Postgres URL).

The actual models (RagDocument, RagChunk, VectorBase) live under
`app.models.rag_vector` so that:
- all ORM models are grouped in the models package
- RAG tables still use their own Base and engine bound to VECTOR_DB_URL
"""

from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.models.rag_vector import VectorBase, RagDocument, RagChunk


_vector_engine = None
_VectorSessionLocal: Optional[sessionmaker] = None


def get_vector_engine():
    """
    Lazily create a SQLAlchemy engine for the vector database.
    Uses settings.VECTOR_DB_URL and fails fast if it's not configured.
    """
    global _vector_engine, _VectorSessionLocal

    if _vector_engine is not None:
        return _vector_engine

    if not settings.VECTOR_DB_URL:
        raise RuntimeError(
            "VECTOR_DB_URL is not configured. "
            "Set it in .env to enable RAG vector storage."
        )

    _vector_engine = create_engine(settings.VECTOR_DB_URL, pool_pre_ping=True)
    _VectorSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_vector_engine)
    return _vector_engine


def get_vector_session() -> Session:
    """
    Get a SQLAlchemy session bound to the vector DB.
    """
    global _VectorSessionLocal
    if _VectorSessionLocal is None:
        get_vector_engine()
    assert _VectorSessionLocal is not None
    return _VectorSessionLocal()


def init_vector_db() -> None:
    """
    Create RAG tables in the vector database.

    This is meant to be called from a one-off Python script or migration tool,
    NOT automatically at app startup (to keep control in your hands).
    """
    engine = get_vector_engine()
    VectorBase.metadata.create_all(bind=engine)

