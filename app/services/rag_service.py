from __future__ import annotations

"""
RAG service: ingestion + retrieval on top of a pgvector-backed Postgres database.

Design goals:
- Keep infra concerns (vector DB, chunking, embeddings) separate from voice flow.
- Allow Person B to inject their own embedding function based on Model/provider config.
- Never touch existing tables or migrations; all RAG tables live in a separate DB.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import uuid

from sqlalchemy import asc
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.services.rag_vector_db import (
    RagDocument,
    RagChunk,
    get_vector_session,
)


EmbeddingFunc = Callable[[str], Sequence[float]]


@dataclass
class RagChunkDTO:
    text: str
    source_title: Optional[str]
    source_ref: Optional[str]
    score: Optional[float] = None


class RagService:
    """
    High-level RAG API for:
    - ingesting documents
    - retrieving top-k chunks for a query

    Embedding generation is delegated to a caller-provided function that knows
    which provider/model/API key to use (OpenAI, Gemini, etc.).
    """

    def __init__(self):
        # No heavy work in __init__; vector DB connection is lazy.
        pass

    # -------- Ingestion --------

    def chunk_text(
        self,
        text: str,
        max_chars: int = 1000,
        overlap_chars: int = 100,
    ) -> List[str]:
        """
        Simple, language-agnostic text chunking by characters.
        Person A can refine this later (sentence/paragraph aware) without
        breaking the public interface.
        """
        text = (text or "").strip()
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + max_chars, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == length:
                break
            # move start with overlap
            start = end - overlap_chars
            if start < 0:
                start = 0

        return chunks

    def ingest_document(
        self,
        tenant_id: uuid.UUID,
        agent_id: Optional[uuid.UUID],
        title: str,
        source_type: str,
        source_ref: Optional[str],
        full_text: str,
        embedding_func: EmbeddingFunc,
        max_chars: int = 1000,
        overlap_chars: int = 100,
    ) -> uuid.UUID:
        """
        Ingest a single logical document:
        - create RagDocument
        - chunk text
        - generate embeddings for each chunk
        - store RagChunk rows

        Returns the created document_id.
        """
        chunks = self.chunk_text(full_text, max_chars=max_chars, overlap_chars=overlap_chars)
        if not chunks:
            raise ValueError("Cannot ingest empty document text")

        session: Session = get_vector_session()
        try:
            doc = RagDocument(
                tenant_id=tenant_id,
                agent_id=agent_id,
                title=title,
                source_type=source_type,
                source_ref=source_ref,
            )
            session.add(doc)
            session.flush()  # populate doc.id

            for idx, chunk_text in enumerate(chunks):
                try:
                    embedding = list(embedding_func(chunk_text))
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {idx}: {e}", exc_info=True)
                    raise

                chunk = RagChunk(
                    document_id=doc.id,
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    chunk_index=idx,
                    text=chunk_text,
                    embedding=embedding,
                )
                session.add(chunk)

            session.commit()
            logger.info(
                f"Ingested document into RAG store: title='{title}', "
                f"tenant_id={tenant_id}, agent_id={agent_id}, chunks={len(chunks)}"
            )
            return doc.id  # type: ignore[return-value]
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -------- Retrieval --------

    def retrieve(
        self,
        user_text: str,
        tenant_id: uuid.UUID,
        agent_id: Optional[uuid.UUID],
        embedding_func: EmbeddingFunc,
        top_k: int = 5,
    ) -> List[RagChunkDTO]:
        """
        Retrieve top-k chunks for a user query, filtered by tenant/agent.

        NOTE: We rely on pgvector's distance operators via SQLAlchemy's Vector type.
        """
        query_text = (user_text or "").strip()
        if not query_text:
            return []

        try:
            query_embedding = list(embedding_func(query_text))
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}", exc_info=True)
            return []

        session: Session = get_vector_session()
        try:
            # Use pgvector distance; smaller is closer
            # Vector type exposes .l2_distance() helper
            distance_expr = RagChunk.embedding.l2_distance(query_embedding)

            q = (
                session.query(RagChunk, RagDocument, distance_expr.label("distance"))
                .join(RagDocument, RagChunk.document_id == RagDocument.id)
                .filter(RagChunk.tenant_id == tenant_id)
            )

            if agent_id is not None:
                # Either agent-specific or tenant-shared (agent_id IS NULL)
                q = q.filter(
                    (RagChunk.agent_id == agent_id) | (RagChunk.agent_id.is_(None))
                )

            q = q.order_by(asc("distance")).limit(top_k)

            rows: Iterable[Tuple[RagChunk, RagDocument, float]] = q.all()

            results: List[RagChunkDTO] = []
            for chunk, doc, distance in rows:
                score = None
                try:
                    # Convert distance to a similarity-like score in (0, 1]; best effort
                    score = 1.0 / (1.0 + float(distance))
                except Exception:
                    score = None

                results.append(
                    RagChunkDTO(
                        text=chunk.text,
                        source_title=doc.title,
                        source_ref=doc.source_ref,
                        score=score,
                    )
                )

            logger.debug(
                "RAG retrieve: tenant_id=%s agent_id=%s text_len=%d results=%d",
                tenant_id,
                agent_id,
                len(query_text),
                len(results),
            )
            return results
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}", exc_info=True)
            return []
        finally:
            session.close()

    # -------- Formatting for prompts --------

    def format_rag_context(self, chunks: List[RagChunkDTO]) -> str:
        """
        Build a compact, model-friendly context string from retrieved chunks.
        This is what Person B will inject into system prompts.
        """
        if not chunks:
            return ""

        lines: List[str] = []
        lines.append(
            "Below is company knowledge retrieved from the knowledge base. "
            "Use ONLY this information for factual details. If the answer is "
            "not present here, say that you are not sure."
        )
        lines.append("")

        for idx, c in enumerate(chunks, start=1):
            header_parts = []
            if c.source_title:
                header_parts.append(f"title: {c.source_title}")
            if c.source_ref:
                header_parts.append(f"source: {c.source_ref}")
            header = "; ".join(header_parts) if header_parts else "source: unknown"

            lines.append(f"{idx}) [{header}]")
            lines.append(c.text.strip())
            lines.append("")  # blank line between chunks

        return "\n".join(lines)


# Singleton instance to be imported where needed
rag_service = RagService()

