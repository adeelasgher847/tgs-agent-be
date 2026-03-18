from __future__ import annotations

"""
RAG service: ingestion + retrieval on top of a vector store.

Concrete backend: Pinecone (cloud vector DB) with per-tenant/agent metadata.

Design goals:
- Keep infra concerns (vector DB, chunking, embeddings) separate from voice flow.
- Allow Person B to inject their own embedding function based on Model/provider config.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence
import uuid
import importlib

import pinecone

from app.core.logger import logger
from app.core.config import settings


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
        # Pinecone client + index are created lazily and cached.
        # We avoid importing Pinecone at module import time so that the
        # rest of the app can still start even if the local pinecone
        # package is misconfigured. Any issues are surfaced when RAG
        # is actually used via _get_index().
        self._pc: Optional[object] = None
        self._index = None

    def _get_index(self):
        """
        Lazily initialise and cache the Pinecone client and index handle.
        Host resolution priority:
        1) settings.PINECONE_INDEX_HOST (explicit host from console)
        2) settings.VECTOR_DB_URL (if you've stored the host URL there)
        3) settings.PINECONE_INDEX_NAME via describe_index(...)
        """
        if self._index is not None:
            return self._index

        if not settings.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not configured; cannot use RAG.")

        # We expect the official pinecone SDK (>=6.x) which exposes a
        # Pinecone client class. If it's missing, we fail fast with a
        # clear error instead of an import-time crash.
        PineconeClient = getattr(pinecone, "Pinecone", None)
        if PineconeClient is None:
            raise RuntimeError(
                "The installed 'pinecone' package does not expose a 'Pinecone' client. "
                "Please ensure you have the official SDK installed (e.g. `pip install 'pinecone>=6.0.0'`) "
                "and that any legacy 'pinecone-client' package has been removed."
            )

        pc = PineconeClient(api_key=settings.PINECONE_API_KEY)

        host = settings.PINECONE_INDEX_HOST or settings.VECTOR_DB_URL
        if not host:
            if not settings.PINECONE_INDEX_NAME:
                raise RuntimeError(
                    "Neither PINECONE_INDEX_HOST nor VECTOR_DB_URL nor PINECONE_INDEX_NAME "
                    "is set; cannot resolve Pinecone index host."
                )
            desc = pc.describe_index(settings.PINECONE_INDEX_NAME)
            host = desc.host

        index = pc.Index(host=host)
        self._pc = pc
        self._index = index
        logger.info(f"Connected to Pinecone index host: {host}")
        return self._index

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
        Ingest a single logical document into the vector store:
        - chunk text
        - generate embeddings for each chunk
        - upsert vectors with rich metadata (tenant/agent/source info)

        Returns a synthetic document_id (UUID) for reference.
        """
        chunks = self.chunk_text(full_text, max_chars=max_chars, overlap_chars=overlap_chars)
        if not chunks:
            raise ValueError("Cannot ingest empty document text")

        index = self._get_index()

        # Synthetic document id, just for grouping in metadata / vector IDs
        document_id = uuid.uuid4()

        vectors = []
        for idx, chunk_text in enumerate(chunks):
            try:
                embedding = list(embedding_func(chunk_text))
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {idx}: {e}", exc_info=True)
                raise

            vector_id = f"{tenant_id}:{agent_id or 'all'}:{document_id}:{idx}"
            metadata = {
                "tenant_id": str(tenant_id),
                "agent_id": str(agent_id) if agent_id else None,
                "title": title,
                "source_type": source_type,
                "source_ref": source_ref,
                "chunk_index": idx,
                "text": chunk_text,
            }
            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        # Batch upsert into Pinecone
        index.upsert(vectors=vectors)
        logger.info(
            f"Ingested document into Pinecone RAG index: title='{title}', "
            f"tenant_id={tenant_id}, agent_id={agent_id}, chunks={len(chunks)}"
        )
        return document_id

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
        Retrieve top-k chunks for a user query, filtered by tenant/agent,
        using Pinecone vector similarity search with metadata filters.
        """
        query_text = (user_text or "").strip()
        if not query_text:
            return []

        try:
            query_embedding = list(embedding_func(query_text))
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}", exc_info=True)
            return []

        index = self._get_index()

        # Build metadata filter: tenant is mandatory; agent-specific or shared
        pinecone_filter: dict = {"tenant_id": str(tenant_id)}
        if agent_id is not None:
            pinecone_filter["$or"] = [
                {"agent_id": str(agent_id)},
                {"agent_id": None},
            ]

        try:
            res = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter,
            )
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}", exc_info=True)
            return []

        results: List[RagChunkDTO] = []
        try:
            matches = getattr(res, "matches", []) or []
            for match in matches:
                md = getattr(match, "metadata", None) or {}
                text = md.get("text") or ""
                if not text:
                    continue
                score = None
                try:
                    raw_score = getattr(match, "score", None)
                    score = float(raw_score) if raw_score is not None else None
                except Exception:
                    score = None

                results.append(
                    RagChunkDTO(
                        text=text,
                        source_title=md.get("title"),
                        source_ref=md.get("source_ref"),
                        score=score,
                    )
                )

            logger.debug(
                "RAG retrieve (Pinecone): tenant_id=%s agent_id=%s text_len=%d results=%d",
                tenant_id,
                agent_id,
                len(query_text),
                len(results),
            )
            return results
        except Exception as e:
            logger.error(f"Error processing Pinecone results: {e}", exc_info=True)
            return []

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

