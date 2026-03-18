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
import re

import pinecone

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.core.config import settings

from app.models.knowledge_base_document import KnowledgeBaseDocument
from app.models.knowledge_base_chunk import KnowledgeBaseChunk


EmbeddingFunc = Callable[[str], Sequence[float]]


@dataclass
class RagChunkDTO:
    text: str
    source_title: Optional[str]
    source_ref: Optional[str]
    score: Optional[float] = None
    vector_id: Optional[str] = None
    chunk_index: Optional[int] = None


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

    def delete_vectors(self, vector_ids: List[str]) -> None:
        """
        Delete Pinecone vectors by explicit IDs.

        Requires the caller to provide a complete vector ID list.
        In this codebase, the KB chunk inventory table provides those IDs.
        """
        if not vector_ids:
            return
        index = self._get_index()
        index.delete(ids=vector_ids)

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
        version: str = "v1",
        db_session: Optional[Session] = None,
        replace_existing: bool = True,
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

        # Deterministic document id for stable vector IDs and safe re-ingest.
        agent_key = str(agent_id) if agent_id else "all"
        document_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"kb:{tenant_id}:{agent_key}:{source_type}:{source_ref}",
        )

        if db_session is not None:
            # Upsert document metadata in DB (chunk text still lives in Pinecone metadata).
            doc = (
                db_session.query(KnowledgeBaseDocument)
                .filter(KnowledgeBaseDocument.id == document_id)
                .first()
            )
            if not doc:
                doc = KnowledgeBaseDocument(
                    id=document_id,
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    title=title,
                    source_type=source_type,
                    source_ref=source_ref or "",
                    version=version,
                    is_active=True,
                )
                db_session.add(doc)
            else:
                doc.tenant_id = tenant_id
                doc.agent_id = agent_id
                doc.title = title
                doc.source_type = source_type
                doc.source_ref = source_ref or ""
                doc.version = version
                doc.is_active = True
            db_session.commit()

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
                "document_id": str(document_id),
                "version": version,
            }
            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        # If re-ingesting and we have chunk inventory in the DB, delete stale vectors first.
        # This prevents old chunks from lingering in Pinecone.
        if db_session is not None and replace_existing:
            existing_chunks = (
                db_session.query(KnowledgeBaseChunk)
                .filter(KnowledgeBaseChunk.document_id == document_id)
                .all()
            )
            stale_vector_ids = [c.vector_id for c in existing_chunks if c.vector_id]
            if stale_vector_ids:
                try:
                    index.delete(ids=stale_vector_ids)
                except Exception as e:
                    logger.warning("Pinecone deletion failed during re-ingest: %s", e, exc_info=True)

            # Clear chunk inventory rows; we will insert the new ones after upsert.
            try:
                db_session.query(KnowledgeBaseChunk).filter(
                    KnowledgeBaseChunk.document_id == document_id
                ).delete(synchronize_session=False)
                db_session.commit()
            except Exception as e:
                logger.warning("DB chunk inventory deletion failed: %s", e, exc_info=True)

        # Batch upsert into Pinecone
        index.upsert(vectors=vectors)
        logger.info(
            f"Ingested document into Pinecone RAG index: title='{title}', "
            f"tenant_id={tenant_id}, agent_id={agent_id}, chunks={len(chunks)}"
        )

        if db_session is not None:
            # Insert fresh chunk inventory rows so we can safely delete stale vectors next time.
            try:
                chunk_rows = []
                for v in vectors:
                    md = v.get("metadata") or {}
                    chunk_rows.append(
                        KnowledgeBaseChunk(
                            document_id=document_id,
                            chunk_index=int(md.get("chunk_index", 0)),
                            vector_id=v.get("id"),
                            text_preview=((md.get("text") or "")[:500] or None),
                        )
                    )
                db_session.add_all(chunk_rows)
                db_session.commit()
            except Exception as e:
                logger.warning("Failed to insert KB chunk inventory rows: %s", e, exc_info=True)

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
                        vector_id=getattr(match, "id", None),
                        chunk_index=md.get("chunk_index"),
                    )
                )

            # Optional reranking:
            # Pinecone similarity provides the primary ranking; we apply a small lexical
            # overlap adjustment to improve exact-match cases (FAQs, policies, prices).
            if settings.RAG_ENABLE_RERANK and results:
                vector_weight = float(getattr(settings, "RAG_RERANK_VECTOR_WEIGHT", 0.8))
                vector_weight = max(0.0, min(1.0, vector_weight))
                query_tokens = set(
                    t for t in re.findall(r"[a-z0-9]+", query_text.lower()) if len(t) >= 3
                )

                def lexical_overlap_score(chunk_text: str) -> float:
                    if not query_tokens:
                        return 0.0
                    chunk_tokens = set(
                        t for t in re.findall(r"[a-z0-9]+", (chunk_text or "").lower()) if len(t) >= 3
                    )
                    if not chunk_tokens:
                        return 0.0
                    return len(query_tokens.intersection(chunk_tokens)) / max(1, len(query_tokens))

                def combined_rank(c: RagChunkDTO) -> float:
                    vec_score = float(c.score or 0.0)
                    lex_score = lexical_overlap_score(c.text)
                    # Rank only; keep c.score unchanged so voice-layer gating stays stable.
                    return (vector_weight * vec_score) + ((1.0 - vector_weight) * lex_score)

                results = sorted(results, key=combined_rank, reverse=True)

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

    def format_rag_context(self, chunks: List[RagChunkDTO], max_chars: Optional[int] = None) -> str:
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

            # Use square-bracket indices so the model can cite as [1], [2], ...
            lines.append(f"[{idx}] [{header}]")
            lines.append(c.text.strip())
            lines.append("")  # blank line between chunks

        context = "\n".join(lines)

        # Safety valve: cap context size so we don't blow up prompt length
        # (which can cause latency spikes and lower answer quality).
        if max_chars is not None and max_chars > 0 and len(context) > max_chars:
            context = context[: max_chars - 64].rstrip() + "\n...[TRUNCATED]"

        return context


# Singleton instance to be imported where needed
rag_service = RagService()

