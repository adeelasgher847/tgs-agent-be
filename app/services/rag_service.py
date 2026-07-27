"""
RAG service: ingestion + retrieval backed by pgvector on Postgres.

Replaces the previous Pinecone backend. Embeddings use OpenAI
text-embedding-ada-002 (1536 dims) stored in the kb_chunks table.

Public interface is backward-compatible so voice/rag_context.py and
agent_service.py continue to work without changes.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.core.config import settings


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
    Thin facade over the pgvector knowledge-base tables.

    retrieve() and ingest_document() accept an optional db_session.
    When none is provided they create one from SessionLocal — pragmatic
    exception to the session-injection rule for voice-path callers that
    don't have a session in scope.
    """

    # ── Session helper ────────────────────────────────────────────────────────

    def _session(self, provided: Optional[Session]) -> tuple[Session, bool]:
        """Return (session, should_close). If we create it, caller must close it."""
        if provided is not None:
            return provided, False
        from app.db.session import SessionLocal  # pragmatic: voice path has no session

        return SessionLocal(), True

    # ── Chunking (kept for backward-compat callers) ───────────────────────────

    def chunk_text(
        self,
        text: str,
        max_chars: int = 1000,
        overlap_chars: int = 100,
    ) -> List[str]:
        """Character-based chunker (legacy). New code uses kb_ingestion_service.chunk_text."""
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
            start = end - overlap_chars
            if start < 0:
                start = 0
        return chunks

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        user_text: str,
        tenant_id: Optional[uuid.UUID],
        agent_id: Optional[uuid.UUID],
        embedding_func: EmbeddingFunc,
        top_k: int = 5,
        trace: Optional[dict] = None,
        db_session: Optional[Session] = None,
    ) -> List[RagChunkDTO]:
        query_text = (user_text or "").strip()
        if not query_text or not tenant_id:
            return []

        try:
            query_embedding = list(embedding_func(query_text))
        except Exception as e:
            logger.error("Embedding failed for RAG query: %s", e, exc_info=True)
            if trace is not None:
                trace["retrieve_error"] = "embedding_failed"
                trace["retrieve_error_msg"] = str(e)[:200]
            return []

        # Format vector for pgvector: "[0.1,0.2,...]"
        vec_str = "[" + ",".join(str(f) for f in query_embedding) + "]"

        db, should_close = self._session(db_session)
        try:
            stmt = text(
                """
                SELECT
                    c.id::text            AS id,
                    c.content,
                    c.metadata            AS chunk_metadata,
                    kb.name               AS kb_name,
                    1 - (c.embedding::vector <=> CAST(:vec AS vector)) AS score
                FROM kbchunk c
                JOIN knowledgebase kb ON c.kb_id = kb.id
                WHERE kb.workspace_id = :workspace_id
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding::vector <=> CAST(:vec AS vector)
                LIMIT :top_k
                """
            )
            rows = db.execute(
                stmt,
                {
                    "vec": vec_str,
                    "workspace_id": str(tenant_id),
                    "top_k": top_k,
                },
            ).fetchall()
        except Exception as e:
            logger.error("pgvector retrieval failed: %s", e, exc_info=True)
            if trace is not None:
                trace["retrieve_error"] = "pgvector_query_failed"
                trace["retrieve_error_msg"] = str(e)[:200]
            return []
        finally:
            if should_close:
                db.close()

        results: List[RagChunkDTO] = []
        for row in rows:
            meta = row.chunk_metadata or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            results.append(
                RagChunkDTO(
                    text=row.content,
                    source_title=row.kb_name,
                    source_ref=meta.get("filename") or meta.get("source"),
                    score=row.score,
                    vector_id=row.id,
                    chunk_index=meta.get("chunk_index"),
                )
            )

        if trace is not None:
            trace["pinecone_results_count"] = len(results)
        logger.debug(
            "RAG retrieve (pgvector): workspace=%s results=%d", tenant_id, len(results)
        )
        return results

    # ── Ingestion (legacy path — used by agent_service auto-ingest) ───────────

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
        Ingest a text document into the KB for the given workspace.

        Gets or creates a default 'Auto-Ingest' knowledge base for the tenant,
        then inserts chunks with embeddings into kb_chunks.
        Returns a synthetic document_id (UUID5 deterministic key).
        """
        chunks = self.chunk_text(full_text, max_chars=max_chars, overlap_chars=overlap_chars)
        if not chunks:
            raise ValueError("Cannot ingest empty document text")

        db, should_close = self._session(db_session)
        try:
            kb = self._get_or_create_auto_kb(db, tenant_id)

            # Deterministic document_id for safe re-ingest
            agent_key = str(agent_id) if agent_id else "all"
            document_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"kb:{tenant_id}:{agent_key}:{source_type}:{source_ref}",
            )

            if replace_existing:
                self._delete_chunks_by_meta_source_ref(db, kb.id, str(document_id))

            from app.models.knowledge_base_chunk import KbChunk

            chunk_rows = []
            for idx, chunk_content in enumerate(chunks):
                try:
                    embedding = list(embedding_func(chunk_content))
                except Exception as e:
                    logger.error("Embedding failed for chunk %d: %s", idx, e, exc_info=True)
                    raise

                chunk_rows.append(
                    KbChunk(
                        id=uuid.uuid4(),
                        kb_id=kb.id,
                        file_id=None,
                        content=chunk_content,
                        embedding=json.dumps(embedding),
                        chunk_metadata={
                            "chunk_index": idx,
                            "document_id": str(document_id),
                            "source_type": source_type,
                            "source_ref": source_ref,
                            "title": title,
                            "version": version,
                        },
                    )
                )

            db.add_all(chunk_rows)
            db.commit()
            logger.info(
                "ingest_document: workspace=%s title=%r chunks=%d",
                tenant_id, title, len(chunks),
            )
            return document_id

        except Exception:
            db.rollback()
            raise
        finally:
            if should_close:
                db.close()

    def _get_or_create_auto_kb(self, db: Session, workspace_id: uuid.UUID):
        from app.models.knowledge_base_document import KnowledgeBase

        kb = (
            db.query(KnowledgeBase)
            .filter(
                KnowledgeBase.workspace_id == workspace_id,
                KnowledgeBase.name == "Auto-Ingest",
            )
            .first()
        )
        if kb is None:
            kb = KnowledgeBase(
                id=uuid.uuid4(),
                workspace_id=workspace_id,
                name="Auto-Ingest",
                description="System-managed knowledge base for auto-ingested content",
            )
            db.add(kb)
            db.flush()
        return kb

    def _delete_chunks_by_meta_source_ref(
        self, db: Session, kb_id: uuid.UUID, document_id: str
    ) -> None:
        from app.models.knowledge_base_chunk import KbChunk

        # SQLite-safe delete using Python-side filter (metadata is JSON column)
        rows = (
            db.query(KbChunk)
            .filter(KbChunk.kb_id == kb_id, KbChunk.file_id.is_(None))
            .all()
        )
        to_delete = [
            r for r in rows
            if isinstance(r.chunk_metadata, dict) and r.chunk_metadata.get("document_id") == document_id
        ]
        for r in to_delete:
            db.delete(r)
        if to_delete:
            db.flush()

    # ── Vector deletion (legacy delete endpoint) ──────────────────────────────

    def delete_vectors(
        self,
        vector_ids: List[str],
        db_session: Optional[Session] = None,
    ) -> None:
        if not vector_ids:
            return
        from app.models.knowledge_base_chunk import KbChunk

        db, should_close = self._session(db_session)
        try:
            for vid in vector_ids:
                try:
                    row = db.get(KbChunk, uuid.UUID(vid))
                    if row:
                        db.delete(row)
                except (ValueError, Exception):
                    pass
            db.commit()
        finally:
            if should_close:
                db.close()

    # ── Context formatting ────────────────────────────────────────────────────

    def format_rag_context(
        self, chunks: List[RagChunkDTO], max_chars: Optional[int] = None
    ) -> str:
        if not chunks:
            return ""

        lines: List[str] = [
            "Below is company knowledge retrieved from the knowledge base. "
            "Use ONLY this information for factual details. If the answer is "
            "not present here, say that you are not sure.",
            "",
        ]
        for idx, c in enumerate(chunks, start=1):
            header_parts = []
            if c.source_title:
                header_parts.append(f"title: {c.source_title}")
            if c.source_ref:
                header_parts.append(f"source: {c.source_ref}")
            header = "; ".join(header_parts) if header_parts else "source: unknown"
            lines.append(f"[{idx}] [{header}]")
            lines.append(c.text.strip())
            lines.append("")

        context = "\n".join(lines)
        if max_chars and max_chars > 0 and len(context) > max_chars:
            context = context[: max_chars - 64].rstrip() + "\n...[TRUNCATED]"
        return context


rag_service = RagService()
