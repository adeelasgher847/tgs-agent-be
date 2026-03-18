"""
Ingest a directory of KB text files into the Pinecone-backed RAG index.

Usage (examples):
  python -m scripts.ingest_kb_text_dir --dir ./kb --tenant-id <uuid> --agent-id <uuid>
  python -m scripts.ingest_kb_text_dir --dir ./kb --agent-id <uuid>           # tenant resolved from agent
  python -m scripts.ingest_kb_text_dir --dir ./kb                                 # uses first Agent in DB

This script is intentionally simple (baseline source type).
You can extend it later with PDF/HTML/DB loaders.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Optional, Callable, Sequence

import html as html_lib
import re
from html.parser import HTMLParser

from app.db.session import SessionLocal
from app.models.agent import Agent
from app.core.config import settings
from app.services.rag_service import rag_service, EmbeddingFunc
from app.services.openai_service import openai_service


class _HTMLTextExtractor(HTMLParser):
    """
    Minimal HTML -> text converter (no external deps).
    Strips script/style and returns visible text blocks.
    """

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts).strip()


def load_html_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    # Normalize whitespace to reduce embedding noise.
    text = html_lib.unescape(parser.get_text())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            parts.append(page_text)
    # Normalize: keep paragraph-ish spacing.
    text = "\n\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _parse_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    if not value:
        return None
    return uuid.UUID(value)


def _resolve_agent_and_tenant(
    db,
    tenant_id: Optional[uuid.UUID],
    agent_id: Optional[uuid.UUID],
) -> tuple[uuid.UUID, Optional[uuid.UUID], str]:
    """
    Returns: (tenant_id, agent_id, resolved_by)
    """
    if agent_id is not None:
        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.is_deleted == False).first()  # noqa: E712
        if not agent:
            raise ValueError(f"Agent not found for agent_id={agent_id}")
        return agent.tenant_id, agent.id, "agent_id"

    if tenant_id is not None:
        # Caller supplied tenant_id but not agent_id; ingest as shared KB (agent_id=None).
        return tenant_id, None, "tenant_id"

    # Nothing provided: use first non-deleted agent as default anchor.
    agent = db.query(Agent).filter(Agent.is_deleted == False).first()  # noqa: E712
    if not agent:
        raise RuntimeError("No agent found in DB; create an agent first or pass --tenant-id/--agent-id.")
    return agent.tenant_id, agent.id, "first_agent"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing .txt/.md files")
    parser.add_argument("--tenant-id", default=None, help="Tenant UUID")
    parser.add_argument("--agent-id", default=None, help="Agent UUID (optional). If omitted, tenant-only/shared ingest.")
    parser.add_argument("--version", default=None, help="Document version label (for future extensions)")
    parser.add_argument("--source-type", default="kb_dir", help="Metadata source_type")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model name")
    parser.add_argument("--chunk-max-chars", type=int, default=1000, help="Chunk max chars")
    parser.add_argument("--chunk-overlap-chars", type=int, default=100, help="Chunk overlap chars")
    args = parser.parse_args()

    kb_dir = Path(args.dir).expanduser().resolve()
    if not kb_dir.exists() or not kb_dir.is_dir():
        raise ValueError(f"--dir must be an existing directory: {kb_dir}")

    tenant_id = _parse_uuid(args.tenant_id)
    agent_id = _parse_uuid(args.agent_id)

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured; cannot generate embeddings.")
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not configured; cannot reach Pinecone.")

    db = SessionLocal()
    try:
        resolved_tenant_id, resolved_agent_id, resolved_by = _resolve_agent_and_tenant(
            db=db,
            tenant_id=tenant_id,
            agent_id=agent_id,
        )

        def embedding_func(text: str) -> Sequence[float]:
            # rag_service.ingest_document expects an embedding vector.
            return openai_service.embed_text(
                text=text,
                model_name=args.embedding_model,
                api_key=None,  # use settings.OPENAI_API_KEY
            )

        # Ingest supported files under the directory.
        exts = {".txt", ".md", ".html", ".htm", ".pdf", ".faq"}
        files = sorted([p for p in kb_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        if not files:
            raise RuntimeError(f"No supported KB files (.txt/.md/.html/.pdf/.faq) found under {kb_dir}")

        print(f"[INFO] Ingesting {len(files)} file(s) from {kb_dir}")
        print(f"[INFO] Resolved tenant_id={resolved_tenant_id} agent_id={resolved_agent_id} (by {resolved_by})")

        for idx, path in enumerate(files, start=1):
            rel = path.relative_to(kb_dir)
            title = path.stem
            source_ref = str(rel).replace(os.sep, "/")

            ext = path.suffix.lower()
            if ext in {".txt", ".md", ".faq"}:
                full_text = path.read_text(encoding="utf-8", errors="ignore").strip()
            elif ext in {".html", ".htm"}:
                full_text = load_html_text(path)
            elif ext == ".pdf":
                full_text = load_pdf_text(path)
            else:
                raise RuntimeError(f"Unsupported extension encountered: {ext}")
            if not full_text:
                print(f"[SKIP] ({idx}/{len(files)}) Empty file: {source_ref}")
                continue

            print(f"[STEP] ({idx}/{len(files)}) Ingesting {source_ref} ...")
            rag_service.ingest_document(
                tenant_id=resolved_tenant_id,
                agent_id=resolved_agent_id,
                title=title,
                source_type=args.source_type,
                source_ref=source_ref,
                full_text=full_text,
                embedding_func=embedding_func,
                version=args.version or "v1",
                db_session=db,
                replace_existing=True,
                max_chars=args.chunk_max_chars,
                overlap_chars=args.chunk_overlap_chars,
            )

        print("[OK] Ingestion complete.")
    finally:
        db.close()


if __name__ == "__main__":
    main()

