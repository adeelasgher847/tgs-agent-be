"""
Temporary test script to validate RAG wiring end-to-end.

Run from project root (with venv active):
    venv\\Scripts\\python -m scripts.test_rag_flow

Safe to delete after verification.
"""

from __future__ import annotations

from app.db.session import SessionLocal
from app.models.agent import Agent
from app.core.config import settings
from app.services.rag_service import rag_service
from app.services.openai_service import openai_service
from app.voice.rag_context import build_rag_context_block


def main() -> None:
    db = SessionLocal()
    try:
        agent = (
            db.query(Agent)
            .filter(Agent.is_deleted == False)  # noqa: E712
            .first()
        )
        if not agent:
            print("[ERROR] No agent found in DB – create an agent first.")
            return

        tenant_id = agent.tenant_id
        agent_id = agent.id

        print(f"[OK] Using agent={agent.name!r} id={agent_id} tenant_id={tenant_id}")

        if not settings.OPENAI_API_KEY:
            print("[ERROR] OPENAI_API_KEY is not set in .env – RAG embeddings cannot run.")
            return

        if not settings.PINECONE_API_KEY:
            print("[ERROR] PINECONE_API_KEY is not set – RAG cannot reach Pinecone.")
            return

        # --- 1) Ingest a small test document into RAG for this tenant/agent ---
        def embedding_func(text: str):
            # Uses settings.OPENAI_API_KEY under the hood
            return openai_service.embed_text(
                text=text,
                model_name="text-embedding-3-small",
                api_key=None,
            )

        doc_text = (
            "Our company support hours are 9am to 5pm, Monday to Friday. "
            "We only support U.S. customers by phone. "
            "Email support is available 24/7 for all regions."
        )

        print("[STEP] Ingesting test document into Pinecone RAG index...")
        document_id = rag_service.ingest_document(
            tenant_id=tenant_id,
            agent_id=agent_id,
            title="Test Support FAQ (RAG verification)",
            source_type="rag_test",
            source_ref="test-support-faq",
            full_text=doc_text,
            embedding_func=embedding_func,
        )
        print(f"[OK] Ingested test document_id={document_id}")

        # --- 2) Build RAG context block for a representative user query ---
        query_text = "What are your support hours?"
        print(f"[STEP] Building RAG context block for query: {query_text!r}")

        ctx_block = build_rag_context_block(
            user_text=query_text,
            tenant_id=tenant_id,
            agent_id=agent_id,
        )

        print("\n----- RAG CONTEXT BLOCK START -----")
        print(ctx_block)
        print("----- RAG CONTEXT BLOCK END -----\n")

        # Quick human-readable verdict
        if "9am to 5pm" in ctx_block:
            print("[RESULT] RAG test PASSED: support hours text is present in context block.")
        else:
            print("[RESULT] RAG test INCONCLUSIVE: context block did not clearly include the test support hours.")

    finally:
        db.close()


if __name__ == "__main__":
    main()

