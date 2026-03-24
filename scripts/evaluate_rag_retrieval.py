"""
Lightweight retrieval evaluation harness.

This script verifies that for a set of queries, the retrieved KB context
contains expected phrases (or keywords). It does not call an LLM.

Example input JSON:
[
  {"user_text": "What are your support hours?", "expected_any": ["9am to 5pm", "9:00", "Monday to Friday"]},
  {"user_text": "Do you support US customers only?", "expected_any": ["U.S. customers"]}
]
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Optional, Sequence

from app.core.config import settings
from app.services.openai_service import openai_service
from app.services.rag_service import rag_service


def _parse_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    if not value:
        return None
    return uuid.UUID(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-json", required=True, help="Path to JSON file with queries")
    parser.add_argument("--tenant-id", required=True, help="Tenant UUID")
    parser.add_argument("--agent-id", default=None, help="Optional agent UUID")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k for retrieval")
    args = parser.parse_args()

    tenant_id = uuid.UUID(args.tenant_id)
    agent_id = _parse_uuid(args.agent_id)
    top_k = args.top_k if args.top_k is not None else settings.RAG_TOP_K

    queries_path = Path(args.queries_json).expanduser().resolve()
    items = json.loads(queries_path.read_text(encoding="utf-8"))

    def embedding_func(text: str) -> Sequence[float]:
        return openai_service.embed_text(
            text=text,
            model_name="text-embedding-3-small",
            api_key=None,
        )

    passed = 0
    total = 0

    for item in items:
        total += 1
        user_text = item["user_text"]
        expected_any = item.get("expected_any") or []

        rag_chunks = rag_service.retrieve(
            user_text=user_text,
            tenant_id=tenant_id,
            agent_id=agent_id,
            embedding_func=embedding_func,
            top_k=top_k,
        )

        filtered = []
        for c in rag_chunks:
            score = c.score or 0.0
            if score >= settings.RAG_SCORE_THRESHOLD:
                filtered.append(c)

        context_text = "\n".join([c.text for c in filtered])
        ok = False
        for phrase in expected_any:
            if phrase and phrase.lower() in context_text.lower():
                ok = True
                break

        if ok:
            passed += 1
        print(f"[{passed}/{total}] query={user_text!r} -> {'PASS' if ok else 'FAIL'}")

    print(f"\nRESULT: passed={passed}/{total} ({(passed/total*100) if total else 0:.1f}%)")


if __name__ == "__main__":
    main()

