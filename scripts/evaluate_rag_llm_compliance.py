"""
LLM-level RAG compliance evaluation (grounding / refusal + optional citations).

This script is designed for production regression checks and manual QA.
It calls an LLM (OpenAI via app.services.openai_service) and compares behavior
with RAG context ON vs OFF.

Input JSON examples:
[
  {
    "question": "What are your support hours?",
    "expected_any": ["9am to 5pm", "Monday to Friday"],
    "should_refuse": false,
    "require_citations": true
  },
  {
    "question": "Do you offer support in Germany?",
    "expected_any": ["not available", "don't know"],
    "should_refuse": true,
    "require_citations": false
  }
]
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Optional, Sequence, Any

from app.core.config import settings
from app.services.openai_service import openai_service
from app.voice.rag_context import build_rag_context_block_with_trace


REFUSAL_PATTERNS = [
    r"information is not available",
    r"not available",
    r"i do not know",
    r"i don't know",
    r"not sure",
    r"cannot provide",
]


EMPTY_KB_BLOCK = """
# KNOWLEDGE BASE CONTEXT
No relevant knowledge base entries were found for this query.
If the user asks for specific factual, pricing, or policy details you do not see in the conversation history,
respond that this information is not available instead of guessing or inventing details.
""".strip()


SYSTEM_PREFIX = """
You are a phone assistant. Use knowledge base context when available.
If the answer is not clearly supported by the context, respond that the information is not available.
Do not guess.
""".strip()


def _parse_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    if not value:
        return None
    return uuid.UUID(value)


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    t = (text or "").lower()
    for p in phrases:
        if p and p.lower() in t:
            return True
    return False


def _looks_like_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


def _has_citations(text: str) -> bool:
    # Match [1], [2], ... style citations.
    return bool(re.search(r"\[\d+\]", text or ""))


def _build_system_prompt(context_block: str) -> str:
    return f"{SYSTEM_PREFIX}\n\n# CONTEXT\n{context_block}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-json", required=True, help="Path to JSON list of test items")
    parser.add_argument("--tenant-id", required=True, help="Tenant UUID")
    parser.add_argument("--agent-id", default=None, help="Optional agent UUID (defaults to tenant scope)")
    parser.add_argument("--model-name", default="gpt-3.5-turbo", help="OpenAI model name")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on number of test items")
    args = parser.parse_args()

    tenant_id = uuid.UUID(args.tenant_id)
    agent_id = _parse_uuid(args.agent_id)

    queries_path = Path(args.questions_json).expanduser().resolve()
    items: list[dict[str, Any]] = json.loads(queries_path.read_text(encoding="utf-8"))
    if args.max_items is not None:
        items = items[: args.max_items]

    passed = 0
    total = 0

    def embed_passthrough(_text: str) -> Sequence[float]:
        # Not used directly: context builder handles embedding via voice rag_context.
        raise RuntimeError("embed_passthrough should not be called")

    for item in items:
        total += 1
        q = (item.get("question") or "").strip()
        expected_any = item.get("expected_any") or []
        should_refuse = bool(item.get("should_refuse", False))
        require_citations = bool(item.get("require_citations", False))

        # RAG ON
        context_on, trace = build_rag_context_block_with_trace(
            user_text=q,
            tenant_id=tenant_id,
            agent_id=agent_id,
        )
        system_on = _build_system_prompt(context_on)
        resp_on = openai_service.generate_text(
            prompt=q,
            system_prompt=system_on,
            model_name=args.model_name,
            temperature=0.0,
            max_tokens=250,
        )
        out_on = resp_on.get("content") if isinstance(resp_on, dict) else str(resp_on)

        # RAG OFF
        system_off = _build_system_prompt(EMPTY_KB_BLOCK)
        resp_off = openai_service.generate_text(
            prompt=q,
            system_prompt=system_off,
            model_name=args.model_name,
            temperature=0.0,
            max_tokens=250,
        )
        out_off = resp_off.get("content") if isinstance(resp_off, dict) else str(resp_off)

        # Check compliance (use RAG ON output as the primary signal)
        ok_on = True
        if should_refuse:
            ok_on = _looks_like_refusal(out_on)
        else:
            ok_on = _contains_any(out_on, expected_any) if expected_any else True
            if require_citations:
                ok_on = ok_on and _has_citations(out_on)

        # If you want: you can also compare against RAG-OFF, but we keep
        # it informational to avoid overfitting heuristics.
        if ok_on:
            passed += 1

        print(f"[{passed}/{total}] should_refuse={should_refuse} citations_req={require_citations} -> {'PASS' if ok_on else 'FAIL'}")
        if not ok_on:
            print(f"  Q: {q}")
            print(f"  OUT_RAG_ON: {out_on}")
            print(f"  OUT_RAG_OFF: {out_off}")

    print(f"\nRESULT: passed={passed}/{total} ({(passed/total*100) if total else 0:.1f}%)")


if __name__ == "__main__":
    main()

