"""
Automated provider-strategy test for RAG embeddings.

What this validates:
1) Direct OpenAI embedding call
2) Direct Gemini embedding call
3) Unified embedding path (embed_text_for_rag) with Gemini-primary behavior

Run from project root (venv active):
    ./venv/bin/python -m scripts.test_embedding_provider_strategy
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from app.services.embedding_service import embed_text_for_rag
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run_check(name: str, fn) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name=name, ok=True, detail=detail)
    except Exception as exc:
        return CheckResult(name=name, ok=False, detail=str(exc))


def _check_openai() -> str:
    vec = openai_service.embed_text("provider strategy test", model_name="text-embedding-3-small")
    return f"vector_len={len(vec)}"


def _check_gemini() -> str:
    vec = gemini_service.embed_text(
        "provider strategy test",
        model_name="gemini-embedding-001",
        output_dimensionality=1536,
    )
    return f"vector_len={len(vec)}"


def _check_unified_path() -> str:
    vec = embed_text_for_rag("provider strategy test")
    return f"vector_len={len(vec)}"


def main() -> None:
    print("\n=== Embedding Provider Strategy Test ===\n")

    results = [
        _run_check("OpenAI direct embedding", _check_openai),
        _run_check("Gemini direct embedding", _check_gemini),
        _run_check("Unified embedding path", _check_unified_path),
    ]

    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    openai_ok = results[0].ok
    gemini_ok = results[1].ok
    unified_ok = results[2].ok

    print("\n=== Verdict ===")
    if gemini_ok and unified_ok:
        if openai_ok:
            print("PASS: Both providers work; unified path is healthy.")
        else:
            print("PASS: OpenAI failed, but Gemini + unified path work (Gemini-first strategy is effective).")
        sys.exit(0)

    print("FAIL: Gemini or unified path failed. Investigate before proceeding.")
    sys.exit(1)


if __name__ == "__main__":
    main()
