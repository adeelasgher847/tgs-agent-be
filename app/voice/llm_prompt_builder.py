"""
Pure, unit-testable helpers for building LLM prompt inputs for the voice path.

No I/O, no DB, no side effects. Imported by bidirectional_stream and Vertex service.
"""
from __future__ import annotations


def prune_history_to_turns(
    history: list[tuple[str, str]],
    max_turns: int,
) -> list[tuple[str, str]]:
    """
    Prune history to at most max_turns conversation turns.

    1 turn = 1 user message + 1 agent reply (a pair).  Oldest pairs are dropped first.
    Unpaired trailing messages are kept (e.g. a lone user message at the start).
    """
    if not history or max_turns <= 0:
        return []
    max_messages = max_turns * 2
    if len(history) <= max_messages:
        return list(history)
    return list(history[-max_messages:])


def build_kb_context_for_vertex(
    inbound_kb: str,
    business_kb: str,
    *,
    empty_guard: str = "",
) -> str:
    """
    Combine inbound + business KB blocks for Vertex user-turn context.
    Falls back to empty_guard when no KB content is loaded.
    """
    parts = [p.strip() for p in (inbound_kb, business_kb) if p and p.strip()]
    if parts:
        return "\n\n".join(parts)
    return (empty_guard or "").strip()


def build_history_text(
    history: list[tuple[str, str]],
    max_turns: int = 20,
) -> str:
    """
    Render pruned history as the plain-text block used by non-Vertex LLM paths.

    Output: "Client: ...\nAgent: ...\n..."
    """
    pruned = prune_history_to_turns(history, max_turns)
    return "\n".join(f"{role.capitalize()}: {content}" for role, content in pruned)


def build_vertex_contents(
    conversation_history: list[tuple[str, str]] | None,
    caller_transcript: str,
    kb_context: str | None = None,
    max_turns: int = 20,
) -> list:
    """
    Build a Vertex AI Content list for generate_content().

    Maps (role, text) tuples:  "client" → "user",  "agent" → "model".
    Prunes to max_turns pairs. kb_context is appended to the current user turn.

    Returns a list of vertexai.generative_models.Content objects.
    """
    try:
        from vertexai.generative_models import Content, Part
    except ImportError as exc:
        raise ImportError(
            "google-cloud-aiplatform is required for build_vertex_contents. "
            "Add it to requirements.txt."
        ) from exc

    pruned = prune_history_to_turns(list(conversation_history or []), max_turns)
    contents = []
    for role, text in pruned:
        if not text:
            continue
        vertex_role = "user" if role == "client" else "model"
        contents.append(Content(role=vertex_role, parts=[Part.from_text(text)]))

    user_text = caller_transcript or ""
    if kb_context:
        user_text = f"{user_text}\n\n[CONTEXT]\n{kb_context}"

    if user_text:
        contents.append(Content(role="user", parts=[Part.from_text(user_text)]))

    return contents
