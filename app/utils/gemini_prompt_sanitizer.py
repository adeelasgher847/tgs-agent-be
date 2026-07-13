"""Deterministic Gemini prompt sanitizer — no LLM calls, no external I/O."""
import re

_ROLE_PREFIX = "You are an AI voice agent for phone calls.\n\n"
_ROLE_STARTERS = ("you are", "you're")
# Matches a fenced code block wrapping the entire prompt
_FENCE_RE = re.compile(r"^```[^\n]*\n(.*)\n```$", re.DOTALL)
# Control characters except \n (0x0a) and \t (0x09)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def sanitize_prompt_for_gemini(prompt_text: str) -> str:
    """Deterministic sanitization for Gemini — no external API calls.

    Transformations (in order):
    1. Return "" for empty/whitespace-only input.
    2. Strip wrapping markdown code fences if the whole text is fenced.
    3. Remove null bytes and non-printable control chars (keep \\n and \\t).
    4. Normalize Windows/classic-Mac line endings to \\n.
    5. Strip trailing whitespace from every line.
    6. Collapse 3+ consecutive blank lines to two.
    7. Strip leading/trailing whitespace of the whole text.
    8. Prepend role prefix if prompt does not already open with a role statement.
    9. Ensure exactly one trailing newline.
    """
    if not prompt_text or not prompt_text.strip():
        return ""

    cleaned = prompt_text

    # 2 — strip code fences wrapping the entire content
    fence_match = _FENCE_RE.match(cleaned.strip())
    if fence_match:
        cleaned = fence_match.group(1)

    # 3 — remove control characters (keep \n and \t)
    cleaned = _CTRL_RE.sub("", cleaned)

    # 4 — normalize line endings
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # 5 — strip trailing whitespace per line
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

    # 6 — collapse 3+ consecutive newlines to two
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 7 — strip outer whitespace
    cleaned = cleaned.strip()

    # 8 — prepend role prefix if missing
    if not cleaned.lower().startswith(_ROLE_STARTERS):
        cleaned = _ROLE_PREFIX + cleaned

    # 9 — single trailing newline
    cleaned = cleaned.rstrip("\n") + "\n"

    return cleaned
