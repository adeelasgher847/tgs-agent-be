"""Shared phone-call STYLE & TONE lines for both Twilio and LiveKit prompts."""

HUMAN_SPEECH_STYLE_LINES = """\
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- HUMAN: Sound like a person on a phone, not a chatbot or form letter. Use contractions (I'm, I'll, that's).
- NATURAL: At most one fitting interjection per reply: "hmm", "oh", "alright", "hang on", "one moment". Never stack them. Do not invent mid-sentence "uh"/"umm" that would click in audio.
- WARM: Acknowledge first, then help. Match the caller's energy — never chipper if they are upset or sad.
- CONCISE: One idea per sentence. Ask one question at a time.
- NO ROBOT TALK: Never say "As an AI", "How may I assist you", "Please be advised", or "I apologize for any inconvenience". Say "Sorry about that" or "How can I help" instead. Prefer "Hey," "Hi," or "Hello."
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML")."""


def build_style_and_tone_section(
    *,
    output_plain_text_rule: str = "",
    no_bracket_tags_line: str = "",
    greeting_instruction_block: str = "",
) -> str:
    """Assemble the STYLE & TONE prompt block used by both call transports."""
    parts = ["# STYLE & TONE", HUMAN_SPEECH_STYLE_LINES]
    if output_plain_text_rule:
        parts.append(output_plain_text_rule)
    if no_bracket_tags_line:
        parts.append(no_bracket_tags_line)
    if greeting_instruction_block:
        parts.append(greeting_instruction_block.lstrip("\n"))
    return "\n".join(p for p in parts if p).rstrip()
