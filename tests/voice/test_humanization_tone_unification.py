"""
Phase 4A: centralizing tone_adapter() into the shared HumanizationEngine so
Twilio and LiveKit/browser use exactly one text-level tone-adaptation path.

Covers:
  1. Existing Twilio tone behavior remains correct
  2. Tone adaptation is not applied twice
  3. LiveKit can use the same shared logic where safe
  4. Normal text remains semantically unchanged
  5. Numbers remain unchanged
  6. Names remain unchanged
  7. Business/booking information remains unchanged
  8. Empty/short text remains safe
  9. Humanization disabled preserves existing (pass-through) behavior
  10. Humanization failure returns original text
  11. Tone adaptation failure specifically returns original text
"""
from __future__ import annotations

import ast
import asyncio
import inspect

import app.voice.humanization_engine as humanization_engine
from app.voice.humanization_engine import analyze_response
from app.voice.tts_pipeline import TtsPipeline
from app.voice.turn_signals import UserMood


# ---------------------------------------------------------------------------
# 1. Existing Twilio tone behavior remains correct (measured, not assumed)
# ---------------------------------------------------------------------------


def test_leading_chipper_opener_is_stripped_when_not_ssml():
    """
    The one substitution tone_adapter's regexes actually perform in practice:
    a leading "Awesome!"/"Great!"/"Perfect!"-style opener is stripped when
    use_ssml=False. This must survive the move into the shared engine.
    """
    d = analyze_response(
        "Awesome! Let me check that for you.",
        user_text="hi there",
        use_ssml=False,
    )
    assert d.text == "Let me check that for you."


def test_tone_adapter_is_noop_when_use_ssml_true():
    """
    tone_adapter's own calling convention (unchanged by this phase): every
    substitution branch is gated on `not use_ssml`. Both Twilio and the
    browser handler hardcode use_ssml=True in production today, so this is
    the actual production behavior — verify the centralized call site
    preserves it exactly (text passes through, only stripped).
    """
    d = analyze_response(
        "Awesome! Let me check that for you.",
        user_text="hi there",
        use_ssml=True,
    )
    assert d.text == "Awesome! Let me check that for you."


# ---------------------------------------------------------------------------
# 2. Tone adaptation is not applied twice
# ---------------------------------------------------------------------------


def test_bidirectional_stream_no_longer_calls_tone_adapter_directly():
    """
    Static regression guard: bidirectional_stream.py must not import or call
    tone_adapter() itself anymore — it now runs exactly once, centrally,
    inside TtsPipeline._process_chunk. A reintroduced direct call would
    double-process (and reintroduce transport-specific duplication).
    """
    import app.routers.bidirectional_stream as mod

    source = inspect.getsource(mod)
    tree = ast.parse(source)
    calls = [
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "tone_adapter"
    ]
    assert calls == []
    assert not hasattr(mod, "tone_adapter")


def test_conversation_orchestrator_still_does_not_call_tone_adapter_directly():
    import app.voice.conversation_orchestrator as mod

    source = inspect.getsource(mod)
    tree = ast.parse(source)
    calls = [
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "tone_adapter"
    ]
    assert calls == []


def test_tone_adapter_applied_exactly_once_not_double_stripped():
    """
    Feeding already-tone-adapted text back through the engine a second time
    must not change it further (idempotent for the one real transformation
    tone_adapter performs) — guards against a double-invocation regression
    producing over-stripped text.
    """
    once = analyze_response(
        "Awesome! Let's get that booked.", user_text="hi", use_ssml=False
    ).text
    twice = analyze_response(once, user_text="hi", use_ssml=False).text
    assert once == twice == "Let's get that booked."


# ---------------------------------------------------------------------------
# 3. LiveKit can use the same shared logic where safe
# ---------------------------------------------------------------------------


class _FakeHandler:
    def __init__(self):
        self._tts_cancel = asyncio.Event()
        self._current_turn_user_text = ""
        self._current_turn_stt_confidence = 0.0
        self.prefetch_calls = []

    async def _prefetch_tts_audio(self, task):
        self.prefetch_calls.append(dict(task))
        return b"\xff" * 160

    async def _stream_tts_chunk(
        self,
        text,
        use_ssml=False,
        is_final=False,
        prefetched_bytes=None,
        pacing=None,
        previous_text=None,
    ):
        pass


async def _drive_chunk(handler, text, user_text="", use_ssml=False):
    handler._current_turn_user_text = user_text
    pipeline = TtsPipeline(handler)
    await pipeline.queue_tts(
        {"text": text, "chunk_id": 0, "use_ssml": use_ssml, "is_final": True}
    )
    task = pipeline._synthesis_tasks.get(0)
    if task is not None:
        await task


def test_livekit_style_handler_gets_same_tone_adapted_text_as_twilio_style():
    """
    Both transports funnel through the identical TtsPipeline._process_chunk,
    so a "Twilio-like" and a "browser-like" fake handler must receive the
    exact same tone-adapted text for the exact same input — proving there is
    no separate/duplicated humanization implementation per transport.
    """
    twilio_like = _FakeHandler()
    browser_like = _FakeHandler()

    asyncio.run(
        _drive_chunk(
            twilio_like, "Awesome! Here's your confirmation.", user_text="hi", use_ssml=False
        )
    )
    asyncio.run(
        _drive_chunk(
            browser_like, "Awesome! Here's your confirmation.", user_text="hi", use_ssml=False
        )
    )

    text_a = twilio_like.prefetch_calls[0]["text"]
    text_b = browser_like.prefetch_calls[0]["text"]
    assert text_a == text_b == "Here's your confirmation."


# ---------------------------------------------------------------------------
# 4-8. Text-safety guarantees
# ---------------------------------------------------------------------------


def test_normal_text_semantically_unchanged():
    text = "I can help you with that request today."
    d = analyze_response(text, user_text="hi", use_ssml=True)
    assert d.text == text


def test_numbers_remain_unchanged():
    text = "Your total comes to 42 dollars and the code is 8675309."
    d = analyze_response(text, user_text="hi", use_ssml=False)
    assert "42" in d.text
    assert "8675309" in d.text


def test_names_remain_unchanged():
    text = "Awesome! Dr. Sarah Chen will see you at your appointment."
    d = analyze_response(text, user_text="hi", use_ssml=False)
    assert "Dr. Sarah Chen" in d.text


def test_business_and_booking_information_remains_unchanged():
    text = (
        "Awesome! Your appointment is confirmed for 3:00 PM on March 5th "
        "at 123 Main Street, call us at 555-123-4567 if anything changes."
    )
    d = analyze_response(text, user_text="hi", use_ssml=False)
    assert "3:00 PM" in d.text
    assert "March 5th" in d.text
    assert "123 Main Street" in d.text
    assert "555-123-4567" in d.text
    # Only the leading chipper opener is stripped — everything after it
    # (all the factual content) must be byte-for-byte intact.
    assert d.text == (
        "Your appointment is confirmed for 3:00 PM on March 5th "
        "at 123 Main Street, call us at 555-123-4567 if anything changes."
    )


def test_url_remains_unchanged():
    text = "Great! You can find it at https://example.com/booking?id=42."
    d = analyze_response(text, user_text="hi", use_ssml=False)
    assert "https://example.com/booking?id=42" in d.text


def test_empty_text_remains_safe():
    d = analyze_response("", user_text="hi", use_ssml=False)
    assert d.text == ""
    assert d.mood == UserMood.NEUTRAL


def test_short_text_remains_safe():
    d = analyze_response("Okay.", user_text="hi", use_ssml=False)
    assert d.text == "Okay."


# ---------------------------------------------------------------------------
# 9. Feature flag disabled preserves existing (pass-through) behavior
# ---------------------------------------------------------------------------


def test_disabled_flag_returns_original_text_even_with_chipper_opener(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_HUMANIZATION_ENGINE", False
    )
    original = "Awesome! Here's your confirmation for 3:00 PM."
    d = analyze_response(original, user_text="hi", use_ssml=False)
    assert d.text == original


# ---------------------------------------------------------------------------
# 10 & 11. Failure fallback returns original text
# ---------------------------------------------------------------------------


def test_engine_failure_returns_original_text(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("engine failure")

    monkeypatch.setattr(humanization_engine, "build_turn_context", _boom, raising=False)
    monkeypatch.setattr("app.voice.turn_signals.build_turn_context", _boom)

    original = "Awesome! Here's your confirmation for 3:00 PM."
    d = analyze_response(original, user_text="hi", use_ssml=False)
    assert d.text == original
    assert d.mood == UserMood.NEUTRAL


def test_tone_adapter_failure_returns_original_text(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("tone_adapter failure")

    monkeypatch.setattr(humanization_engine, "tone_adapter", _boom)

    original = "Awesome! Here's your confirmation for 3:00 PM."
    d = analyze_response(original, user_text="hi", use_ssml=False)
    assert d.text == original


def test_tts_pipeline_survives_humanization_failure_end_to_end(monkeypatch):
    """Full pipeline check: a tone_adapter crash must not stop TTS."""

    def _boom(*args, **kwargs):
        raise RuntimeError("tone_adapter failure")

    monkeypatch.setattr(humanization_engine, "tone_adapter", _boom)

    handler = _FakeHandler()
    original = "Awesome! Here's your confirmation for 3:00 PM."
    asyncio.run(_drive_chunk(handler, original, user_text="hi", use_ssml=False))

    assert len(handler.prefetch_calls) == 1
    assert handler.prefetch_calls[0]["text"] == original
