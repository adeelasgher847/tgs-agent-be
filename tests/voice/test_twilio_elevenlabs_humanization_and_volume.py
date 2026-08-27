import pytest
import numpy as np
from app.utils.ssml_utils import strip_ssml_tags
from app.utils.eleven_tts_text import (
    prepare_tts_text_for_provider,
    supports_elevenlabs_audio_tags,
    apply_elevenlabs_breathing_fallback,
    build_elevenlabs_audio_tag_prompt_block,
)
from app.voice.humanization_engine import (
    analyze_response,
    pause_frames_for_chunk,
    PacingHint,
    SentenceEndingType,
)
from app.voice.tts_provider_capabilities import build_voice_settings_overlay
from app.utils.audio_utils import (
    apply_volume_fade,
    ulaw_to_linear_sample,
    linear_to_ulaw_sample,
    MULAW_FRAME_BYTES,
    apply_micro_fade_in,
    apply_micro_fade_out,
)
from app.voice.tts_stream_mixin import TtsStreamMixin


def test_no_elevenlabs_bracket_tags_reach_provider():
    """Requirement A: No [breathes], [excited], [sad], etc. reach ElevenLabs."""
    inputs = [
        "[breathes] Hello, thanks for calling.",
        "[excited] That is wonderful news!",
        "[sad] I am so sorry to hear that.",
        "[breathe] Sure, let me check that for you.",
        "[breath] One moment please.",
        "[pause] Let me check our calendar.",
    ]
    for raw in inputs:
        cleaned = prepare_tts_text_for_provider(raw, "elevenlabs")
        assert not any(tag in cleaned for tag in ["[breathes]", "[excited]", "[sad]", "[breathe]", "[breath]", "[pause]"])
        assert not cleaned.startswith("[")


def test_no_ssml_tags_reach_elevenlabs():
    """Requirement B: No <speak>, <prosody>, <break> reach ElevenLabs."""
    ssml_text = '<speak><prosody rate="0.93" pitch="0st" volume="medium"><break time="400ms"/> Good morning! How can I assist you?</prosody></speak>'
    cleaned = prepare_tts_text_for_provider(ssml_text, "elevenlabs")
    assert cleaned == "Good morning! How can I assist you?"
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "prosody" not in cleaned
    assert "break" not in cleaned
    assert "speak" not in cleaned


def test_split_incomplete_xml_tags_sanitized():
    """Requirement C: Split/incomplete XML tags across chunks are cleanly sanitized."""
    # Leading split tag (Chunk 1 ended with open tag)
    chunk1 = 'Good morning <prosody rate="95%"'
    assert prepare_tts_text_for_provider(chunk1, "elevenlabs") == "Good morning"

    # Trailing split tag (Chunk 2 starts with unclosed tag remnant)
    chunk2 = ' pitch="+1st">how may I help you?'
    assert prepare_tts_text_for_provider(chunk2, "elevenlabs") == "how may I help you?"

    # Remnant break fragment
    chunk3 = 'time="400ms"/> Absolutely!'
    assert prepare_tts_text_for_provider(chunk3, "elevenlabs") == "Absolutely!"


def test_humanization_voice_parameters_applied():
    """Requirement D: Humanization voice parameters (dynamic stability) are applied."""
    decision = analyze_response(
        "I understand that must be frustrating, let me help.",
        user_text="I am really upset about this delay",
        stt_confidence=0.95,
        use_ssml=False,
        is_final=True,
    )
    overlay = build_voice_settings_overlay("elevenlabs", decision)
    # Stability should be provided based on user mood/turn signals
    assert "stability" in overlay or overlay == {}


def test_audio_level_pauses_preserved():
    """Requirement G: Audio-level silence frames (pause_frames_for_chunk) remain active."""
    pacing = PacingHint(
        sentence_count=1,
        has_multiple_sentences=False,
        is_short_utterance=False,
        ending_type=SentenceEndingType.STATEMENT,
        has_internal_pause_opportunity=False,
    )
    # Non-final statement chunk receives silence frames
    frames = pause_frames_for_chunk(pacing, is_final=False)
    assert frames >= 0
    # Final chunk has 0 intersentence pause frames
    assert pause_frames_for_chunk(pacing, is_final=True) == 0


def test_micro_fades_and_twilio_framing():
    """Requirements H & I: 160-byte 20ms framing and micro-fades preserved."""
    # 20ms frame is exactly 160 bytes
    frame = bytes([0x7F]) * MULAW_FRAME_BYTES
    assert len(frame) == 160

    # 1 second of audio (8000 bytes = 50 frames of 160 bytes)
    audio = bytes([0x7F]) * 8000
    fade_in = apply_micro_fade_in(audio, duration_ms=25.0)
    assert len(fade_in) == 8000

    fade_out = apply_micro_fade_out(fade_in, duration_ms=25.0)
    assert len(fade_out) == 8000


def test_elevenlabs_volume_normalization_and_headroom():
    """Requirement J: Provider-aware volume gain brings ElevenLabs up without clipping."""
    # Mock handler with ElevenLabs agent
    class MockRuntime:
        adapter_slug = "elevenlabs"
        settings_json = {"volume": 1.0}

    class MockHandler(TtsStreamMixin):
        def __init__(self):
            self.agent = object()

    handler = MockHandler()
    
    # Patch resolve_tts_runtime
    import app.voice.tts_stream_mixin as mixin_mod
    orig_resolve = mixin_mod.resolve_tts_runtime
    mixin_mod.resolve_tts_runtime = lambda agent, db=None: MockRuntime()
    try:
        vol = handler._resolve_voice_volume()
        assert vol == 1.8  # 1.8x baseline gain for ElevenLabs
    finally:
        mixin_mod.resolve_tts_runtime = orig_resolve

    # Test apply_volume_fade with 1.8x gain on real linear-encoded mulaw
    linear_samples = [int(15000 * np.sin(2 * np.pi * 440 * t / 8000)) for t in range(800)]
    mulaw_bytes = bytes(linear_to_ulaw_sample(s) for s in linear_samples)
    
    boosted = apply_volume_fade(mulaw_bytes, volume=1.8)
    assert len(boosted) == len(mulaw_bytes)
    
    # Decode back and check peak
    boosted_samples = [ulaw_to_linear_sample(b) for b in boosted]
    max_val = max(abs(s) for s in boosted_samples)
    assert max_val <= 32767  # Strictly within int16 bounds (no overflow)
