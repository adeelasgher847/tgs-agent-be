"""
Scope guard: VOICE_STT_INCOMPLETE_FINAL_GRACE_MS must only reach Twilio
calls. LiveKitBrowserCallHandler (Share Demo Link) reported humanization
as "working fine" earlier this session -- its turn-taking timing must not
be altered by this fix.

VoiceOrchestrator gates this via `hasattr(h, "websocket")`:
BidirectionalStreamHandler (Twilio) sets `self.websocket` to the raw
Media Streams WebSocket; LiveKitBrowserCallHandler has no such attribute
(it publishes/subscribes via LiveKit room tracks instead).
"""

from __future__ import annotations

import inspect

from app.voice import voice_orchestrator as vo_module


def test_grace_ms_is_gated_on_twilio_only_websocket_attribute():
    source = inspect.getsource(vo_module)
    assert "_incomplete_final_grace_ms" in source
    assert 'hasattr(h, "websocket")' in source


def test_livekit_browser_handler_has_no_websocket_attribute():
    from app.voice.livekit_browser_call_handler import LiveKitBrowserCallHandler

    assert not hasattr(LiveKitBrowserCallHandler, "websocket")
    source = inspect.getsource(LiveKitBrowserCallHandler)
    assert "self.websocket" not in source


def test_bidirectional_stream_handler_sets_websocket():
    from app.routers.bidirectional_stream import BidirectionalStreamHandler

    source = inspect.getsource(BidirectionalStreamHandler)
    assert "self.websocket = websocket" in source
