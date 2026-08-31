"""
Regression tests for the PII-redaction bug in
BidirectionalStreamHandler._process_transcript (app/routers/bidirectional_stream.py).

Bug: client speech was unconditionally run through app.core.pii_redactor.redact_pii()
before being persisted/broadcast, regardless of the call flow's HIPAA setting. This
masked PII (phone numbers, emails, etc.) on every Twilio call and duplicated/conflicted
with the already-correctly-gated HIPAA redaction pipeline
(call_control_mixin._add_to_transcript -> transcript_service.add_and_broadcast_message
-> dlp_service.redact_phi_if_hipaa), which only redacts when
call_flow.hipaa_compliance is True.

Fix: _process_transcript now passes the raw transcript straight to
_add_to_transcript, which already resolves hipaa_enabled from self.call_flow and
routes redaction decisions through the HIPAA-gated pipeline. This file confirms:
  1. The raw transcript (with PII) reaches _add_to_transcript unmodified — the
     redundant unconditional redact_pii() call is gone.
  2. app.core.pii_redactor.redact_pii is never invoked from this code path.

HIPAA-enabled-flow redaction itself is already covered end-to-end by
tests/api/v2/test_hipaa.py::TestCallControlMixinHipaaEnabled and
TestTranscriptServiceHipaaRedaction — untouched by this fix and not duplicated here.

Run: pytest tests/services/test_bidirectional_transcript_pii_redaction.py -q
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.routers.bidirectional_stream import BidirectionalStreamHandler as Handler


def _handler_for_process_transcript() -> Handler:
    """Minimal handler exercising _process_transcript's client-transcript persistence step.

    All gating checks upstream of the persistence call (barge-in, dedupe, self-echo,
    goodbye/voicemail/screener/IVR/anti-bot detection) are stubbed to pass through so we
    reach the `_add_to_transcript("client", transcript, ...)` call under test.
    """
    h = object.__new__(Handler)

    # Barge-in gate (skipped entirely — no TTS playing).
    h._is_tts_playing = False

    # Lock + dedupe/self-echo state.
    h._voice_transcript_lock = asyncio.Lock()
    h._cancel_silence_watchdog = lambda: None  # type: ignore[method-assign, assignment]
    h._should_accept_final_transcript = lambda *_a, **_k: True  # type: ignore[method-assign, assignment]
    h._stt_last_final_raw = ""
    h._stt_last_final_monotonic = 0.0
    h._STT_DEDUP_FINAL_WINDOW_SEC = 6.0
    h._is_agent_self_echo = lambda *_a, **_k: False  # type: ignore[method-assign, assignment]

    h._voice_metrics = AsyncMock()
    h._voice_metrics.begin_turn_at_stt_final = lambda *_a, **_k: None
    h._metric_stt_final_ts = 0.0

    h._check_and_end_call_if_goodbye = AsyncMock(return_value=False)
    h._check_and_end_call_if_voicemail = AsyncMock(return_value=False)
    h._check_and_handle_call_screener = AsyncMock(return_value=False)
    h._check_and_handle_ivr_and_hold = AsyncMock(return_value=False)
    h._check_and_handle_anti_bot = AsyncMock(return_value=False)
    h._check_and_handle_compliance_monitoring = AsyncMock(return_value=None)

    h._in_progress_sent = True  # skip "in-progress" status branch
    h._send_in_progress_status = AsyncMock()

    h._add_to_transcript = AsyncMock()  # captured / asserted on in tests
    h._update_booking_memory_from_user_turn = lambda *_a, **_k: None  # type: ignore[method-assign, assignment]

    h._speculative_prefetch_task = None
    h._run_speculative_tts_prefetch = AsyncMock()
    h._complete_llm_turn_after_stt_final = AsyncMock()

    return h


def test_process_transcript_persists_raw_client_speech_no_redaction() -> None:
    """No HIPAA gating happens in _process_transcript itself — raw transcript with PII
    (phone number + email) must reach _add_to_transcript unmodified. (Whether it is
    subsequently redacted depends solely on call_flow.hipaa_compliance, resolved
    downstream inside _add_to_transcript — see tests/api/v2/test_hipaa.py.)"""

    async def _body() -> None:
        h = _handler_for_process_transcript()
        raw_transcript = "Call me at 555-123-4567 or email me at jane@example.com"

        await Handler._process_transcript(h, raw_transcript, 0.9)

        h._add_to_transcript.assert_awaited_once()
        call_args = h._add_to_transcript.await_args
        assert call_args.args[0] == "client"
        assert call_args.args[1] == raw_transcript
        assert "555-123-4567" in call_args.args[1]
        assert "jane@example.com" in call_args.args[1]

    asyncio.run(_body())


def test_process_transcript_never_calls_redact_pii() -> None:
    """The removed redundant redact_pii() call must not be reintroduced."""

    async def _body() -> None:
        h = _handler_for_process_transcript()

        with patch("app.core.pii_redactor.redact_pii") as mock_redact_pii:
            await Handler._process_transcript(
                h, "my number is 555-987-6543", 0.9
            )
            mock_redact_pii.assert_not_called()

    asyncio.run(_body())
