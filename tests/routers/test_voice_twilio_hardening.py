"""
Unit tests for the Twilio security-hardening fixes on the Media Streams webhook
surface (branch fix/twilio-security-hardening):

1. X-Twilio-Signature validation re-enabled / added on previously-unprotected
   webhooks in app/routers/voice.py and app/routers/voice_gather.py.
2. SSRF guard on RecordingUrl before it is fetched with injected credentials.
3. Twilio Media Streams `clear` event sent on barge-in.
4. Twilio CallStatus -> internal status mapping gaps (queued / canceled).
5. Retry/backoff for transient (429 / 5xx) Twilio call-creation failures.

DB is an in-memory SQLite fixture (same pattern as tests/routers/test_amd_webhook.py).
Signature math itself is not exercised here (that's twilio_validation's own
responsibility) — these tests verify the *call sites* enforce validation and
propagate a 403 on failure, using the same bypass/patch pattern as
test_amd_webhook.py.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


_SHARED_CONN = sqlite3.connect(":memory:", check_same_thread=False)
_engine = create_engine(
    "sqlite://", creator=lambda: _SHARED_CONN, connect_args={"check_same_thread": False}
)
_Session = sessionmaker(bind=_engine)


def _setup_db():
    import app.models.agent  # noqa: F401
    import app.models.call_session  # noqa: F401
    import app.models.tenant  # noqa: F401
    import app.models.user  # noqa: F401
    import app.models.plan  # noqa: F401
    import app.models.subscription  # noqa: F401
    import app.models.usage_record  # noqa: F401
    import app.models.role  # noqa: F401
    import app.models.phone_number  # noqa: F401

    Base.metadata.drop_all(bind=_engine)
    Base.metadata.create_all(bind=_engine)


@pytest.fixture()
def db():
    _setup_db()
    session = _Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def _bypass_signature_validation():
    """Most tests exercise business logic, not signature validation — bypass it
    by default; signature-focused tests override this explicitly."""
    with patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", True):
        yield


class _FakeURL:
    def __init__(self, path="/api/v1/voice/webhook/call-events"):
        self.path = path
        self.query = ""
        self.scheme = "https"
        self.netloc = "example.com"


class _FakeRequest:
    def __init__(self, form_data: dict, headers: dict | None = None):
        self._form_data = form_data
        self.headers = headers or {}
        self.method = "POST"
        self.url = _FakeURL()

    async def form(self):
        return self._form_data

    async def body(self):
        return b""


def _make_tenant(db):
    from app.models.tenant import Tenant

    t = Tenant(name="WS", schema_name=f"ws_{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t.id


def _make_agent(db, tenant_id):
    from app.models.agent import Agent

    a = Agent(tenant_id=tenant_id, name="TestAgent", system_prompt="", status="ready")
    db.add(a)
    db.commit()
    db.refresh(a)
    return a.id


def _make_user(db):
    from app.models.user import User

    u = User(
        id=uuid.uuid4(),
        first_name="Voice",
        last_name="Test",
        email=f"voice-{uuid.uuid4().hex}@example.com",
        hashed_password="x",
    )
    db.add(u)
    db.flush()
    return u.id


def _make_call_session(db, tenant_id, agent_id, user_id):
    from datetime import datetime, timezone

    from app.models.call_session import CallSession

    cs = CallSession(
        id=uuid.uuid4(),
        user_id=user_id,
        agent_id=agent_id,
        tenant_id=tenant_id,
        start_time=datetime.now(timezone.utc),
        status="active",
        call_type="outbound",
        twilio_call_sid="CAtest123",
    )
    db.add(cs)
    db.commit()
    db.refresh(cs)
    return cs


# ─────────────────────────────────────────────────────────────────────────────
# 1. Signature validation
# ─────────────────────────────────────────────────────────────────────────────


class TestCallEventsWebhookSignature:
    def test_rejects_missing_auth_headers_when_not_bypassed(self, db):
        from app.routers.voice import handle_call_events_webhook

        request = _FakeRequest({"CallStatus": "ringing", "CallSid": "CA1"}, headers={})

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=None,
                    userId=None,
                    callSessionId=None,
                    timeout=None,
                    body="",
                    db=db,
                )
            )
        assert exc_info.value.status_code == 403

    def test_rejects_invalid_twilio_signature(self, db):
        from app.routers.voice import handle_call_events_webhook

        request = _FakeRequest(
            {"CallStatus": "ringing", "CallSid": "CA1"},
            headers={"X-Twilio-Signature": "bogus"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            patch("app.routers.voice.validate_twilio_signature", return_value=False),
            patch(
                "app.routers.voice.validate_twilio_signature_with_token", return_value=False
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=None,
                    userId=None,
                    callSessionId=None,
                    timeout=None,
                    body="",
                    db=db,
                )
            )
        assert exc_info.value.status_code == 403

    def test_accepts_valid_twilio_signature(self, db):
        from app.routers.voice import handle_call_events_webhook

        request = _FakeRequest(
            {"CallStatus": "ringing", "CallSid": "CA1"},
            headers={"X-Twilio-Signature": "goodsig"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            patch("app.routers.voice.validate_twilio_signature", return_value=True),
        ):
            resp = asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=None,
                    userId=None,
                    callSessionId=None,
                    timeout=None,
                    body="",
                    db=db,
                )
            )
        assert resp.status_code == 200

    def test_forged_authorization_header_no_longer_bypasses_signature_check(self, db):
        """
        Regression guard: this endpoint used to accept ANY `Authorization: Bearer
        <anything>` header via a no-op `validate_webrtc_auth()` stub, letting an
        attacker skip X-Twilio-Signature enforcement entirely by sending an
        Authorization header instead of (or without) a valid Twilio signature.
        This endpoint is Twilio-status-callback-only (no real WebRTC caller exists
        anywhere in the codebase) — the bypass branch was removed, not fixed.
        """
        from app.routers.voice import handle_call_events_webhook

        request = _FakeRequest(
            {"CallStatus": "completed", "CallSid": "CA1"},
            headers={"Authorization": "Bearer attacker-supplied-token"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=None,
                    userId=None,
                    callSessionId=None,
                    timeout=None,
                    body="",
                    db=db,
                )
            )
        assert exc_info.value.status_code == 403

    def test_validate_webrtc_auth_no_longer_imported_or_referenced(self):
        """The dead WebRTC-bypass branch (and its no-op stub import) must be gone,
        not just unreachable."""
        import app.routers.voice as voice_module

        assert not hasattr(voice_module, "validate_webrtc_auth")


class TestRecordingCallbackSignature:
    def test_rejects_invalid_signature(self, db):
        from app.routers.voice import handle_recording_callback

        request = _FakeRequest(
            {"RecordingStatus": "completed", "CallSid": "CA1"},
            headers={"X-Twilio-Signature": "bogus"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            patch("app.routers.voice.validate_twilio_signature", return_value=False),
            patch(
                "app.routers.voice.validate_twilio_signature_with_token", return_value=False
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(
                handle_recording_callback(
                    request=request,
                    agentId=None,
                    userId=None,
                    callSessionId=None,
                    body="",
                    db=db,
                )
            )
        assert exc_info.value.status_code == 403

    def test_accepts_valid_signature_for_status_callback(self, db):
        from app.routers.voice import handle_recording_callback

        request = _FakeRequest(
            {"RecordingStatus": "completed", "CallSid": "CA1"},
            headers={"X-Twilio-Signature": "goodsig"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            patch("app.routers.voice.validate_twilio_signature", return_value=True),
        ):
            resp = asyncio.run(
                handle_recording_callback(
                    request=request,
                    agentId=None,
                    userId=None,
                    callSessionId=None,
                    body="",
                    db=db,
                )
            )
        assert resp.status_code == 200


class TestGatherSpeechWebhookSignature:
    def test_rejects_invalid_signature(self, db):
        from app.routers.voice import handle_gather_speech_webhook

        request = _FakeRequest(
            {"CallSid": "CA1", "SpeechResult": "hello"},
            headers={"X-Twilio-Signature": "bogus"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            patch("app.routers.voice.validate_twilio_signature", return_value=False),
            patch(
                "app.routers.voice.validate_twilio_signature_with_token", return_value=False
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(
                handle_gather_speech_webhook(
                    request=request,
                    agentId=None,
                    callSessionId=None,
                    body="",
                    db=db,
                )
            )
        assert exc_info.value.status_code == 403


class TestRecordingStatusWebhookSignature:
    def test_rejects_invalid_signature(self, db):
        from app.routers.voice import handle_recording_status_webhook

        request = _FakeRequest(
            {"RecordingSid": "RE1", "CallSid": "CA1", "RecordingStatus": "completed"},
            headers={"X-Twilio-Signature": "bogus"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            patch("app.routers.voice.validate_twilio_signature", return_value=False),
            patch(
                "app.routers.voice.validate_twilio_signature_with_token", return_value=False
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(handle_recording_status_webhook(request=request, db=db))
        assert exc_info.value.status_code == 403

    def test_accepts_valid_signature_and_updates_session(self, db):
        from app.routers.voice import handle_recording_status_webhook

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        request = _FakeRequest(
            {
                "RecordingSid": "RE1",
                "CallSid": cs.twilio_call_sid,
                "RecordingStatus": "completed",
                "RecordingUrl": "https://api.twilio.com/2010-04-01/Accounts/ACxx/Recordings/RExx",
                "RecordingDuration": "5",
            },
            headers={"X-Twilio-Signature": "goodsig"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            patch("app.routers.voice.validate_twilio_signature", return_value=True),
        ):
            resp = asyncio.run(handle_recording_status_webhook(request=request, db=db))
        assert resp.status_code == 200
        db.refresh(cs)
        assert cs.recording_url == "https://api.twilio.com/2010-04-01/Accounts/ACxx/Recordings/RExx"


class TestStreamingGreetingWebhookSignature:
    def test_rejects_invalid_signature(self, db):
        from app.routers.voice_gather import streaming_greeting_webhook

        request = _FakeRequest(
            {"CallSid": "CA1", "CallStatus": "in-progress"},
            headers={"X-Twilio-Signature": "bogus"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            patch("app.routers.voice_gather.validate_twilio_signature", return_value=False),
            patch(
                "app.routers.voice_gather.validate_twilio_signature_with_token",
                return_value=False,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(
                streaming_greeting_webhook(
                    request=request,
                    agentId=None,
                    userId=None,
                    callSessionId=None,
                    body="",
                    db=db,
                )
            )
        assert exc_info.value.status_code == 403

    def test_accepts_valid_signature(self, db):
        from app.routers.voice_gather import streaming_greeting_webhook

        request = _FakeRequest(
            {"CallSid": "CA1", "CallStatus": "queued"},
            headers={"X-Twilio-Signature": "goodsig"},
        )

        with (
            patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False),
            patch("app.routers.voice_gather.validate_twilio_signature", return_value=True),
        ):
            resp = asyncio.run(
                streaming_greeting_webhook(
                    request=request,
                    agentId=None,
                    userId=None,
                    callSessionId=None,
                    body="",
                    db=db,
                )
            )
        # "queued" isn't answered/in-progress yet -> pause/redirect TwiML, not an error
        assert resp.status_code == 200
        assert "Redirect" in resp.body.decode()


# ─────────────────────────────────────────────────────────────────────────────
# 2. SSRF guard on RecordingUrl
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordingUrlSSRFGuard:
    def test_build_authenticated_recording_url_accepts_documented_shape(self):
        from app.routers.voice import _build_authenticated_recording_url

        url = _build_authenticated_recording_url(
            "https://api.twilio.com/2010-04-01/Accounts/ACxx/Recordings/RExx",
            "ACxx",
            "secret",
        )
        assert url == (
            "https://ACxx:secret@api.twilio.com"
            "/2010-04-01/Accounts/ACxx/Recordings/RExx.wav"
        )

    def test_build_authenticated_recording_url_accepts_relative_path(self):
        from app.routers.voice import _build_authenticated_recording_url

        url = _build_authenticated_recording_url(
            "/2010-04-01/Accounts/ACxx/Recordings/RExx", "ACxx", "secret"
        )
        assert url == (
            "https://ACxx:secret@api.twilio.com"
            "/2010-04-01/Accounts/ACxx/Recordings/RExx.wav"
        )

    def test_build_authenticated_recording_url_rejects_foreign_host(self):
        from app.routers.voice import _build_authenticated_recording_url

        url = _build_authenticated_recording_url(
            "https://evil.example.com/2010-04-01/Accounts/ACxx/Recordings/RExx",
            "ACxx",
            "secret",
        )
        assert url is None

    def test_build_authenticated_recording_url_rejects_no_op_replace_bypass(self):
        """Regression guard: a URL that doesn't start with the Twilio host must
        not silently pass through the old naive .replace()-based construction."""
        from app.routers.voice import _build_authenticated_recording_url

        url = _build_authenticated_recording_url(
            "http://attacker.example.com/steal-me", "ACxx", "secret"
        )
        assert url is None

    def test_build_authenticated_recording_url_rejects_malformed_path(self):
        from app.routers.voice import _build_authenticated_recording_url

        url = _build_authenticated_recording_url(
            "https://api.twilio.com/../../etc/passwd", "ACxx", "secret"
        )
        assert url is None

    def test_recording_callback_rejects_malicious_recording_url(self, db):
        """End-to-end: a malicious RecordingUrl must never reach requests.get()."""
        from app.routers.voice import handle_recording_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        request = _FakeRequest(
            {
                "RecordingUrl": "https://attacker.example.com/payload",
                "CallSid": cs.twilio_call_sid,
            }
        )

        with (
            patch(
                "app.routers.voice.get_twilio_credentials_for_call",
                return_value=("ACxx", "secret"),
            ),
            patch("app.routers.voice.requests.get") as mock_get,
        ):
            resp = asyncio.run(
                handle_recording_callback(
                    request=request,
                    agentId=str(agent_id),
                    userId=None,
                    callSessionId=str(cs.id),
                    body="",
                    db=db,
                )
            )

        mock_get.assert_not_called()
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 3. Twilio Media Streams `clear` event on barge-in
# ─────────────────────────────────────────────────────────────────────────────


class TestBargeInClearEvent:
    def _handler(self, *, stream_sid="MZ123"):
        from app.routers.bidirectional_stream import BidirectionalStreamHandler

        h = object.__new__(BidirectionalStreamHandler)
        h.websocket = MagicMock()
        h.websocket.send_json = AsyncMock()
        h.stream_sid = stream_sid
        h._llm_response_task = None
        h._tts_pipeline = None
        h._rag_prefetch_task = None
        h._speculative_prefetch_task = None
        h._rag_prefetch_user_text = ""
        return h

    def test_send_twilio_clear_event_sends_clear_with_stream_sid(self):
        h = self._handler(stream_sid="MZ999")
        asyncio.run(h._send_twilio_clear_event())
        h.websocket.send_json.assert_awaited_once_with(
            {"event": "clear", "streamSid": "MZ999"}
        )

    def test_send_twilio_clear_event_noop_without_stream_sid(self):
        h = self._handler(stream_sid=None)
        asyncio.run(h._send_twilio_clear_event())
        h.websocket.send_json.assert_not_called()

    def test_cancel_inflight_llm_response_sends_clear_event(self):
        """The barge-in cancellation path must flush Twilio's already-buffered
        audio, not just stop our own local TTS task."""
        h = self._handler(stream_sid="MZbargein")
        asyncio.run(h._cancel_inflight_llm_response())
        h.websocket.send_json.assert_awaited_once_with(
            {"event": "clear", "streamSid": "MZbargein"}
        )

    def test_cancel_inflight_llm_response_cancels_tts_pipeline_and_sends_clear(self):
        h = self._handler(stream_sid="MZbargein2")
        h._tts_pipeline = MagicMock()
        h._tts_pipeline.cancel_current_and_clear_queue = AsyncMock()

        asyncio.run(h._cancel_inflight_llm_response())

        h._tts_pipeline.cancel_current_and_clear_queue.assert_awaited_once()
        h.websocket.send_json.assert_awaited_once_with(
            {"event": "clear", "streamSid": "MZbargein2"}
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Twilio CallStatus -> internal status mapping gaps
# ─────────────────────────────────────────────────────────────────────────────


class TestCallStatusMapping:
    def test_queued_maps_to_initiated(self):
        from app.routers.voice import _TWILIO_TO_INTERNAL_STATUS

        assert _TWILIO_TO_INTERNAL_STATUS["queued"] == "initiated"

    def test_canceled_maps_to_failed(self):
        from app.routers.voice import _TWILIO_TO_INTERNAL_STATUS

        assert _TWILIO_TO_INTERNAL_STATUS["canceled"] == "failed"

    def test_existing_mappings_unchanged(self):
        from app.routers.voice import _TWILIO_TO_INTERNAL_STATUS

        assert _TWILIO_TO_INTERNAL_STATUS["initiated"] == "initiated"
        assert _TWILIO_TO_INTERNAL_STATUS["ringing"] == "ringing"
        assert _TWILIO_TO_INTERNAL_STATUS["in-progress"] == "connected"
        assert _TWILIO_TO_INTERNAL_STATUS["answered"] == "connected"
        assert _TWILIO_TO_INTERNAL_STATUS["completed"] == "completed"
        assert _TWILIO_TO_INTERNAL_STATUS["failed"] == "failed"
        assert _TWILIO_TO_INTERNAL_STATUS["no-answer"] == "no_answer"
        assert _TWILIO_TO_INTERNAL_STATUS["busy"] == "no_answer"

    def test_batch_ended_reason_mapping_includes_canceled(self):
        """notify_batch_call_ended() silently no-ops on an unmapped ended_reason —
        without this entry a canceled batch call would never leave 'active'."""
        from app.services.batch_call_completion_service import _TWILIO_TO_ENDED_REASON

        assert _TWILIO_TO_ENDED_REASON["canceled"] == "failed"
        assert _TWILIO_TO_ENDED_REASON["failed"] == "failed"


class TestQueuedStatusIsPreConnectionNoOp:
    """
    Confirms the reviewer's stated assumption: unlike "canceled", "queued" needs no
    business-effect wiring. Credit monitoring only starts on first media packet
    (app/voice/tts_stream_mixin.py, on call answer) and a batch record is marked
    "active" at dispatch time (before any status webhook arrives) — so at "queued"
    (before ringing) there is no batch/credit state open yet that needs closing.
    """

    def test_queued_does_not_stop_credit_monitoring_or_notify_batch(self, db):
        from app.routers.voice import handle_call_events_webhook

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        request = _FakeRequest(
            {
                "CallStatus": "queued",
                "CallSid": cs.twilio_call_sid,
                "From": "+15550001111",
                "To": "+15550002222",
                "Direction": "outbound-api",
            }
        )

        with (
            patch("app.routers.voice.credit_service.stop_credit_monitoring") as mock_stop,
            patch(
                "app.routers.voice.notify_batch_call_ended", new=AsyncMock()
            ) as mock_notify,
        ):
            resp = asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=str(agent_id),
                    userId=None,
                    callSessionId=str(cs.id),
                    timeout=None,
                    body="",
                    db=db,
                )
            )

        assert resp.status_code == 200
        mock_stop.assert_not_called()
        mock_notify.assert_not_awaited()


class TestCanceledStatusSideEffects:
    """
    Blocker 2 follow-up: `handle_call_events_webhook`'s business-effect dispatch
    branches on the literal Twilio CallStatus string, not the mapped internal
    status, so "canceled" needs its own elif mirroring "failed" — otherwise a
    canceled outbound (esp. batch) call is left stuck without ever being marked
    ended (no webhook fire, no credit-monitoring stop, no batch completion).
    """

    def test_canceled_stops_credit_monitoring_and_notifies_batch(self, db):
        from app.routers.voice import handle_call_events_webhook

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        request = _FakeRequest(
            {
                "CallStatus": "canceled",
                "CallSid": cs.twilio_call_sid,
                "From": "+15550001111",
                "To": "+15550002222",
                "Direction": "outbound-api",
            }
        )

        with (
            patch("app.routers.voice.credit_service.stop_credit_monitoring") as mock_stop,
            patch(
                "app.routers.voice.notify_batch_call_ended", new=AsyncMock()
            ) as mock_notify,
            patch("app.services.webhook_service.fire_webhooks", new=AsyncMock()),
        ):
            resp = asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=str(agent_id),
                    userId=None,
                    callSessionId=str(cs.id),
                    timeout=None,
                    body="",
                    db=db,
                )
            )

        assert resp.status_code == 200
        mock_stop.assert_called_once_with(cs.id)
        mock_notify.assert_awaited_once()
        assert mock_notify.call_args.args[1] == cs.id
        assert mock_notify.call_args.args[2] == "canceled"

    def test_canceled_fires_call_failed_webhook(self, db):
        from app.routers.voice import handle_call_events_webhook

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        request = _FakeRequest(
            {
                "CallStatus": "canceled",
                "CallSid": cs.twilio_call_sid,
                "From": "+15550001111",
                "To": "+15550002222",
                "Direction": "outbound-api",
            }
        )

        background_tasks = BackgroundTasks()

        with (
            patch("app.routers.voice.credit_service.stop_credit_monitoring"),
            patch("app.routers.voice.notify_batch_call_ended", new=AsyncMock()),
        ):
            asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=background_tasks,
                    agentId=str(agent_id),
                    userId=None,
                    callSessionId=str(cs.id),
                    timeout=None,
                    body="",
                    db=db,
                )
            )

        # A call.failed webhook (reason=canceled) was scheduled as a background task.
        assert len(background_tasks.tasks) == 1
        scheduled = background_tasks.tasks[0]
        assert scheduled.args[1] == "call.failed"
        assert scheduled.args[2]["reason"] == "canceled"

    def test_canceled_status_is_persisted_via_fresh_db_read(self, db):
        """
        Blocker B fix: the "canceled" -> "failed" status mutation must actually be
        committed, not just set on the in-memory ORM object. Verified via a fresh
        query on a separate Session bound to the same underlying connection (not
        db.refresh() on the same session/object, which would trivially pass even
        without a commit as long as the object is still attached and unexpired).
        """
        from app.routers.voice import handle_call_events_webhook

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        cs_id = cs.id

        request = _FakeRequest(
            {
                "CallStatus": "canceled",
                "CallSid": cs.twilio_call_sid,
                "From": "+15550001111",
                "To": "+15550002222",
                "Direction": "outbound-api",
            }
        )

        with (
            patch("app.routers.voice.credit_service.stop_credit_monitoring"),
            patch("app.routers.voice.notify_batch_call_ended", new=AsyncMock()),
        ):
            asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=str(agent_id),
                    userId=None,
                    callSessionId=str(cs.id),
                    timeout=None,
                    body="",
                    db=db,
                )
            )

        fresh_session = _Session()
        try:
            from app.models.call_session import CallSession

            reloaded = fresh_session.get(CallSession, cs_id)
            assert reloaded.status == "failed"
        finally:
            fresh_session.close()


class TestTerminalStatusPersistence:
    """
    Blocker B regression coverage: "failed" / "busy" / "no-answer" previously had
    the exact same missing-commit bug as "canceled" (they all only mutated the
    in-memory call_session.status with no db.commit() anywhere in the request).
    Each is verified via a fresh Session/query, not db.refresh() on the same
    session (which doesn't prove persistence).
    """

    def _run_and_reload(self, db, cs, call_status: str):
        from app.models.call_session import CallSession
        from app.routers.voice import handle_call_events_webhook

        request = _FakeRequest(
            {
                "CallStatus": call_status,
                "CallSid": cs.twilio_call_sid,
                "From": "+15550001111",
                "To": "+15550002222",
                "Direction": "outbound-api",
            }
        )

        with (
            patch("app.routers.voice.credit_service.stop_credit_monitoring"),
            patch("app.routers.voice.notify_batch_call_ended", new=AsyncMock()),
        ):
            asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=str(cs.agent_id),
                    userId=None,
                    callSessionId=str(cs.id),
                    timeout=None,
                    body="",
                    db=db,
                )
            )

        fresh_session = _Session()
        try:
            return fresh_session.get(CallSession, cs.id).status
        finally:
            fresh_session.close()

    def test_failed_status_is_persisted(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        assert self._run_and_reload(db, cs, "failed") == "failed"

    def test_busy_status_is_persisted_as_no_answer(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        assert self._run_and_reload(db, cs, "busy") == "no_answer"

    def test_no_answer_status_is_persisted(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        assert self._run_and_reload(db, cs, "no-answer") == "no_answer"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Retry/backoff for transient Twilio call-creation failures
# ─────────────────────────────────────────────────────────────────────────────


class TestTwilioCallRetryBackoff:
    """
    `calls.create` is non-idempotent and side-effecting (it dials the customer) and
    Twilio has no idempotency-key mechanism for it. Only 429 is safe to retry —
    Twilio guarantees a 429 means the call was never queued/dialed. A 5xx is
    ambiguous (may have already dialed before erroring) and must NOT be retried,
    to avoid double-dialing a customer with no way to detect/track the duplicate
    (only one call_session.twilio_call_sid is ever stored).
    """

    def test_is_transient_twilio_error_true_only_for_429(self):
        from app.services.voice_call_service import _is_transient_twilio_error

        assert _is_transient_twilio_error(SimpleNamespace(status=429)) is True

    def test_is_transient_twilio_error_false_for_5xx(self):
        """5xx must NOT be retried — ambiguous whether Twilio already dialed."""
        from app.services.voice_call_service import _is_transient_twilio_error

        assert _is_transient_twilio_error(SimpleNamespace(status=500)) is False
        assert _is_transient_twilio_error(SimpleNamespace(status=502)) is False
        assert _is_transient_twilio_error(SimpleNamespace(status=503)) is False

    def test_is_transient_twilio_error_false_for_permanent_4xx(self):
        from app.services.voice_call_service import _is_transient_twilio_error

        # e.g. Twilio 21211 (invalid 'To' number) / 21214 (unverified caller id)
        assert _is_transient_twilio_error(SimpleNamespace(status=400)) is False
        assert _is_transient_twilio_error(SimpleNamespace(status=401)) is False
        assert _is_transient_twilio_error(SimpleNamespace(status=404)) is False

    def test_retries_429_then_succeeds(self):
        from twilio.base.exceptions import TwilioRestException

        from app.services.voice_call_service import _create_twilio_call_with_retry

        call_fn = MagicMock(
            side_effect=[
                TwilioRestException(status=429, uri="/Calls", msg="rate limited"),
                TwilioRestException(status=429, uri="/Calls", msg="rate limited"),
                SimpleNamespace(sid="CAsuccess"),
            ]
        )

        with patch("app.services.voice_call_service.asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(_create_twilio_call_with_retry(call_fn, to_number="+1"))

        assert result.sid == "CAsuccess"
        assert call_fn.call_count == 3

    def test_does_not_retry_permanent_4xx_error(self):
        from twilio.base.exceptions import TwilioRestException

        from app.services.voice_call_service import _create_twilio_call_with_retry

        call_fn = MagicMock(
            side_effect=TwilioRestException(
                status=400, uri="/Calls", msg="invalid number", code=21211
            )
        )

        with patch("app.services.voice_call_service.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            with pytest.raises(TwilioRestException):
                asyncio.run(_create_twilio_call_with_retry(call_fn, to_number="+1"))

        assert call_fn.call_count == 1
        mock_sleep.assert_not_called()

    def test_does_not_retry_5xx_error(self):
        """Regression guard for the double-dial risk: a 5xx must fail on the
        first attempt, never retried, since the call may have already been
        dialed server-side before Twilio returned the error."""
        from twilio.base.exceptions import TwilioRestException

        from app.services.voice_call_service import _create_twilio_call_with_retry

        call_fn = MagicMock(
            side_effect=TwilioRestException(status=500, uri="/Calls", msg="server error")
        )

        with patch("app.services.voice_call_service.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            with pytest.raises(TwilioRestException):
                asyncio.run(_create_twilio_call_with_retry(call_fn, to_number="+1"))

        assert call_fn.call_count == 1
        mock_sleep.assert_not_called()

    def test_exhausts_429_retries_and_raises(self):
        from twilio.base.exceptions import TwilioRestException

        from app.services.voice_call_service import (
            _TWILIO_CALL_MAX_RETRIES,
            _create_twilio_call_with_retry,
        )

        call_fn = MagicMock(
            side_effect=TwilioRestException(status=429, uri="/Calls", msg="rate limited")
        )

        with patch("app.services.voice_call_service.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(TwilioRestException):
                asyncio.run(_create_twilio_call_with_retry(call_fn, to_number="+1"))

        assert call_fn.call_count == _TWILIO_CALL_MAX_RETRIES + 1
