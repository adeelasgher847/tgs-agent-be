"""
Integration tests for LiveKit — require a live server.

Run only when RUN_LIVEKIT_INTEGRATION=1 and LIVEKIT_URL, LIVEKIT_API_KEY,
LIVEKIT_API_SECRET are set. The explicit opt-in flag (mirroring
RUN_GOOGLE_STT_INTEGRATION in test_google_stt_live.py) is required in
addition to the credential vars because those vars are also present in the
shared .env file and get loaded into os.environ as a side effect of
collecting unrelated test modules (e.g. test_google_stt_live.py calls
load_dotenv() at import time) — without the flag this module would
silently attempt a real connection whenever it happens to be collected
alongside such a module, instead of skipping as intended.

These tests confirm both HTTP/API-plane and WebSocket/RTC connectivity from
the API server to the LiveKit server, covering the full room lifecycle:

  create_room → generate_agent_token → generate_caller_token
  → RTC participant join → list_participants → disconnect
  → close_room → verify closed

LOCAL DEV — start a throwaway LiveKit server with Docker:

    docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \\
        -e LIVEKIT_KEYS="devkey: secret" livekit/livekit-server --dev

    Then run:

    RUN_LIVEKIT_INTEGRATION=1 \\
    LIVEKIT_URL=http://localhost:7880 \\
    LIVEKIT_API_KEY=devkey \\
    LIVEKIT_API_SECRET=secret \\
    pytest tests/integration/test_livekit_integration.py -v -m integration
"""

from __future__ import annotations

import os
import uuid

import pytest


# ---------------------------------------------------------------------------
# Skip entire module unless explicitly opted in with LiveKit env vars set
# ---------------------------------------------------------------------------

_RUN_LIVEKIT_INTEGRATION = os.environ.get("RUN_LIVEKIT_INTEGRATION", "").lower() in (
    "1",
    "true",
    "yes",
)
_LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
_LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
_LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

pytestmark = pytest.mark.integration

_skip_unless_livekit = pytest.mark.skipif(
    not (_RUN_LIVEKIT_INTEGRATION and _LIVEKIT_URL and _LIVEKIT_API_KEY and _LIVEKIT_API_SECRET),
    reason=(
        "Set RUN_LIVEKIT_INTEGRATION=1 and LIVEKIT_URL, LIVEKIT_API_KEY, "
        "LIVEKIT_API_SECRET to run live LiveKit integration tests"
    ),
)

# Skip RTC tests if the native livekit package cannot be imported
# (e.g. no platform wheel available for this Python version).
try:
    import livekit.rtc as _rtc  # noqa: F401

    _HAS_RTC = True
except Exception:
    _HAS_RTC = False

_skip_unless_rtc = pytest.mark.skipif(
    not _HAS_RTC,
    reason="livekit RTC package unavailable — install livekit>=1.1.2",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service():
    """Return a RoomService wired to the real LiveKit credentials from env."""
    from app.services.livekit_service import RoomService

    svc = RoomService()
    _real_creds = (_LIVEKIT_URL, _LIVEKIT_API_KEY, _LIVEKIT_API_SECRET)
    svc._get_credentials = lambda: _real_creds  # type: ignore[method-assign]
    return svc


def _livekit_ws_url(http_url: str) -> str:
    """Convert http(s):// → ws(s):// for LiveKit RTC connections."""
    from app.services.livekit_service import _http_to_ws_url

    return _http_to_ws_url(http_url)


# ---------------------------------------------------------------------------
# Full lifecycle integration test (with RTC participant join)
# ---------------------------------------------------------------------------


@_skip_unless_livekit
@_skip_unless_rtc
@pytest.mark.asyncio
async def test_full_room_lifecycle():
    """
    Create → token → RTC participant join → list participants → disconnect
    → close → verify closed.

    Steps:
    1.  create_room(call_id, agent_id, flow_id)
    2.  generate_agent_token(room_name)   — verify JWT structure
    3.  generate_caller_token(room_name)  — verify JWT structure, distinct from agent token
    4.  Connect as agent participant using livekit RTC SDK
    5.  list_participants(room_name) — must contain the joined agent identity
    6.  Disconnect participant cleanly
    7.  close_room(call_id)
    8.  Verify API remains reachable after deletion
    """
    from livekit import rtc

    svc = _make_service()
    call_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    flow_id = uuid.uuid4()
    room_name = f"room_{call_id}"

    try:
        # Step 1 — create room (with flow_id so metadata is exercised)
        room = await svc.create_room(call_id, agent_id, flow_id=flow_id)
        assert room.sid, "Room SID must be set after creation"

        # Step 2 — agent token
        agent_token = svc.generate_agent_token(room_name)
        assert isinstance(agent_token, str)
        assert len(agent_token.split(".")) == 3, "JWT must be header.payload.signature"

        # Step 3 — caller token (distinct from agent token)
        caller_token = svc.generate_caller_token(room_name)
        assert isinstance(caller_token, str)
        assert len(caller_token.split(".")) == 3
        assert agent_token != caller_token

        # Step 4 — connect as agent via RTC
        ws_url = _livekit_ws_url(_LIVEKIT_URL)
        rtc_room = rtc.Room()
        try:
            await rtc_room.connect(ws_url, agent_token)

            # Step 5 — list participants; the connected agent must be visible
            participants = await svc.list_participants(room_name)
            identities = [p["identity"] for p in participants]
            assert f"agent-{room_name}" in identities, (
                f"agent identity missing from participants list: {identities}"
            )

        finally:
            # Step 6 — disconnect cleanly (always runs)
            await rtc_room.disconnect()

    finally:
        # Step 7 — close room (always runs)
        await svc.close_room(call_id)

    # Step 8 — API must remain reachable
    assert await svc.check_connectivity(), "LiveKit API must remain reachable after room deletion"


# ---------------------------------------------------------------------------
# Idempotent create
# ---------------------------------------------------------------------------


@_skip_unless_livekit
@pytest.mark.asyncio
async def test_create_room_idempotent():
    """Calling create_room twice with the same call_id returns the same room."""
    svc = _make_service()
    call_id = uuid.uuid4()
    agent_id = uuid.uuid4()

    try:
        room1 = await svc.create_room(call_id, agent_id)
        room2 = await svc.create_room(call_id, agent_id)
        assert room1.sid == room2.sid, "Idempotent: same room SID on retry"
    finally:
        await svc.close_room(call_id)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


@_skip_unless_livekit
@pytest.mark.asyncio
async def test_close_nonexistent_room_does_not_raise():
    """close_room on a room that doesn't exist must not raise."""
    svc = _make_service()
    await svc.close_room(uuid.uuid4())


# ---------------------------------------------------------------------------
# HTTP API connectivity
# ---------------------------------------------------------------------------


@_skip_unless_livekit
@pytest.mark.asyncio
async def test_check_connectivity_returns_true():
    """API plane: HTTP/gRPC connectivity to LiveKit must succeed."""
    svc = _make_service()
    assert await svc.check_connectivity() is True, "LiveKit must be reachable"


# ---------------------------------------------------------------------------
# RTC WebSocket connectivity (via verify_rtc_connectivity)
# ---------------------------------------------------------------------------


@_skip_unless_livekit
@_skip_unless_rtc
@pytest.mark.asyncio
async def test_rtc_connectivity_returns_true():
    """WebSocket plane: verify_rtc_connectivity() must return True when server is up."""
    svc = _make_service()
    result = await svc.verify_rtc_connectivity()
    assert result is True, "LiveKit RTC WebSocket connectivity must succeed"


# ---------------------------------------------------------------------------
# Token payload verification
# ---------------------------------------------------------------------------


@_skip_unless_livekit
@pytest.mark.asyncio
async def test_token_identity_embedded_in_jwt():
    """Agent and caller identities must be recoverable from the JWT payload."""
    import base64

    svc = _make_service()
    call_id = uuid.uuid4()
    room_name = f"room_{call_id}"

    try:
        await svc.create_room(call_id, uuid.uuid4())

        agent_token = svc.generate_agent_token(room_name)
        caller_token = svc.generate_caller_token(room_name)

        def _decode_payload(token: str) -> str:
            payload_b64 = token.split(".")[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            return base64.urlsafe_b64decode(payload_b64).decode()

        assert f"agent-{room_name}" in _decode_payload(agent_token)
        assert f"caller-{room_name}" in _decode_payload(caller_token)
    finally:
        await svc.close_room(call_id)


# ---------------------------------------------------------------------------
# flow_id in room metadata
# ---------------------------------------------------------------------------


@_skip_unless_livekit
@pytest.mark.asyncio
async def test_flow_id_in_room_metadata():
    """flowId passed to create_room must appear in the LiveKit room metadata."""
    import json

    svc = _make_service()
    call_id = uuid.uuid4()
    flow_id = uuid.uuid4()

    try:
        room = await svc.create_room(call_id, uuid.uuid4(), flow_id=flow_id)
        metadata = json.loads(room.metadata)
        assert metadata["flowId"] == str(flow_id), (
            f"Expected flowId={flow_id!s}, got {metadata.get('flowId')!r}"
        )
    finally:
        await svc.close_room(call_id)
