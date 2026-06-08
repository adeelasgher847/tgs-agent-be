"""
Unit tests for app/services/livekit_service.py.

All LiveKit SDK calls are mocked — no real LiveKit server is required.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock the livekit package at sys.modules level before any import that
# touches livekit.  This prevents ImportError when livekit-api is not yet
# installed in the CI environment.
# ---------------------------------------------------------------------------

def _make_livekit_mock():
    lk_mod = MagicMock()
    api_mod = MagicMock()

    # LiveKitAPI async context manager
    lk_api_instance = MagicMock()
    lk_api_instance.__aenter__ = AsyncMock(return_value=lk_api_instance)
    lk_api_instance.__aexit__ = AsyncMock(return_value=False)
    lk_api_instance.room = MagicMock()

    lk_api_class = MagicMock(return_value=lk_api_instance)
    api_mod.LiveKitAPI = lk_api_class

    # Request objects — simple pass-through constructors
    api_mod.CreateRoomRequest = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    api_mod.DeleteRoomRequest = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    api_mod.ListParticipantsRequest = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    api_mod.ListRoomsRequest = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))

    lk_mod.api = api_mod
    return lk_mod, api_mod, lk_api_instance


_lk_mod, _api_mod, _lk_instance = _make_livekit_mock()
sys.modules.setdefault("livekit", _lk_mod)
sys.modules.setdefault("livekit.api", _api_mod)

# AccessToken / VideoGrants live in livekit.api
_api_mod.AccessToken = MagicMock()
_api_mod.VideoGrants = MagicMock()

# Now safe to import
from app.services.livekit_service import RoomService  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CALL_ID = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
VALID_AGENT_ID = uuid.UUID("660e8400-e29b-41d4-a716-446655440001")
VALID_ROOM_NAME = f"room_{VALID_CALL_ID}"

_CREDS_PATCH = "app.services.livekit_service.RoomService._get_credentials"
_CREDS = ("http://livekit:7880", "test-api-key", "test-api-secret")


def _fresh_service() -> RoomService:
    return RoomService()


# ---------------------------------------------------------------------------
# Room naming convention
# ---------------------------------------------------------------------------

class TestRoomNaming:
    def test_room_name_uses_call_id(self):
        name = f"room_{VALID_CALL_ID}"
        assert name == VALID_ROOM_NAME

    def test_room_name_regex_accepts_valid(self):
        from app.services.livekit_service import _ROOM_NAME_RE

        assert _ROOM_NAME_RE.match(f"room_{VALID_CALL_ID}")

    def test_room_name_regex_rejects_no_prefix(self):
        from app.services.livekit_service import _ROOM_NAME_RE

        assert not _ROOM_NAME_RE.match(str(VALID_CALL_ID))

    def test_room_name_regex_rejects_short_uuid(self):
        from app.services.livekit_service import _ROOM_NAME_RE

        assert not _ROOM_NAME_RE.match("room_abc123")

    def test_validate_room_name_raises_for_invalid(self):
        svc = _fresh_service()
        with pytest.raises(ValueError, match="Invalid room name"):
            svc._validate_room_name("room_not-a-uuid")

    def test_validate_room_name_passes_for_valid(self):
        svc = _fresh_service()
        svc._validate_room_name(VALID_ROOM_NAME)  # must not raise


# ---------------------------------------------------------------------------
# create_room — max_participants enforced at SDK level
# ---------------------------------------------------------------------------

class TestCreateRoom:
    @pytest.mark.asyncio
    async def test_creates_room_with_correct_name(self):
        svc = _fresh_service()
        mock_room = SimpleNamespace(sid="RM_test_sid", name=VALID_ROOM_NAME)
        _lk_instance.room.create_room = AsyncMock(return_value=mock_room)

        with patch(_CREDS_PATCH, return_value=_CREDS):
            room = await svc.create_room(VALID_CALL_ID, VALID_AGENT_ID)

        assert room.sid == "RM_test_sid"
        _api_mod.CreateRoomRequest.assert_called()
        call_kwargs = _api_mod.CreateRoomRequest.call_args.kwargs
        assert call_kwargs["name"] == VALID_ROOM_NAME

    @pytest.mark.asyncio
    async def test_max_participants_is_2(self):
        """SDK-level enforcement: 3rd participant is rejected by LiveKit server."""
        svc = _fresh_service()
        _lk_instance.room.create_room = AsyncMock(
            return_value=SimpleNamespace(sid="sid1", name=VALID_ROOM_NAME)
        )

        with patch(_CREDS_PATCH, return_value=_CREDS):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.LIVEKIT_MAX_PARTICIPANTS = 2
                mock_settings.LIVEKIT_ROOM_EMPTY_TIMEOUT = 30
                await svc.create_room(VALID_CALL_ID, VALID_AGENT_ID)

        call_kwargs = _api_mod.CreateRoomRequest.call_args.kwargs
        assert call_kwargs["max_participants"] == 2, (
            "max_participants MUST be 2 so LiveKit rejects any 3rd participant"
        )

    @pytest.mark.asyncio
    async def test_metadata_contains_required_fields(self):
        svc = _fresh_service()
        _lk_instance.room.create_room = AsyncMock(
            return_value=SimpleNamespace(sid="sid2", name=VALID_ROOM_NAME)
        )
        flow_id = uuid.uuid4()

        with patch(_CREDS_PATCH, return_value=_CREDS):
            await svc.create_room(VALID_CALL_ID, VALID_AGENT_ID, flow_id=flow_id)

        call_kwargs = _api_mod.CreateRoomRequest.call_args.kwargs
        metadata = json.loads(call_kwargs["metadata"])
        assert metadata["callId"] == str(VALID_CALL_ID)
        assert metadata["agentId"] == str(VALID_AGENT_ID)
        assert metadata["flowId"] == str(flow_id)
        assert "startedAt" in metadata
        # startedAt must parse as ISO-8601
        datetime.fromisoformat(metadata["startedAt"])

    @pytest.mark.asyncio
    async def test_metadata_flow_id_none_when_not_provided(self):
        svc = _fresh_service()
        _lk_instance.room.create_room = AsyncMock(
            return_value=SimpleNamespace(sid="sid3", name=VALID_ROOM_NAME)
        )

        with patch(_CREDS_PATCH, return_value=_CREDS):
            await svc.create_room(VALID_CALL_ID, VALID_AGENT_ID)

        call_kwargs = _api_mod.CreateRoomRequest.call_args.kwargs
        metadata = json.loads(call_kwargs["metadata"])
        assert metadata["flowId"] is None

    @pytest.mark.asyncio
    async def test_idempotent_retry_returns_existing_room(self):
        """create_room must be safe to call twice — LiveKit returns existing room."""
        svc = _fresh_service()
        existing_room = SimpleNamespace(sid="RM_existing", name=VALID_ROOM_NAME)
        _lk_instance.room.create_room = AsyncMock(return_value=existing_room)

        with patch(_CREDS_PATCH, return_value=_CREDS):
            room1 = await svc.create_room(VALID_CALL_ID, VALID_AGENT_ID)
            room2 = await svc.create_room(VALID_CALL_ID, VALID_AGENT_ID)

        assert room1.sid == room2.sid == "RM_existing"
        assert _lk_instance.room.create_room.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_timeout_set(self):
        svc = _fresh_service()
        _lk_instance.room.create_room = AsyncMock(
            return_value=SimpleNamespace(sid="sid4", name=VALID_ROOM_NAME)
        )

        with patch(_CREDS_PATCH, return_value=_CREDS):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.LIVEKIT_MAX_PARTICIPANTS = 2
                mock_settings.LIVEKIT_ROOM_EMPTY_TIMEOUT = 30
                await svc.create_room(VALID_CALL_ID, VALID_AGENT_ID)

        call_kwargs = _api_mod.CreateRoomRequest.call_args.kwargs
        assert call_kwargs["empty_timeout"] == 30


# ---------------------------------------------------------------------------
# close_room
# ---------------------------------------------------------------------------

class TestCloseRoom:
    @pytest.mark.asyncio
    async def test_close_room_success(self):
        svc = _fresh_service()
        _lk_instance.room.delete_room = AsyncMock()

        with patch(_CREDS_PATCH, return_value=_CREDS):
            await svc.close_room(VALID_CALL_ID)

        _api_mod.DeleteRoomRequest.assert_called()
        call_kwargs = _api_mod.DeleteRoomRequest.call_args.kwargs
        assert call_kwargs["room"] == VALID_ROOM_NAME

    @pytest.mark.asyncio
    async def test_close_room_not_found_does_not_raise(self):
        """Deleting a non-existent room must silently log a warning, not raise."""
        svc = _fresh_service()
        _lk_instance.room.delete_room = AsyncMock(
            side_effect=Exception("room not found")
        )

        with patch(_CREDS_PATCH, return_value=_CREDS):
            # Must not raise
            await svc.close_room(VALID_CALL_ID)


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------

class TestTokenGeneration:
    def _setup_access_token(self):
        """Return a mock AccessToken that produces a recognizable JWT string."""
        mock_token_instance = MagicMock()
        mock_token_instance.with_identity.return_value = mock_token_instance
        mock_token_instance.with_name.return_value = mock_token_instance
        mock_token_instance.with_grants.return_value = mock_token_instance
        mock_token_instance.with_ttl.return_value = mock_token_instance
        mock_token_instance.to_jwt.return_value = "header.payload.signature"
        _api_mod.AccessToken.return_value = mock_token_instance
        return mock_token_instance

    def test_generate_agent_token_returns_jwt_string(self):
        svc = _fresh_service()
        mock_tok = self._setup_access_token()

        with patch(_CREDS_PATCH, return_value=_CREDS):
            token = svc.generate_agent_token(VALID_ROOM_NAME)

        assert token == "header.payload.signature"
        mock_tok.to_jwt.assert_called_once()

    def test_generate_caller_token_returns_jwt_string(self):
        svc = _fresh_service()
        mock_tok = self._setup_access_token()

        with patch(_CREDS_PATCH, return_value=_CREDS):
            token = svc.generate_caller_token(VALID_ROOM_NAME)

        assert token == "header.payload.signature"
        mock_tok.to_jwt.assert_called_once()

    def test_agent_token_identity_prefix(self):
        svc = _fresh_service()
        mock_tok = self._setup_access_token()

        with patch(_CREDS_PATCH, return_value=_CREDS):
            svc.generate_agent_token(VALID_ROOM_NAME)

        mock_tok.with_identity.assert_called_with(f"agent-{VALID_ROOM_NAME}")

    def test_caller_token_identity_prefix(self):
        svc = _fresh_service()
        mock_tok = self._setup_access_token()

        with patch(_CREDS_PATCH, return_value=_CREDS):
            svc.generate_caller_token(VALID_ROOM_NAME)

        mock_tok.with_identity.assert_called_with(f"caller-{VALID_ROOM_NAME}")

    def test_token_ttl_matches_config(self):
        svc = _fresh_service()
        mock_tok = self._setup_access_token()

        with patch(_CREDS_PATCH, return_value=_CREDS):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.LIVEKIT_TOKEN_TTL = 3600
                svc.generate_agent_token(VALID_ROOM_NAME)

        from datetime import timedelta
        mock_tok.with_ttl.assert_called_with(timedelta(seconds=3600))

    def test_token_expiry_approx_one_hour(self):
        """Verify TTL config drives the timedelta passed to with_ttl."""
        svc = _fresh_service()
        mock_tok = self._setup_access_token()

        with patch(_CREDS_PATCH, return_value=_CREDS):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.LIVEKIT_TOKEN_TTL = 3600
                svc.generate_agent_token(VALID_ROOM_NAME)

        from datetime import timedelta
        call_arg = mock_tok.with_ttl.call_args[0][0]
        assert call_arg == timedelta(seconds=3600), (
            "exp - nbf should equal LIVEKIT_TOKEN_TTL (3600s = 1 hour)"
        )

    def test_generate_token_rejects_invalid_room_name(self):
        svc = _fresh_service()
        with patch(_CREDS_PATCH, return_value=_CREDS):
            with pytest.raises(ValueError, match="Invalid room name"):
                svc.generate_agent_token("bad-room-name")

    def test_generate_caller_token_rejects_invalid_room_name(self):
        svc = _fresh_service()
        with patch(_CREDS_PATCH, return_value=_CREDS):
            with pytest.raises(ValueError, match="Invalid room name"):
                svc.generate_caller_token("not_a_room_uuid")


# ---------------------------------------------------------------------------
# list_participants
# ---------------------------------------------------------------------------

class TestListParticipants:
    @pytest.mark.asyncio
    async def test_returns_correct_shape(self):
        svc = _fresh_service()
        p1 = SimpleNamespace(identity="agent-room_abc", sid="PA_1", state=1)
        p2 = SimpleNamespace(identity="caller-room_abc", sid="PA_2", state=1)
        mock_resp = SimpleNamespace(participants=[p1, p2])
        _lk_instance.room.list_participants = AsyncMock(return_value=mock_resp)

        with patch(_CREDS_PATCH, return_value=_CREDS):
            result = await svc.list_participants(VALID_ROOM_NAME)

        assert len(result) == 2
        assert result[0] == {"identity": "agent-room_abc", "sid": "PA_1", "state": 1}
        assert result[1] == {"identity": "caller-room_abc", "sid": "PA_2", "state": 1}

    @pytest.mark.asyncio
    async def test_rejects_invalid_room_name(self):
        svc = _fresh_service()
        with patch(_CREDS_PATCH, return_value=_CREDS):
            with pytest.raises(ValueError, match="Invalid room name"):
                await svc.list_participants("bad-name")

    @pytest.mark.asyncio
    async def test_empty_room_returns_empty_list(self):
        svc = _fresh_service()
        _lk_instance.room.list_participants = AsyncMock(
            return_value=SimpleNamespace(participants=[])
        )

        with patch(_CREDS_PATCH, return_value=_CREDS):
            result = await svc.list_participants(VALID_ROOM_NAME)

        assert result == []
