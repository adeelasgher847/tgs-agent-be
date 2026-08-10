"""
Regression tests for LiveKitRecordingService.start_room_recording().

Covers a bug where start_room_recording() wrapped the already-built
RoomCompositeEgressRequest in a nonexistent api.StartEgressRequest(
room_composite=...) wrapper. Against the pinned livekit-api==1.1.0 /
livekit-protocol==1.1.17, StartEgressRequest has no room_composite field,
so every call raised ValueError before any network call, was swallowed by
the function's own broad except-block, and silently returned None —
meaning recording never started for either the Twilio or LiveKit-browser
transport.

These tests deliberately use the *real* installed livekit.api protobuf
classes to construct the expected request (not a mock), since a mocked
request object would not have caught this bug. Only the network/RPC
boundary (LiveKitAPI client + egress.start_room_composite_egress) is
mocked.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.livekit_recording_service import livekit_recording_service


def _make_lk_api_mock(start_room_composite_egress_mock: AsyncMock) -> MagicMock:
    """Build a MagicMock standing in for api.LiveKitAPI(...) used as an
    `async with` context manager, exposing .egress.start_room_composite_egress.
    """
    lk_instance = MagicMock()
    lk_instance.egress.start_room_composite_egress = start_room_composite_egress_mock
    lk_instance.__aenter__ = AsyncMock(return_value=lk_instance)
    lk_instance.__aexit__ = AsyncMock(return_value=False)

    lk_api_cls = MagicMock(return_value=lk_instance)
    return lk_api_cls


@pytest.mark.asyncio
async def test_start_room_recording_calls_egress_with_unwrapped_request():
    """The fix: start_room_composite_egress() must be called with the
    RoomCompositeEgressRequest object directly, not wrapped in a
    StartEgressRequest(room_composite=...). If this regresses back to the
    old wrapper, this test fails because the mock is never called with a
    RoomCompositeEgressRequest instance (it would either blow up
    constructing StartEgressRequest(room_composite=...), since that kwarg
    doesn't exist on the real protobuf class, or -- if some future SDK
    happens to add that kwarg back -- the call_args type check below would
    catch a StartEgressRequest instead of a RoomCompositeEgressRequest).
    """
    from livekit import api

    call_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    gcs_path = "recordings/some-call.ogg"

    fake_egress_info = MagicMock()
    fake_egress_info.egress_id = "EG_abc123"
    start_egress_mock = AsyncMock(return_value=fake_egress_info)
    lk_api_cls = _make_lk_api_mock(start_egress_mock)

    with (
        patch.object(
            livekit_recording_service,
            "_get_credentials",
            return_value=("wss://fake.livekit", "fake-key", "fake-secret"),
        ),
        patch("livekit.api.LiveKitAPI", lk_api_cls),
    ):
        egress_id = await livekit_recording_service.start_room_recording(
            call_id=call_id,
            workspace_id=workspace_id,
            gcs_path=gcs_path,
        )

    assert egress_id == "EG_abc123"
    start_egress_mock.assert_awaited_once()

    (called_arg,) = start_egress_mock.call_args.args
    # Proves the fix: the call receives the RoomCompositeEgressRequest
    # directly, not wrapped in StartEgressRequest.
    assert isinstance(called_arg, api.RoomCompositeEgressRequest)
    assert not isinstance(called_arg, api.StartEgressRequest)
    assert called_arg.room_name == f"room_{call_id}"
    assert called_arg.audio_only is True
    assert called_arg.file.filepath == gcs_path


@pytest.mark.asyncio
async def test_start_room_recording_returns_none_and_logs_on_rpc_failure():
    """Existing error-handling behavior must be preserved: if the egress
    RPC call itself raises, start_room_recording() catches it, logs an
    error, and returns None — not a new/different error path.
    """
    call_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    gcs_path = "recordings/some-call.ogg"

    start_egress_mock = AsyncMock(side_effect=RuntimeError("egress RPC unreachable"))
    lk_api_cls = _make_lk_api_mock(start_egress_mock)

    with (
        patch.object(
            livekit_recording_service,
            "_get_credentials",
            return_value=("wss://fake.livekit", "fake-key", "fake-secret"),
        ),
        patch("livekit.api.LiveKitAPI", lk_api_cls),
        patch("app.services.livekit_recording_service.logger") as mock_logger,
    ):
        egress_id = await livekit_recording_service.start_room_recording(
            call_id=call_id,
            workspace_id=workspace_id,
            gcs_path=gcs_path,
        )

    assert egress_id is None
    start_egress_mock.assert_awaited_once()
    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_start_room_recording_would_have_failed_with_old_wrapper():
    """Documents the exact previous failure mode: constructing
    api.StartEgressRequest(room_composite=<RoomCompositeEgressRequest>) on
    the currently-pinned SDK raises before any network call is made, since
    StartEgressRequest has no room_composite field. This is what let the
    bug silently swallow every recording attempt via the function's broad
    except-block.
    """
    from livekit import api

    egress_request = api.RoomCompositeEgressRequest(
        room_name="room_test",
        audio_only=True,
    )

    with pytest.raises(ValueError):
        api.StartEgressRequest(room_composite=egress_request)
