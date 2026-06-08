"""
LiveKit room management service.

Manages real-time audio rooms for the voice agent platform.
Rooms are created BEFORE Twilio call initiation so the agent
is already listening when the caller connects.

Room naming: room_{call_session.id}  (deterministic, reusable)
Max participants: 2 (agent + caller) — 3rd join rejected at SDK level.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import settings
from app.core.logger import logger

def _http_to_ws_url(http_url: str) -> str:
    """Convert http(s):// to ws(s):// for LiveKit RTC WebSocket connections."""
    if http_url.startswith("https://"):
        return "wss://" + http_url[8:]
    if http_url.startswith("http://"):
        return "ws://" + http_url[7:]
    return http_url


_ROOM_NAME_RE = re.compile(
    r"^room_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class RoomService:
    """
    LiveKit room management for multi-tenant voice agent calls.

    All credential access is deferred to method call time so that
    LIVEKIT_ENABLED=False in local dev never causes import-time errors.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_room_name(self, room_name: str) -> None:
        if not _ROOM_NAME_RE.match(room_name):
            raise ValueError(
                f"Invalid room name '{room_name}'. "
                "Expected format: room_<uuid> (e.g. room_550e8400-e29b-41d4-a716-446655440000)"
            )

    def _get_credentials(self) -> tuple[str, str, str]:
        from app.core.secret_manager import get_livekit_credentials

        return get_livekit_credentials()

    # ------------------------------------------------------------------
    # Room lifecycle
    # ------------------------------------------------------------------

    async def create_room(
        self,
        call_id: uuid.UUID,
        agent_id: uuid.UUID,
        flow_id: uuid.UUID | None = None,
    ) -> Any:
        """
        Create (or return existing) LiveKit room for a call session.

        Idempotent: LiveKit's CreateRoomRequest returns the existing room
        if one with the same name already exists, so retries are safe.

        max_participants=2 is set at the SDK request level — a 3rd participant
        attempting to join is rejected by the LiveKit server.
        """
        from livekit import api

        room_name = f"room_{call_id}"
        url, api_key, api_secret = self._get_credentials()
        metadata = json.dumps(
            {
                "callId": str(call_id),
                "flowId": str(flow_id) if flow_id else None,
                "agentId": str(agent_id),
                "startedAt": datetime.now(timezone.utc).isoformat(),
            }
        )

        async with api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret) as lk:
            room = await lk.room.create_room(
                api.CreateRoomRequest(
                    name=room_name,
                    max_participants=settings.LIVEKIT_MAX_PARTICIPANTS,
                    empty_timeout=settings.LIVEKIT_ROOM_EMPTY_TIMEOUT,
                    metadata=metadata,
                )
            )

        logger.info(
            "LiveKit room ready: name=%s sid=%s max_participants=%d",
            room_name,
            room.sid,
            settings.LIVEKIT_MAX_PARTICIPANTS,
        )
        return room

    async def close_room(self, call_id: uuid.UUID) -> None:
        """
        Delete a LiveKit room.  Safe to call if the room no longer exists.
        Logs a warning on failure rather than raising so post-call cleanup
        never blocks response paths.
        """
        from livekit import api

        room_name = f"room_{call_id}"
        url, api_key, api_secret = self._get_credentials()
        try:
            async with api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret) as lk:
                await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
            logger.info("LiveKit room closed: %s", room_name)
        except Exception as exc:
            logger.warning("LiveKit close_room failed for %s: %s", room_name, exc)

    async def list_participants(self, room_name: str) -> list[dict[str, Any]]:
        """Return [{identity, sid, state}] for every participant in the room."""
        from livekit import api

        self._validate_room_name(room_name)
        url, api_key, api_secret = self._get_credentials()

        async with api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret) as lk:
            resp = await lk.room.list_participants(
                api.ListParticipantsRequest(room=room_name)
            )

        return [
            {"identity": p.identity, "sid": p.sid, "state": p.state}
            for p in resp.participants
        ]

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    def generate_agent_token(self, room_name: str) -> str:
        """
        Return a signed JWT granting agent-{room_name} join access to the room.
        TTL = LIVEKIT_TOKEN_TTL seconds (default 3600).
        Never logged — callers must treat the return value as a secret.
        """
        from livekit.api import AccessToken, VideoGrants

        self._validate_room_name(room_name)
        _, api_key, api_secret = self._get_credentials()
        identity = f"agent-{room_name}"
        token = (
            AccessToken(api_key=api_key, api_secret=api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .with_ttl(timedelta(seconds=settings.LIVEKIT_TOKEN_TTL))
            .to_jwt()
        )
        return token

    def generate_caller_token(self, room_name: str) -> str:
        """
        Return a signed JWT granting caller-{room_name} join access to the room.
        TTL = LIVEKIT_TOKEN_TTL seconds (default 3600).
        Never logged — callers must treat the return value as a secret.
        """
        from livekit.api import AccessToken, VideoGrants

        self._validate_room_name(room_name)
        _, api_key, api_secret = self._get_credentials()
        identity = f"caller-{room_name}"
        token = (
            AccessToken(api_key=api_key, api_secret=api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .with_ttl(timedelta(seconds=settings.LIVEKIT_TOKEN_TTL))
            .to_jwt()
        )
        return token

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    async def health_check(self) -> str:
        """
        Return "ok" or "degraded".  Never raises — safe to call from /health.

        Returns "ok" immediately when LIVEKIT_ENABLED=False.
        """
        if not settings.LIVEKIT_ENABLED:
            return "ok"
        try:
            reachable = await self.check_connectivity()
            return "ok" if reachable else "degraded"
        except Exception:
            return "degraded"

    async def check_connectivity(self) -> bool:
        """
        Verify the API server can reach the LiveKit server via HTTP/gRPC.

        Makes a lightweight list_rooms call — no rooms need to exist.
        Used for staging connectivity confirmation and /health probing.
        """
        from livekit import api

        try:
            url, api_key, api_secret = self._get_credentials()
        except (ValueError, RuntimeError) as exc:
            logger.warning("LiveKit credentials not available for connectivity check: %s", exc)
            return False

        try:
            async with api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret) as lk:
                await lk.room.list_rooms(api.ListRoomsRequest())
            return True
        except Exception as exc:
            logger.warning("LiveKit connectivity check failed: %s", exc)
            return False

    async def verify_rtc_connectivity(
        self, room_name: str | None = None
    ) -> bool:
        """
        Lightweight RTC WebSocket connectivity probe.

        Creates an ephemeral room (unless room_name is given), connects as a
        short-lived probe participant via the RTC SDK, then disconnects and
        cleans up. Returns True on success, False on any failure. Never raises.

        This is intentionally NOT called from health_check (too slow for
        every /health request). Use it from integration tests or manual probes.
        """
        from livekit import api, rtc

        probe_call_id = uuid.uuid4()
        probe_room = room_name or f"room_{probe_call_id}"
        owns_room = room_name is None

        try:
            url, api_key, api_secret = self._get_credentials()
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "LiveKit credentials unavailable for RTC probe: %s", exc
            )
            return False

        try:
            if owns_room:
                async with api.LiveKitAPI(
                    url=url, api_key=api_key, api_secret=api_secret
                ) as lk:
                    await lk.room.create_room(
                        api.CreateRoomRequest(
                            name=probe_room,
                            max_participants=1,
                            empty_timeout=10,
                        )
                    )

            from livekit.api import AccessToken, VideoGrants

            probe_token = (
                AccessToken(api_key=api_key, api_secret=api_secret)
                .with_identity(f"probe-{probe_call_id}")
                .with_grants(VideoGrants(room_join=True, room=probe_room))
                .with_ttl(timedelta(seconds=60))
                .to_jwt()
            )

            ws_url = _http_to_ws_url(url)
            rtc_room = rtc.Room()
            try:
                await rtc_room.connect(ws_url, probe_token)
            finally:
                await rtc_room.disconnect()

            logger.info("LiveKit RTC connectivity probe succeeded: %s", ws_url)
            return True

        except Exception as exc:
            logger.warning("LiveKit RTC connectivity probe failed: %s", exc)
            return False
        finally:
            if owns_room:
                try:
                    await self.close_room(probe_call_id)
                except Exception:
                    pass


livekit_service = RoomService()
