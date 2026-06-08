"""
LiveKit Recording Service — room-composite egress for call audio capture.

Starts a mix-minus audio-only egress on the LiveKit room when
recording_enabled=True.  The egress writes an Opus file directly to GCS.

After the call ends, call_recording_upload_service polls the egress status,
confirms completion, and updates the DB record.

Room naming matches livekit_service.py: room_{call_session_id}
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings
from app.core.logger import logger


class LiveKitRecordingService:

    def _get_credentials(self):
        from app.core.secret_manager import get_livekit_credentials

        return get_livekit_credentials()

    def _gcs_credentials_json(self) -> Optional[str]:
        """
        Load the GCS service account JSON string for LiveKit's GCPUpload.

        Reads GOOGLE_APPLICATION_CREDENTIALS (file path) and returns the file
        contents as a JSON string.  Returns None if credentials are unavailable
        (LiveKit will fall back to instance metadata in GKE).
        """
        if not settings.GOOGLE_APPLICATION_CREDENTIALS:
            return None
        try:
            with open(settings.GOOGLE_APPLICATION_CREDENTIALS, "r") as fh:
                return fh.read()
        except Exception as exc:
            logger.warning("Could not read GCS credentials file for LiveKit egress: %s", exc)
            return None

    async def start_room_recording(
        self,
        call_id: uuid.UUID,
        workspace_id: uuid.UUID,
        gcs_path: str,
    ) -> Optional[str]:
        """
        Start a room-composite audio-only egress that uploads to GCS.

        Returns the LiveKit egress_id, or None on failure (recording_enabled
        will remain True but egress won't run — caller logs and continues).

        The room is expected to exist (created in voice_call_service /
        handle_start_message before this is called).
        """
        from livekit import api

        room_name = f"room_{call_id}"
        url, api_key, api_secret = self._get_credentials()

        gcs_creds = self._gcs_credentials_json()

        gcp_upload = api.GCPUpload(
            bucket=settings.GCS_RECORDINGS_BUCKET,
        )
        if gcs_creds:
            gcp_upload = api.GCPUpload(
                bucket=settings.GCS_RECORDINGS_BUCKET,
                credentials=gcs_creds,
            )

        file_output = api.EncodedFileOutput(
            file_type=api.EncodedFileType.OGG,
            filepath=gcs_path,
            gcp=gcp_upload,
        )

        egress_request = api.RoomCompositeEgressRequest(
            room_name=room_name,
            audio_only=True,
            file=file_output,
        )

        try:
            async with api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret) as lk:
                info = await lk.egress.start_room_composite_egress(
                    api.StartEgressRequest(room_composite=egress_request)
                )
            egress_id = info.egress_id
            logger.info(
                "LiveKit recording egress started: egress_id=%s room=%s gcs=%s",
                egress_id,
                room_name,
                gcs_path,
            )
            return egress_id
        except Exception as exc:
            logger.error(
                "LiveKit egress start failed for room %s: %s",
                room_name,
                exc,
                exc_info=True,
            )
            return None

    async def stop_room_recording(self, egress_id: str) -> bool:
        """
        Stop a running egress and trigger upload completion.

        Returns True on success, False on failure.  Called on call end —
        failure is non-fatal (LiveKit may already be stopping the egress).
        """
        from livekit import api

        url, api_key, api_secret = self._get_credentials()
        try:
            async with api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret) as lk:
                await lk.egress.stop_egress(api.StopEgressRequest(egress_id=egress_id))
            logger.info("LiveKit egress stopped: %s", egress_id)
            return True
        except Exception as exc:
            logger.warning("LiveKit egress stop failed for %s: %s", egress_id, exc)
            return False

    async def get_egress_info(self, egress_id: str) -> Optional[object]:
        """
        Return the EgressInfo proto for the given egress_id, or None on error.
        Used by upload_service to check completion status.
        """
        from livekit import api

        url, api_key, api_secret = self._get_credentials()
        try:
            async with api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret) as lk:
                resp = await lk.egress.list_egress(
                    api.ListEgressRequest(egress_id=egress_id)
                )
            items = list(resp.items)
            return items[0] if items else None
        except Exception as exc:
            logger.warning("LiveKit get_egress_info failed for %s: %s", egress_id, exc)
            return None


livekit_recording_service = LiveKitRecordingService()
