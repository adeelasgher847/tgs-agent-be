"""
LiveKitAudioSubscriber — subscribe to caller audio track and feed SttPipeline.

Call flow (Google STT path):
  LiveKit room → caller audio track (PCM via rtc.AudioStream)
    → LiveKitAudioProcessor (resample to LINEAR16 16 kHz if needed)
      → SttPipeline.feed_audio_chunk(pcm_bytes)

The subscriber runs as a background asyncio Task created by VoiceOrchestrator
when agent.stt_provider_slug == "google" and LIVEKIT_ENABLED=True.
Twilio MULAW path is unaffected.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import settings
from app.core.logger import logger
from app.voice.audio_transcoder import LiveKitAudioProcessor


class LiveKitAudioSubscriber:
    """
    Subscribes to the caller participant's audio track in a LiveKit room,
    normalises frames to LINEAR16 16 kHz mono, and feeds SttPipeline.
    """

    def __init__(
        self,
        room_name: str,
        stt_pipeline: Any,
        output_sample_rate: int = 16000,
    ) -> None:
        self._room_name = room_name
        self._stt_pipeline = stt_pipeline
        self._output_sample_rate = output_sample_rate
        self._stop_event = asyncio.Event()
        self._processor: LiveKitAudioProcessor | None = None
        self._vad_note_logged = False
        # Rate-limits the "still waiting for caller audio track" warning to
        # once (the first 30s timeout) rather than once per retry, so a
        # sustained wait (slow mic permission grant, ICE/TURN negotiation on
        # a restrictive network, etc.) doesn't flood the logs while still
        # being fully visible the first time it happens.
        self._caller_track_wait_timeout_logged = False

    async def run(self) -> None:
        """Connect to LiveKit room, subscribe to caller audio, feed STT."""
        if not settings.LIVEKIT_ENABLED:
            logger.info("[LiveKitAudioSubscriber] LIVEKIT_ENABLED=false — no-op")
            return

        from app.services.livekit_service import livekit_service

        try:
            url, _, _ = livekit_service._get_credentials()
        except Exception as exc:
            logger.error("[LiveKitAudioSubscriber] credentials error: %s", exc)
            return

        from app.services.livekit_service import _http_to_ws_url

        ws_url = _http_to_ws_url(url)
        agent_token = livekit_service.generate_agent_token(self._room_name)

        async with LiveKitAudioProcessor(
            output_sample_rate=self._output_sample_rate
        ) as processor:
            self._processor = processor
            await self._subscribe_and_transcode(ws_url, agent_token)
        self._processor = None

    async def _wait_for_caller_track(
        self,
        caller_track_found: asyncio.Event,
        room_disconnected: asyncio.Event,
    ) -> bool:
        """
        Wait for the caller's audio track to be subscribed, for the effective
        lifetime of the call — not just a single 30s window.

        A live web-demo call can run for minutes (bounded only by the caller
        hanging up or an explicit end-call), and the browser side may take
        longer than 30s to actually publish its mic track on a first-time
        visit (mic permission prompt, restrictive-network ICE/TURN
        negotiation, etc.). Giving up permanently after one timeout silently
        kills STT ingestion for the rest of the call. Instead, loop the wait
        indefinitely, exiting only when the track is found, the caller
        explicitly requests stop (``self._stop_event``), or the room itself
        disconnects.

        Returns True if the caller track was found, False otherwise (in
        which case the caller should not proceed to consume audio_stream).
        """
        stop_task = asyncio.ensure_future(self._stop_event.wait())
        disconnect_task = asyncio.ensure_future(room_disconnected.wait())
        track_task = asyncio.ensure_future(caller_track_found.wait())

        try:
            while True:
                done, _pending = await asyncio.wait(
                    {stop_task, disconnect_task, track_task},
                    timeout=30.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if track_task in done:
                    return True

                if stop_task in done:
                    logger.info(
                        "[LiveKitAudioSubscriber] stop requested while waiting "
                        "for caller audio track room=%s", self._room_name,
                    )
                    return False

                if disconnect_task in done:
                    logger.warning(
                        "[LiveKitAudioSubscriber] room disconnected while "
                        "waiting for caller audio track room=%s", self._room_name,
                    )
                    return False

                # 30s elapsed with none of the above — keep waiting, but
                # don't spam the logs on every subsequent retry.
                if not self._caller_track_wait_timeout_logged:
                    logger.warning(
                        "[LiveKitAudioSubscriber] still waiting for caller "
                        "audio track after 30s (room=%s) — continuing to "
                        "wait for the rest of the call instead of "
                        "disconnecting; further waits logged at debug level",
                        self._room_name,
                    )
                    self._caller_track_wait_timeout_logged = True
                else:
                    logger.debug(
                        "[LiveKitAudioSubscriber] still waiting for caller "
                        "audio track (room=%s)", self._room_name,
                    )
        finally:
            for task in (stop_task, disconnect_task, track_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                stop_task, disconnect_task, track_task, return_exceptions=True
            )

    async def _subscribe_and_transcode(self, ws_url: str, token: str) -> None:
        try:
            from livekit import rtc
        except ImportError:
            logger.error("[LiveKitAudioSubscriber] livekit package not available")
            return

        room = rtc.Room()
        audio_stream: Any | None = None
        caller_track_found = asyncio.Event()
        room_disconnected = asyncio.Event()

        def on_track_subscribed(track, publication, participant):
            nonlocal audio_stream
            if (
                track.kind == rtc.TrackKind.KIND_AUDIO
                and "caller" in (participant.identity or "").lower()
            ):
                logger.info(
                    "[LiveKitAudioSubscriber] subscribed to caller audio track sid=%s",
                    track.sid,
                )
                audio_stream = rtc.AudioStream(track)
                caller_track_found.set()

        def on_room_disconnected(*_args):
            room_disconnected.set()

        room.on("track_subscribed", on_track_subscribed)
        room.on("disconnected", on_room_disconnected)

        try:
            await room.connect(ws_url, token)
            logger.info(
                "[LiveKitAudioSubscriber] connected to room=%s", self._room_name
            )

            if not await self._wait_for_caller_track(
                caller_track_found, room_disconnected
            ):
                return

            if audio_stream is None or self._processor is None:
                return

            if not self._vad_note_logged:
                # No local VAD/silence gate exists on this path — every decoded
                # frame is forwarded to STT (Deepgram/Google) unconditionally;
                # turn-taking ("speech started/stopped") is entirely delegated
                # to the STT provider's own server-side endpointing
                # (speech_final / UtteranceEnd for Deepgram, is_final for
                # Google). Logged once so a future "conversation never
                # starts" investigation doesn't waste time looking for an
                # app-side VAD gate that was never here.
                logger.debug(
                    "[LiveKitAudioSubscriber] VAD decision: no local VAD gate — "
                    "all frames forwarded, turn-taking is provider-side endpointing"
                )
                self._vad_note_logged = True

            async for audio_frame_event in audio_stream:
                if self._stop_event.is_set():
                    break

                frame = audio_frame_event.frame
                raw_bytes = bytes(frame.data)
                if not raw_bytes:
                    continue

                sample_rate = int(getattr(frame, "sample_rate", 48000) or 48000)
                num_channels = int(getattr(frame, "num_channels", 1) or 1)

                try:
                    pcm_bytes = await self._processor.process_frame(
                        raw_bytes,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                    )
                except Exception as exc:
                    # A single frame's processing failure must never abort the
                    # whole subscription loop — that would silently end STT
                    # ingestion for the rest of the call after only the frames
                    # seen so far. Drop this frame and keep listening.
                    logger.error(
                        "[LiveKitAudioSubscriber] frame processing error "
                        "(dropping this frame, continuing): %s", exc, exc_info=True,
                    )
                    continue

                if pcm_bytes:
                    logger.debug(
                        "[LiveKitAudioSubscriber] STT send: bytes=%s", len(pcm_bytes)
                    )
                    await self._stt_pipeline.feed_audio_chunk(pcm_bytes)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "[LiveKitAudioSubscriber] stream error: %s", exc, exc_info=True
            )
        finally:
            try:
                await room.disconnect()
            except Exception as exc:
                logger.debug("[LiveKitAudioSubscriber] room disconnect failed: %s", exc)
            logger.info(
                "[LiveKitAudioSubscriber] disconnected from room=%s", self._room_name
            )

    async def stop(self) -> None:
        """Signal the run loop to exit gracefully."""
        self._stop_event.set()
        if self._processor:
            await self._processor.close()
