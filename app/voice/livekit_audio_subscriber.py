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

    async def _subscribe_and_transcode(self, ws_url: str, token: str) -> None:
        try:
            from livekit import rtc
        except ImportError:
            logger.error("[LiveKitAudioSubscriber] livekit package not available")
            return

        room = rtc.Room()
        audio_stream: Any | None = None
        caller_track_found = asyncio.Event()

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

        room.on("track_subscribed", on_track_subscribed)

        try:
            await room.connect(ws_url, token)
            logger.info(
                "[LiveKitAudioSubscriber] connected to room=%s", self._room_name
            )

            try:
                await asyncio.wait_for(caller_track_found.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "[LiveKitAudioSubscriber] timed out waiting for caller audio track"
                )
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
