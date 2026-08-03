"""
Rime Labs TTS service — streaming HTTP synthesis for real-time calling.

API: POST https://users.rime.ai/v1/rime-tts  (streaming=true)
Audio: returns raw PCM/mulaw chunks streamed over HTTP.
Telephony output: mulaw 8 kHz (matches Twilio MULAW 8000).

Rime mistv2 supports:
  - modelId: "mistv2" (default)
  - speaker: voice ID (default "mistv2_Wildflower")
  - samplingRate: 8000
  - audioFormat: "mulaw"
  - speedAlpha: float (1.0 = normal, <1 slower, >1 faster)
  - reduceLatency: true  — trims inter-sentence silence
  - text: the input text string
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import httpx

from app.core.logger import logger
from app.core.secret_manager import get_rime_api_key

_RIME_TTS_URL = "https://users.rime.ai/v1/rime-tts"
_DEFAULT_MODEL = "mistv2"
_DEFAULT_VOICE = "mistv2_Wildflower"
_DEFAULT_SAMPLE_RATE = 8000
_DEFAULT_AUDIO_FORMAT = "mulaw"

# httpx streaming chunk size (bytes) — small to minimise first-chunk latency.
_STREAM_CHUNK_SIZE = 960  # 120ms of mulaw@8kHz


class RimeTtsService:
    """Thin async wrapper around the Rime TTS HTTP streaming endpoint."""

    def __init__(self) -> None:
        # httpx.AsyncClient's connection pool (httpcore/anyio) is bound to
        # whichever asyncio event loop was running when the client's
        # sockets were opened — it is NOT safe to reuse a single client
        # across different loops (a stale pooled connection tries to
        # call_soon() on its original loop when closed/reused, raising
        # "RuntimeError: Event loop is closed" if that loop already ended).
        #
        # This service is called from two distinct execution contexts that
        # each have their own long-lived loop for the life of the process:
        #   - the live Twilio/LiveKit streaming path (stream_text_to_speech
        #     awaited directly on the app's main event loop)
        #   - RimeTTSAdapter.synthesize()'s sync-bridge path, which now runs
        #     all its coroutines on one dedicated background loop (see
        #     app/utils/tts_adapter.py::_get_rime_sync_bridge_loop) instead
        #     of spinning up/tearing down a fresh loop per call.
        # Keying the cached client by the *running loop* means each of those
        # stable loops gets its own client, created once and reused for that
        # loop's whole lifetime — never shared across loops.
        self._clients: dict[int, httpx.AsyncClient] = {}
        # Resolve once at construction so a missing key fails before any live call.
        self._api_key = get_rime_api_key()

    def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        key = id(loop)
        client = self._clients.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=5.0),
                http2=False,
            )
            self._clients[key] = client
        return client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream_text_to_speech(
        self,
        text: str,
        speaker: str = _DEFAULT_VOICE,
        model_id: str = _DEFAULT_MODEL,
        speed_alpha: float = 1.0,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        audio_format: str = _DEFAULT_AUDIO_FORMAT,
        reduce_latency: bool = True,
    ) -> AsyncIterator[bytes]:
        """
        Async generator yielding raw mulaw audio byte chunks in real-time.

        First chunk arrives within ~100-200ms of the request being sent.
        Caller should pipe chunks directly to Twilio WebSocket frames.

        Args:
            text: Plain text to synthesise (no SSML).
            speaker: Rime voice ID (default mistv2_Wildflower).
            model_id: Rime model ID (default mistv2).
            speed_alpha: Playback speed multiplier (1.0 = normal).
            sample_rate: Output sample rate in Hz (8000 for Twilio).
            audio_format: Output encoding (mulaw for Twilio).
            reduce_latency: Trim silence for lower latency.
        """
        if not text or not text.strip():
            return

        payload = {
            "text": text.strip(),
            "modelId": model_id,
            "speaker": speaker,
            "samplingRate": sample_rate,
            "audioFormat": audio_format,
            "speedAlpha": float(speed_alpha),
            "reduceLatency": reduce_latency,
            "streaming": True,  # required for chunked HTTP streaming response
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # Streaming format is controlled by Accept (not JSON audioFormat).
            # audio/basic returns non-mulaw bytes → Twilio plays as garbled "cheeee" noise.
            "Accept": "audio/x-mulaw",
        }

        t0 = time.perf_counter()
        first_chunk = True

        client = self._get_client()
        try:
            async with client.stream("POST", _RIME_TTS_URL, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for raw_chunk in resp.aiter_bytes(_STREAM_CHUNK_SIZE):
                    if raw_chunk:
                        if first_chunk:
                            latency_ms = (time.perf_counter() - t0) * 1000
                            logger.info(
                                "[Rime] first audio chunk latency: %.0f ms (voice=%s)",
                                latency_ms, speaker,
                            )
                            first_chunk = False
                        yield raw_chunk
        except httpx.HTTPStatusError as exc:
            # Do not call .text on a streaming response (content not yet read).
            logger.error(
                "[Rime] HTTP error %d for voice=%s",
                exc.response.status_code, speaker,
            )
            raise
        except httpx.RequestError as exc:
            logger.error("[Rime] Request error: %s", exc)
            raise

    async def synthesize(
        self,
        text: str,
        speaker: str = _DEFAULT_VOICE,
        model_id: str = _DEFAULT_MODEL,
        speed_alpha: float = 1.0,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        audio_format: str = _DEFAULT_AUDIO_FORMAT,
    ) -> bytes:
        """Collect full audio into memory (used by batch/cache paths)."""
        chunks: list[bytes] = []
        async for chunk in self.stream_text_to_speech(
            text=text,
            speaker=speaker,
            model_id=model_id,
            speed_alpha=speed_alpha,
            sample_rate=sample_rate,
            audio_format=audio_format,
        ):
            chunks.append(chunk)
        return b"".join(chunks)

    async def close(self) -> None:
        """Close whichever client is bound to the *currently running* loop.

        Only the caller's own loop-scoped client can be safely closed from
        here — closing a client bound to a different (possibly already
        stopped) loop would itself risk the same "Event loop is closed"
        failure this file exists to avoid.
        """
        loop = asyncio.get_running_loop()
        client = self._clients.pop(id(loop), None)
        if client and not client.is_closed:
            await client.aclose()


rime_tts_service = RimeTtsService()
