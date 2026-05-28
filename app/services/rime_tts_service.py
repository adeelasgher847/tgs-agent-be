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
from typing import AsyncIterator, Optional

import httpx

from app.core.config import settings
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
        # Client is created lazily (no event loop at module import time).
        self._client: Optional[httpx.AsyncClient] = None

    def _api_key(self) -> str:
        # get_rime_api_key() handles env-resolution for dev/staging/production,
        # including GCP Secret Manager in production, with a clear error on missing key.
        return get_rime_api_key()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=5.0),
                http2=False,
            )
        return self._client

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
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "Accept": "audio/basic",  # mulaw MIME
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
        if self._client and not self._client.is_closed:
            await self._client.aclose()


rime_tts_service = RimeTtsService()
