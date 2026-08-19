"""
Live AssemblyAI Universal-Streaming STT integration tests — require a real
ASSEMBLYAI_API_KEY.

Run (loads .env automatically via python-dotenv):

    RUN_ASSEMBLYAI_STT_INTEGRATION=1 pytest tests/integration/test_assemblyai_stt_live.py -v

Like the xAI live test, this doesn't assert transcript accuracy against a
fixed phrase (no shared speech fixture exists for this provider). Instead it
verifies the real session lifecycle end-to-end against AssemblyAI's actual
API: connect + auth succeeds, Begin arrives, audio streams without error,
and graceful shutdown (Terminate -> Termination -> close) completes cleanly.
"""
from __future__ import annotations

import asyncio
import math
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.integration

_SKIP_LIVE = pytest.mark.skipif(
    os.environ.get("RUN_ASSEMBLYAI_STT_INTEGRATION", "").lower() not in ("1", "true", "yes"),
    reason="Set RUN_ASSEMBLYAI_STT_INTEGRATION=1 to run live AssemblyAI STT tests",
)

_SKIP_NO_KEY = pytest.mark.skipif(
    not (os.environ.get("ASSEMBLYAI_API_KEY") or "").strip(),
    reason="ASSEMBLYAI_API_KEY not set — skipping live AssemblyAI STT tests",
)


def _synthetic_mulaw_tone(duration_sec: float = 2.0, sample_rate: int = 8000) -> bytes:
    """Generate a synthetic tone as 8-bit MULAW — enough to exercise the
    session lifecycle (connect/auth/stream/shutdown) without needing a real
    speech fixture. Not expected to produce a meaningful transcript."""

    def linear_to_mulaw(sample: int) -> int:
        sign = 0 if sample >= 0 else 0x80
        sample = min(abs(sample), 32635)
        sample += 132
        exponent = 7
        for exp_mask in (0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100):
            if sample & exp_mask:
                break
            exponent -= 1
        mantissa = (sample >> (exponent + 3)) & 0x0F
        return ~(sign | (exponent << 4) | mantissa) & 0xFF

    freq = 220.0
    frames = bytearray()
    for i in range(int(sample_rate * duration_sec)):
        val = int(8000 * math.sin(2 * math.pi * freq * i / sample_rate))
        frames.append(linear_to_mulaw(val))
    return bytes(frames)


@_SKIP_LIVE
@_SKIP_NO_KEY
class TestAssemblyAiSttLive:
    def test_session_connects_streams_and_shuts_down_cleanly(self):
        from app.services.assemblyai_stt_service import assemblyai_stt_service

        async def _run():
            session = assemblyai_stt_service.create_streaming_session(
                language_code="en",
                encoding="MULAW",
                sample_rate=8000,
            )
            await session.start()

            audio = _synthetic_mulaw_tone()
            chunk_size = 800  # 100ms at 8kHz mulaw
            for offset in range(0, len(audio), chunk_size):
                session.push_audio(audio[offset:offset + chunk_size])
                await asyncio.sleep(0.1)

            session.finish()

            results = []
            saw_done = False
            saw_fatal_error = False
            for _ in range(200):  # bounded drain, avoid hanging forever on a real network call
                result = await asyncio.wait_for(session.get_result(), timeout=10.0)
                results.append(result)
                if result.get("done"):
                    saw_done = True
                    break
                if result.get("error") and not result.get("recoverable", True):
                    saw_fatal_error = True
                    break

            return results, saw_done, saw_fatal_error

        results, saw_done, saw_fatal_error = asyncio.run(_run())

        assert not saw_fatal_error, f"Session ended with a non-recoverable error: {results}"
        assert saw_done, f"Session never reached a clean done state: {results}"
