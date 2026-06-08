"""Shared helpers for live Google STT integration tests."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hello_world_16k_3s.raw"
EXPECTED_PHRASE = "hello"


def google_stt_credentials_configured() -> bool:
    """True when GOOGLE_APPLICATION_CREDENTIALS is set (JSON inline or file path)."""
    raw = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if not raw:
        return False
    if raw.startswith("{"):
        try:
            json.loads(raw)
            return True
        except json.JSONDecodeError:
            return False
    return Path(raw).expanduser().is_file()


def google_cloud_speech_available() -> bool:
    try:
        import google.cloud.speech_v1p1beta1  # noqa: F401

        return True
    except ImportError:
        return False


def load_hello_world_fixture() -> bytes:
    if not FIXTURE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing fixture {FIXTURE_PATH}. "
            "Run: python scripts/generate_stt_fixture.py"
        )
    return FIXTURE_PATH.read_bytes()


async def stream_audio_and_collect(
    audio: bytes,
    *,
    language_code: str = "en-AU",
    chunk_bytes: int = 6400,
    chunk_delay_sec: float = 0.0,
    result_timeout_sec: float = 30.0,
) -> tuple[list[dict[str, Any]], list[int]]:
    """
    Push LINEAR16 audio through GoogleSttService streaming session.
    Returns (final_results, speech_end_to_final_ms_samples).

    Audio is pushed quickly then finish() is called to mimic caller speech end;
    latency is measured from finish() to the post-end final transcript.
    """
    from app.services.google_stt_service import GoogleSttService

    svc = GoogleSttService()
    sess = svc.create_streaming_session(
        language_code=language_code,
        sample_rate_hz=16000,
        encoding="LINEAR16",
        interim_results=True,
        api_config={"google_model": "phone_call"},
        silence_threshold_ms=1500,
    )

    await sess.start()
    await asyncio.sleep(0.05)
    for offset in range(0, len(audio), chunk_bytes):
        sess.push_audio(audio[offset : offset + chunk_bytes])
        if chunk_delay_sec > 0:
            await asyncio.sleep(chunk_delay_sec)
    sess.finish()

    finals: list[dict[str, Any]] = []
    latencies: list[int] = []
    deadline = time.monotonic() + result_timeout_sec
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            result = await asyncio.wait_for(sess.get_result(), timeout=min(remaining, 5.0))
        except asyncio.TimeoutError:
            continue
        if result.get("done"):
            break
        if result.get("error"):
            if not result.get("recoverable"):
                raise RuntimeError(f"Google STT error: {result.get('error')}")
            continue
        if result.get("is_final") and (result.get("transcript") or "").strip():
            finals.append(result)
            ms = result.get("stt_speech_end_to_final_ms")
            if isinstance(ms, (int, float)) and ms >= 0:
                latencies.append(int(ms))

    return finals, latencies


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def bootstrap_google_credentials_from_env() -> None:
    """Write inline JSON creds to a temp file so SpeechClient can load them."""
    raw = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if raw.startswith("{"):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write(raw)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
