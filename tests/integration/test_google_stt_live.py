"""
Live Google STT integration tests — require real Google Cloud credentials.

Run (loads .env automatically via python-dotenv in helpers bootstrap):

    RUN_GOOGLE_STT_INTEGRATION=1 pytest tests/integration/test_google_stt_live.py -v

Optional p95 sample count (default 10):

    RUN_GOOGLE_STT_INTEGRATION=1 GOOGLE_STT_P95_SAMPLES=20 \\
        pytest tests/integration/test_google_stt_live.py -v

Requires:
  - google-cloud-speech installed
  - GOOGLE_APPLICATION_CREDENTIALS in .env (JSON inline or file path)
  - tests/integration/fixtures/hello_world_16k_3s.raw
    (generate: python scripts/generate_stt_fixture.py)
"""
from __future__ import annotations

import asyncio
import os

import pytest
from dotenv import load_dotenv

from tests.integration.google_stt_helpers import (
    EXPECTED_PHRASE,
    bootstrap_google_credentials_from_env,
    google_cloud_speech_available,
    google_stt_credentials_configured,
    load_hello_world_fixture,
    percentile,
    stream_audio_and_collect,
)

load_dotenv()

pytestmark = pytest.mark.integration

_SKIP_LIVE = pytest.mark.skipif(
    os.environ.get("RUN_GOOGLE_STT_INTEGRATION", "").lower() not in ("1", "true", "yes"),
    reason="Set RUN_GOOGLE_STT_INTEGRATION=1 to run live Google STT tests",
)

_SKIP_NO_CREDS = pytest.mark.skipif(
    not google_stt_credentials_configured(),
    reason="GOOGLE_APPLICATION_CREDENTIALS not set — skipping live Google STT tests",
)

_SKIP_NO_SDK = pytest.mark.skipif(
    not google_cloud_speech_available(),
    reason="google-cloud-speech not installed — pip install google-cloud-speech>=2.27.0",
)


@_SKIP_LIVE
@_SKIP_NO_CREDS
@_SKIP_NO_SDK
class TestGoogleSttLive:
    @pytest.fixture(autouse=True)
    def _bootstrap_creds(self):
        bootstrap_google_credentials_from_env()

    def test_3s_audio_clip_final_transcript_matches(self):
        """Inject 3s LINEAR16 clip → final transcript contains expected phrase."""
        audio = load_hello_world_fixture()
        assert len(audio) == 16000 * 2 * 3

        finals, _ = asyncio.run(stream_audio_and_collect(audio))
        assert finals, "Expected at least one final transcript from 3s audio clip"

        combined = " ".join(f["transcript"].lower() for f in finals)
        assert EXPECTED_PHRASE in combined, (
            f"Expected '{EXPECTED_PHRASE}' in transcript, got: {combined!r}"
        )

    def test_speech_end_to_final_latency_metric_captured(self):
        """Verify speech-end→final latency metric is emitted (value varies by network)."""
        audio = load_hello_world_fixture()
        finals, latencies = asyncio.run(stream_audio_and_collect(audio))
        assert finals, "No final transcript received"
        assert latencies, "No stt_speech_end_to_final_ms metric in results"
        assert latencies[0] > 0

    def test_speech_end_to_final_latency_p95_under_400ms(self):
        """
        Ticket acceptance: p95 speech-end→final < 400ms (staging).

        Skipped locally unless STT_STRICT_LATENCY=1 (enable in staging CI).
        Override threshold: STT_P95_THRESHOLD_MS=400
        """
        if os.environ.get("STT_STRICT_LATENCY", "").lower() not in ("1", "true", "yes"):
            pytest.skip(
                "Set STT_STRICT_LATENCY=1 in staging to enforce p95 < 400ms "
                "(local dev network to Google is often > 400ms)"
            )

        sample_count = int(os.environ.get("GOOGLE_STT_P95_SAMPLES", "10"))
        threshold_ms = float(os.environ.get("STT_P95_THRESHOLD_MS", "400"))
        audio = load_hello_world_fixture()

        all_latencies: list[float] = []
        for _ in range(sample_count):
            _, latencies = asyncio.run(stream_audio_and_collect(audio))
            if latencies:
                all_latencies.append(float(latencies[0]))

        assert len(all_latencies) >= max(3, sample_count // 2), (
            f"Only collected {len(all_latencies)}/{sample_count} latency samples"
        )

        p95 = percentile(all_latencies, 95)
        assert p95 < threshold_ms, (
            f"p95 stt_speech_end_to_final={p95:.0f}ms >= {threshold_ms}ms "
            f"(samples={all_latencies})"
        )
