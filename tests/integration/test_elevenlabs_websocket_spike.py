"""
Phase 6-2 spike — live ElevenLabs WebSocket streaming-input tests.

These tests hit the REAL ElevenLabs API over a real WebSocket connection.
They are OFF by default and only run when explicitly opted into, mirroring
the RUN_GOOGLE_STT_INTEGRATION=1 convention used by
tests/integration/test_google_stt_live.py.

Run:

    RUN_ELEVENLABS_WEBSOCKET_SPIKE=1 pytest \\
        tests/integration/test_elevenlabs_websocket_spike.py -v -s

Requires:
  - `websockets` installed (already pinned in requirements.txt)
  - ELEVENLABS_API_KEY in .env / environment, with text_to_speech permission
    (voices_read is NOT required)

This is a standalone spike per Phase 6-2 of the voice-humanization
investigation — it is NOT part of the production test suite's default run,
does not touch app/voice/* or app/services/elevenlabs_service.py, and its
target module (scripts/spikes/elevenlabs_websocket_spike.py) is not
imported anywhere under app/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_SKIP_LIVE = pytest.mark.skipif(
    os.environ.get("RUN_ELEVENLABS_WEBSOCKET_SPIKE", "").lower()
    not in ("1", "true", "yes"),
    reason="Set RUN_ELEVENLABS_WEBSOCKET_SPIKE=1 to run the live ElevenLabs "
    "WebSocket spike (Phase 6-2)",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _has_api_key() -> bool:
    try:
        from scripts.spikes.elevenlabs_websocket_spike import get_api_key

        get_api_key()
        return True
    except Exception:
        return False


_SKIP_NO_KEY = pytest.mark.skipif(
    not _has_api_key(),
    reason="ELEVENLABS_API_KEY not configured — skipping live ElevenLabs "
    "WebSocket spike",
)


@_SKIP_LIVE
@_SKIP_NO_KEY
class TestElevenLabsWebSocketSpike:
    @pytest.mark.asyncio
    async def test_scenario_a_normal_streaming_audio_arrives_before_all_text_sent(self):
        from scripts.spikes.elevenlabs_websocket_spike import (
            run_scenario_a_normal_streaming,
        )

        result = await run_scenario_a_normal_streaming()
        assert result.total_audio_bytes > 0, "Expected some audio to be received"
        assert result.first_audio_t is not None
        assert result.all_text_sent_before_first_audio is not None
        # Core "does streaming actually stream" sanity check.
        assert result.all_text_sent_before_first_audio is False, (
            "Expected first audio chunk to arrive before all text was sent "
            "(true incremental streaming)"
        )

    @pytest.mark.asyncio
    async def test_scenario_b_mid_generation_cancel_records_metrics(self):
        from scripts.spikes.elevenlabs_websocket_spike import (
            run_scenario_b_mid_generation_cancel,
        )

        result = await run_scenario_b_mid_generation_cancel()
        assert result.cancel_requested_t is not None
        assert result.cancel_to_loop_stop_ms is not None
        assert result.receive_task_cancelled_cleanly is not None
        # Just documenting/measuring behavior — no hard assertion on
        # events_after_cancel since ElevenLabs' buffering behavior is
        # exactly the unknown this spike exists to characterize.

    @pytest.mark.asyncio
    async def test_scenario_c_fresh_session_after_cancel_behaves_normally(self):
        from scripts.spikes.elevenlabs_websocket_spike import (
            run_scenario_b_mid_generation_cancel,
            run_scenario_c_fresh_session_after_cancel,
        )

        await run_scenario_b_mid_generation_cancel()
        result = await run_scenario_c_fresh_session_after_cancel()
        assert result.total_audio_bytes > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_scenario_d_multi_turn_no_task_leaks(self):
        from scripts.spikes.elevenlabs_websocket_spike import (
            run_scenario_d_multi_turn,
        )

        results, tasks_before, tasks_after = await run_scenario_d_multi_turn()
        assert len(results) == 4
        leaked = tasks_after - tasks_before
        assert not leaked, f"Orphaned asyncio tasks after multi-turn sequence: {leaked}"
