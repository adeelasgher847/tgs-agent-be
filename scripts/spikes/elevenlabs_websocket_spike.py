"""
Phase 6-2 spike — ElevenLabs WebSocket streaming-input protocol.

STANDALONE / ISOLATED. This file is NOT imported by anything under `app/`
and does not change production behavior in any way. Its sole purpose is to
empirically answer the one open question Phase 6-1 (investigation) flagged
as blocking before any production integration could even be considered:

    On the ElevenLabs `stream-input` WebSocket, what actually happens when
    the client aborts mid-generation (simulating a caller barge-in)? Does
    buffered audio keep arriving after close? Is the connection safely
    discarded? Can a fresh session be opened immediately after?

Protocol reference (as documented by ElevenLabs, nothing invented here):
    wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input
        ?model_id=...&output_format=...&auto_mode=true&inactivity_timeout=20
    Auth: `xi-api-key` header on the WebSocket upgrade request — same
    convention as the REST endpoints in app/services/elevenlabs_service.py.

    Client -> server messages (JSON text frames):
      1. Init:      {"text": " ", "voice_settings": {...}}
      2. Text:       {"text": "<incremental text>"}   (repeated)
      3. Flush:      {"text": "", "flush": true}       (force-generate buffered text)
      4. Close:      {"text": ""}                       (signals end of input)

    Server -> client messages (JSON text frames):
      - Audio chunk: {"audio": "<base64 mulaw/pcm/mp3>", "isFinal": null, ...}
      - Final:       {"audio": null, "isFinal": true, ...}

Run manually (requires a real ELEVENLABS_API_KEY with `text_to_speech`
permission — this key does NOT need `voices_read`, only synthesis):

    source venv/bin/activate
    python scripts/spikes/elevenlabs_websocket_spike.py

Or via the gated pytest wrapper:

    RUN_ELEVENLABS_WEBSOCKET_SPIKE=1 pytest tests/integration/test_elevenlabs_websocket_spike.py -v -s

Nothing in this module is invoked automatically by the test suite or the
app — it is opt-in only, mirroring the RUN_GOOGLE_STT_INTEGRATION=1
convention in tests/conftest.py / tests/integration/test_google_stt_live.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make sure the project root is on sys.path when run directly as a script
# (mirrors scripts/generate_stt_fixture.py / scripts/seed_dev_workspace.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# NOTE: importing app.core.config is read-only (we only read the existing
# ELEVENLABS_API_KEY field) — this script does not modify any app/ file and
# is not imported by any app/ file.
try:
    from app.core.config import settings as _app_settings
except Exception:  # pragma: no cover - script must still run standalone
    _app_settings = None

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'websockets' package (already pinned in requirements.txt) is "
        "not installed in this interpreter. Activate venv/ first: "
        "source venv/bin/activate"
    ) from exc


DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # "Rachel" — public ElevenLabs voice, same one used in tests/conftest.py fixtures
DEFAULT_MODEL_ID = "eleven_flash_v2_5"  # mirrors production default in elevenlabs_service.py / tts_adapter.py
DEFAULT_OUTPUT_FORMAT = "ulaw_8000"  # mirrors production telephony default

WS_BASE = "wss://api.elevenlabs.io/v1/text-to-speech"

# Same voice_settings defaults as ElevenLabsService._default_voice_settings()
DEFAULT_VOICE_SETTINGS: dict[str, Any] = {
    "stability": 0.45,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
    "speed": 1.0,
}


def get_api_key() -> str:
    """Resolve ELEVENLABS_API_KEY the same way the production service does
    (env var / settings field), without importing or modifying the
    production ElevenLabsService itself."""
    if _app_settings is not None:
        key = (getattr(_app_settings, "ELEVENLABS_API_KEY", "") or "").strip()
        if key:
            return key
    env_key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if env_key:
        return env_key
    raise RuntimeError(
        "ELEVENLABS_API_KEY not found in app settings or environment. "
        "This spike requires a real ElevenLabs API key with text_to_speech "
        "permission to run live scenarios."
    )


def build_ws_url(
    voice_id: str = DEFAULT_VOICE_ID,
    model_id: str = DEFAULT_MODEL_ID,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> str:
    return (
        f"{WS_BASE}/{voice_id}/stream-input"
        f"?model_id={model_id}&output_format={output_format}"
        f"&auto_mode=true&inactivity_timeout=20"
    )


@dataclass
class AudioEvent:
    """One received server message, timestamped for latency analysis."""

    t: float
    kind: str  # "audio" | "final" | "other"
    audio_bytes: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionResult:
    """Aggregated observations for a single scenario run."""

    scenario: str
    events: list[AudioEvent] = field(default_factory=list)
    total_audio_bytes: int = 0
    first_audio_t: float | None = None
    last_text_sent_t: float | None = None
    all_text_sent_before_first_audio: bool | None = None
    flush_sent_t: float | None = None
    final_received_t: float | None = None
    close_clean: bool | None = None
    close_exception: str | None = None
    error: str | None = None

    # cancellation-specific
    cancel_requested_t: float | None = None
    events_after_cancel: int = 0
    last_event_after_cancel_t: float | None = None
    cancel_to_loop_stop_ms: float | None = None
    receive_task_cancelled_cleanly: bool | None = None


async def _receive_loop(
    ws: "websockets.ClientConnection",
    result: SessionResult,
    stop_event: asyncio.Event,
) -> None:
    """Drain server messages until the socket closes or stop_event is set
    (stop_event is only used by the cancellation scenario to detect how
    quickly the loop can actually be torn down after a client-initiated
    close)."""
    try:
        async for raw_msg in ws:
            now = time.monotonic()
            try:
                msg = json.loads(raw_msg)
            except (json.JSONDecodeError, TypeError):
                result.events.append(AudioEvent(t=now, kind="other"))
                continue

            audio_b64 = msg.get("audio")
            is_final = bool(msg.get("isFinal"))

            if audio_b64:
                audio_bytes = len(base64.b64decode(audio_b64))
                result.total_audio_bytes += audio_bytes
                if result.first_audio_t is None:
                    result.first_audio_t = now
                result.events.append(
                    AudioEvent(t=now, kind="audio", audio_bytes=audio_bytes)
                )
            elif is_final:
                result.final_received_t = now
                result.events.append(AudioEvent(t=now, kind="final", raw=msg))
            else:
                result.events.append(AudioEvent(t=now, kind="other", raw=msg))

            if result.cancel_requested_t is not None and now >= result.cancel_requested_t:
                result.events_after_cancel += 1
                result.last_event_after_cancel_t = now

            if is_final:
                break
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as exc:
        result.close_exception = f"ConnectionClosedError: {exc}"
    except asyncio.CancelledError:
        raise
    finally:
        stop_event.set()


async def run_scenario_a_normal_streaming(
    text: str = (
        "That's a great option for you. Let me explain how it works. "
        "The process is very simple."
    ),
    word_delay_sec: float = 0.12,
) -> SessionResult:
    """Scenario A — send text progressively (word-by-word), verify audio
    begins arriving before all text has been sent."""
    result = SessionResult(scenario="A_normal_streaming")
    api_key = get_api_key()
    url = build_ws_url()

    async with websockets.connect(
        url, additional_headers={"xi-api-key": api_key}
    ) as ws:
        await ws.send(
            json.dumps({"text": " ", "voice_settings": DEFAULT_VOICE_SETTINGS})
        )

        stop_event = asyncio.Event()
        recv_task = asyncio.create_task(_receive_loop(ws, result, stop_event))

        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else " ")
            await ws.send(json.dumps({"text": chunk}))
            await asyncio.sleep(word_delay_sec)
        result.last_text_sent_t = time.monotonic()

        # Signal end of input (per docs: empty text closes the input stream
        # and triggers final generation of anything still buffered).
        await ws.send(json.dumps({"text": ""}))

        try:
            await asyncio.wait_for(recv_task, timeout=30)
        except asyncio.TimeoutError:
            recv_task.cancel()
            result.error = "receive loop timed out waiting for isFinal"

        result.close_clean = ws.close_code in (None, 1000)

    if result.first_audio_t is not None and result.last_text_sent_t is not None:
        result.all_text_sent_before_first_audio = (
            result.first_audio_t >= result.last_text_sent_t
        )

    return result


async def run_scenario_b_mid_generation_cancel(
    text: str = (
        "I'd like to walk you through the entire process from start to finish, "
        "including every step along the way, so you have a complete picture "
        "before we move forward with scheduling anything."
    ),
    word_delay_sec: float = 0.1,
    cancel_after_audio_events: int = 2,
) -> SessionResult:
    """Scenario B — start a longer response, abort the connection while
    audio is actively arriving (simulated barge-in), and measure exactly
    what happens to in-flight/buffered audio and the receive task."""
    result = SessionResult(scenario="B_mid_generation_cancel")
    api_key = get_api_key()
    url = build_ws_url()

    ws = await websockets.connect(url, additional_headers={"xi-api-key": api_key})
    try:
        await ws.send(
            json.dumps({"text": " ", "voice_settings": DEFAULT_VOICE_SETTINGS})
        )

        stop_event = asyncio.Event()
        recv_task = asyncio.create_task(_receive_loop(ws, result, stop_event))

        # Feed text in the background while we watch for audio to start
        # arriving, so we can cancel truly "mid-generation" rather than
        # after all text has already been sent.
        async def _feed_text() -> None:
            words = text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else " ")
                try:
                    await ws.send(json.dumps({"text": chunk}))
                except websockets.exceptions.ConnectionClosed:
                    return
                await asyncio.sleep(word_delay_sec)

        feed_task = asyncio.create_task(_feed_text())

        # Wait until we've received at least `cancel_after_audio_events`
        # audio chunks, i.e. generation/streaming is demonstrably underway.
        deadline = time.monotonic() + 15
        while (
            sum(1 for e in result.events if e.kind == "audio")
            < cancel_after_audio_events
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.02)

        # --- Cancellation: simulate barge-in by closing the socket. ElevenLabs
        # docs do not describe an explicit "abort generation" message for the
        # single-context stream-input WebSocket (only the multi-context
        # variant's CloseContextClient, out of scope per Phase 6-1) — so the
        # simplest, most defensible mechanism is closing the connection.
        result.cancel_requested_t = time.monotonic()
        feed_task.cancel()
        try:
            await feed_task
        except asyncio.CancelledError:
            pass

        close_start = time.monotonic()
        close_exc: Exception | None = None
        try:
            await ws.close(code=1000, reason="client barge-in cancel")
        except Exception as exc:  # noqa: BLE001 - want to record ANY close-path exception for the report
            close_exc = exc

        try:
            await asyncio.wait_for(recv_task, timeout=10)
            result.receive_task_cancelled_cleanly = True
        except asyncio.TimeoutError:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass
            result.receive_task_cancelled_cleanly = False

        loop_stop_t = time.monotonic()
        result.cancel_to_loop_stop_ms = (loop_stop_t - result.cancel_requested_t) * 1000

        if close_exc is not None:
            result.close_exception = f"{type(close_exc).__name__}: {close_exc}"
            result.close_clean = False
        else:
            result.close_clean = ws.close_code in (None, 1000) and (
                time.monotonic() - close_start
            ) < 10
    finally:
        if ws.state.name != "CLOSED":
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup, already recording close behavior above
                pass

    # Confirm whether the closed socket is actually reusable (per docs it
    # should not be — verify behaviorally).
    try:
        await ws.send(json.dumps({"text": "post-close probe"}))
        result.error = (
            (result.error + "; " if result.error else "")
            + "UNEXPECTED: send() on closed socket did not raise"
        )
    except (
        websockets.exceptions.ConnectionClosed,
        websockets.exceptions.InvalidState,
    ) as exc:
        # Expected: confirms the connection must be discarded, not reused.
        result.raw_post_close_probe = f"{type(exc).__name__}: {exc}"  # type: ignore[attr-defined]

    return result


async def run_scenario_c_fresh_session_after_cancel(
    text: str = "Hello, how can I help you today?",
    word_delay_sec: float = 0.12,
) -> SessionResult:
    """Scenario C — open a brand new WebSocket session immediately after a
    cancelled one and verify no stale audio leaks in and behavior is
    otherwise identical to a first-ever session."""
    result = await run_scenario_a_normal_streaming(
        text=text, word_delay_sec=word_delay_sec
    )
    result.scenario = "C_fresh_session_after_cancel"
    return result


async def run_scenario_d_multi_turn(
) -> tuple[list[SessionResult], set[str], set[str]]:
    """Scenario D — 4 sequential turns (normal, normal, cancelled, normal).
    Checks for orphaned asyncio tasks and cross-turn state leakage."""
    tasks_before = {repr(t) for t in asyncio.all_tasks()}

    results: list[SessionResult] = []

    r1 = await run_scenario_a_normal_streaming(
        text="Sure, I can help with that today.", word_delay_sec=0.1
    )
    r1.scenario = "D_turn1_normal"
    results.append(r1)

    r2 = await run_scenario_a_normal_streaming(
        text="Let's go ahead and get you scheduled.", word_delay_sec=0.1
    )
    r2.scenario = "D_turn2_normal"
    results.append(r2)

    r3 = await run_scenario_b_mid_generation_cancel(
        text="Let me pull up your account details and walk through the "
        "available time slots for this week and next.",
        word_delay_sec=0.08,
        cancel_after_audio_events=1,
    )
    r3.scenario = "D_turn3_cancelled"
    results.append(r3)

    r4 = await run_scenario_a_normal_streaming(
        text="Great, you're all set for Thursday at two PM.", word_delay_sec=0.1
    )
    r4.scenario = "D_turn4_normal"
    results.append(r4)

    # Give any lingering close/cleanup coroutines a beat to finish so we
    # don't false-positive on tasks that are just winding down.
    await asyncio.sleep(0.2)
    tasks_after = {repr(t) for t in asyncio.all_tasks()}

    return results, tasks_before, tasks_after


def _fmt_result(r: SessionResult) -> str:
    lines = [f"--- Scenario: {r.scenario} ---"]
    lines.append(f"  total_audio_bytes: {r.total_audio_bytes}")
    lines.append(f"  audio_events: {sum(1 for e in r.events if e.kind == 'audio')}")
    lines.append(f"  first_audio_t: {r.first_audio_t}")
    lines.append(f"  last_text_sent_t: {r.last_text_sent_t}")
    lines.append(
        f"  audio_began_before_all_text_sent: "
        f"{None if r.all_text_sent_before_first_audio is None else not r.all_text_sent_before_first_audio}"
    )
    lines.append(f"  final_received_t: {r.final_received_t}")
    lines.append(f"  close_clean: {r.close_clean}")
    lines.append(f"  close_exception: {r.close_exception}")
    if r.cancel_requested_t is not None:
        lines.append(f"  cancel_requested_t: {r.cancel_requested_t}")
        lines.append(f"  events_after_cancel: {r.events_after_cancel}")
        lines.append(f"  last_event_after_cancel_t: {r.last_event_after_cancel_t}")
        lines.append(f"  cancel_to_loop_stop_ms: {r.cancel_to_loop_stop_ms}")
        lines.append(
            f"  receive_task_cancelled_cleanly: {r.receive_task_cancelled_cleanly}"
        )
        post_close_probe = getattr(r, "raw_post_close_probe", None)
        lines.append(f"  post_close_send_probe: {post_close_probe}")
    if r.error:
        lines.append(f"  error: {r.error}")
    return "\n".join(lines)


async def main() -> None:
    print("=== ElevenLabs WebSocket streaming-input spike (Phase 6-2) ===")
    print(f"voice_id={DEFAULT_VOICE_ID} model_id={DEFAULT_MODEL_ID} "
          f"output_format={DEFAULT_OUTPUT_FORMAT}")

    r_a = await run_scenario_a_normal_streaming()
    print(_fmt_result(r_a))

    r_b = await run_scenario_b_mid_generation_cancel()
    print(_fmt_result(r_b))

    r_c = await run_scenario_c_fresh_session_after_cancel()
    print(_fmt_result(r_c))

    d_results, tasks_before, tasks_after = await run_scenario_d_multi_turn()
    for r in d_results:
        print(_fmt_result(r))
    leaked = tasks_after - tasks_before
    print("--- Scenario D task-leak check ---")
    print(f"  tasks_before={len(tasks_before)} tasks_after={len(tasks_after)} "
          f"leaked_tasks={len(leaked)}")
    if leaked:
        for t in leaked:
            print(f"    LEAKED: {t}")

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
