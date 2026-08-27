"""
Regression tests for the VoiceTurnMetrics per-call/per-turn scoping bug.

Root cause (see app/voice/metrics.py::VoiceTurnMetrics.start_generation):
`rag_start_mono`, `rag_end_mono`, and `llm_request_mono` were written via
idempotent guards (`if self.x is None: self.x = ...`) but `start_generation()`
— called once per turn — never reset them back to None. Since a
`VoiceTurnMetrics` instance lives for the whole CALL (constructed once in
`BidirectionalStreamHandler.__init__` / `LiveKitBrowserCallHandler.__init__`),
those three fields were written exactly once, on the call's first turn, and
then stayed frozen there for every subsequent turn — corrupting
`rag_latency_ms`, `llm_ttft_ms`, and the raw timestamps in
`build_telemetry_payload()` for every turn after the first.

Fix: `start_generation()` now resets those three fields to None (in addition
to the fields it already reset) and bumps a real per-turn `turn_id` counter.
mark_rag_start / mark_rag_end / mark_llm_request also accept an optional
`expected_turn_id` so a write arriving from a background task that was
scheduled for an earlier (superseded) turn is dropped instead of corrupting
the new turn's fields.
"""
from __future__ import annotations

import time

from app.voice.metrics import VoiceTurnMetrics


def test_start_generation_resets_rag_and_llm_request_fields_across_turns():
    vm = VoiceTurnMetrics()

    # Turn 1: mark everything.
    vm.start_generation()
    vm.mark_rag_start()
    time.sleep(0.001)
    vm.mark_rag_end()
    vm.mark_llm_request()
    turn_1_rag_start = vm.rag_start_mono
    turn_1_rag_end = vm.rag_end_mono
    turn_1_llm_request = vm.llm_request_mono
    assert turn_1_rag_start is not None
    assert turn_1_rag_end is not None
    assert turn_1_llm_request is not None

    # Turn 2 starts — per the bug, these three fields used to survive
    # unreset because start_generation() never touched them and their
    # mark_* guards ("if self.x is None") would then refuse to overwrite
    # the stale turn-1 value.
    vm.start_generation()
    assert vm.rag_start_mono is None
    assert vm.rag_end_mono is None
    assert vm.llm_request_mono is None

    # And a fresh mark for turn 2 must actually take (not be blocked by a
    # stale guard from turn 1).
    time.sleep(0.001)
    vm.mark_rag_start()
    vm.mark_rag_end()
    vm.mark_llm_request()
    assert vm.rag_start_mono is not None
    assert vm.rag_start_mono != turn_1_rag_start
    assert vm.rag_end_mono != turn_1_rag_end
    assert vm.llm_request_mono != turn_1_llm_request


def test_turn_id_increments_on_every_start_generation_call():
    vm = VoiceTurnMetrics()
    assert vm.turn_id is None
    first = vm.start_generation()
    assert first == 1
    assert vm.turn_id == 1
    second = vm.start_generation()
    assert second == 2
    assert vm.turn_id == 2


def test_llm_ttft_computed_from_same_turn_fields_for_sequential_turns():
    """
    Reproduces the production symptom: llm_ttft_ms growing monotonically
    across turns because llm_base = self.llm_request_mono (frozen from turn
    1) while turn_llm_first_token_mono correctly advances every turn. With
    the fix, each turn's llm_ttft_ms must be computed purely from that same
    turn's own llm_request_mono and turn_llm_first_token_mono.
    """
    vm = VoiceTurnMetrics()

    # Turn 1: llm_request -> first_token gap of ~30ms.
    vm.start_generation()
    vm.mark_llm_request()
    time.sleep(0.03)
    vm.mark_llm_first_token()
    turn_1_latencies = vm.calculate_latencies()
    turn_1_ttft = turn_1_latencies["llm_ttft_ms"]
    assert turn_1_ttft is not None
    assert 15 <= turn_1_ttft <= 200  # generous CI-safe bounds around ~30ms

    # Simulate real inter-turn spacing (many seconds of conversation) before
    # the next turn starts — this is exactly the gap that leaked into the
    # corrupted production llm_ttft_ms values.
    time.sleep(0.05)

    # Turn 2: llm_request -> first_token gap of ~60ms (deliberately
    # different from turn 1 so a leaked turn-1 timestamp would be detectable
    # via a wildly different/larger computed value).
    vm.start_generation()
    vm.mark_llm_request()
    time.sleep(0.06)
    vm.mark_llm_first_token()
    turn_2_latencies = vm.calculate_latencies()
    turn_2_ttft = turn_2_latencies["llm_ttft_ms"]
    assert turn_2_ttft is not None
    assert 30 <= turn_2_ttft <= 300  # generous CI-safe bounds around ~60ms

    # The critical assertion: turn 2's TTFT must NOT include the inter-turn
    # spacing (the sleep(0.05) above) that a frozen turn-1 llm_request_mono
    # would have introduced. If the bug were present, turn_2_ttft would be
    # inflated by roughly that inter-turn gap on top of the real ~60ms.
    assert turn_2_ttft < 500


def test_build_telemetry_payload_never_reports_stale_cross_turn_rag_timestamps():
    vm = VoiceTurnMetrics(call_sid="CA_test")

    vm.start_generation()
    vm.mark_rag_start()
    vm.mark_rag_end()
    vm.mark_llm_request()
    turn_1_payload = vm.build_telemetry_payload()
    assert turn_1_payload["timestamps_mono"]["rag_start"] is not None

    # New turn begins with NO rag activity at all this turn (e.g. RAG wasn't
    # configured, or the gate rejected the utterance) — the payload must
    # reflect that honestly as None, not silently reuse turn 1's value.
    vm.start_generation()
    turn_2_payload = vm.build_telemetry_payload()
    assert turn_2_payload["timestamps_mono"]["rag_start"] is None
    assert turn_2_payload["timestamps_mono"]["rag_end"] is None
    assert turn_2_payload["timestamps_mono"]["llm_request"] is None
    # generation_start, by contrast, is turn-scoped and DOES advance.
    assert (
        turn_2_payload["timestamps_mono"]["generation_start"]
        != turn_1_payload["timestamps_mono"]["generation_start"]
    )


def test_stale_turn_write_rejected_via_expected_turn_id():
    """
    Simulates a detached RAG-prefetch task that was scheduled during turn 1
    (snapshotting turn_id=1) but only gets around to writing mark_rag_end()
    after a barge-in has already advanced the metrics object to turn 2 (e.g.
    cancel() was requested but the task's executor thread hadn't yet
    observed CancelledError). The write must be dropped, not applied to
    turn 2's fields.
    """
    vm = VoiceTurnMetrics()

    turn_1_id = vm.start_generation()
    vm.mark_rag_start(expected_turn_id=turn_1_id)
    assert vm.rag_start_mono is not None

    # Barge-in: a new turn starts before the stale task's write lands.
    turn_2_id = vm.start_generation()
    assert turn_2_id != turn_1_id
    assert vm.rag_start_mono is None  # reset for turn 2

    # The stale (turn-1-scoped) background task finally "completes" and
    # tries to write — using the turn_id it captured back when it was
    # scheduled, which no longer matches.
    vm.mark_rag_end(expected_turn_id=turn_1_id)
    assert vm.rag_end_mono is None  # dropped — turn 2 must not see it

    vm.mark_rag_start(expected_turn_id=turn_1_id)
    assert vm.rag_start_mono is None  # also dropped

    # A genuine turn-2-scoped write (matching the CURRENT turn_id) still
    # works normally.
    vm.mark_rag_start(expected_turn_id=turn_2_id)
    assert vm.rag_start_mono is not None


def test_mark_methods_without_expected_turn_id_remain_backward_compatible():
    """Existing call sites that don't pass expected_turn_id (the vast
    majority of production call sites) must keep working exactly as before —
    the turn_id validation is opt-in per caller."""
    vm = VoiceTurnMetrics()
    vm.start_generation()
    vm.mark_rag_start()
    vm.mark_rag_end()
    vm.mark_llm_request()
    assert vm.rag_start_mono is not None
    assert vm.rag_end_mono is not None
    assert vm.llm_request_mono is not None
