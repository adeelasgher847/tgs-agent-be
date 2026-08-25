"""
Voice Pipeline Concurrency & Load Testing Benchmark Harness.

Simulates concurrent voice call sessions (1, 5, 10, 25, 50 calls) on the active FastAPI/asyncio runtime.
Measures P50, P90, P95, P99 latency percentiles for:
  - STT Event Dispatch Latency
  - RAG Retrieval Latency (with cache hit / miss variations)
  - Prompt Assembly Latency
  - LLM Time to First Token (TTFT)
  - TTS Time to First Audio (TTFA)
  - Playback Gate Wait
  - Total End-to-End Turn Latency
  - Event Loop Lag / Scheduler Jitter
"""
from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from typing import Any, Dict, List

from app.voice.metrics import VoiceTurnMetrics


def percentile(data: List[float], p: float) -> float:
    """Compute p-th percentile (0.0 <= p <= 100.0) from raw data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return round(d0 + d1, 2)


async def measure_event_loop_lag(duration_sec: float = 0.5) -> float:
    """Measure event loop scheduling delay under current load."""
    t_start = time.perf_counter()
    await asyncio.sleep(0.01)
    actual_elapsed = time.perf_counter() - t_start
    lag_ms = max(0.0, (actual_elapsed - 0.01) * 1000)
    return round(lag_ms, 2)


async def simulate_single_voice_turn(
    call_id: str,
    turn_id: int,
    rag_cache_hit_prob: float = 0.6,
) -> Dict[str, float]:
    """
    Simulate a complete voice turn through the pipeline stages.
    """
    metrics = VoiceTurnMetrics(
        call_sid=f"CA_bench_{call_id}",
        turn_id=f"turn_{turn_id}",
        transport="telephony",
        agent_id="agent_benchmark",
        provider="gemini",
    )

    t0 = time.perf_counter()
    # Simulated acoustic speech end happened 400ms prior to final STT emission
    acoustic_speech_end = t0 - 0.400
    metrics.begin_turn_at_stt_final(acoustic_speech_end_mono=acoustic_speech_end)

    # 1. STT Dispatch to generation start
    await asyncio.sleep(0.0005)  # micro-yield
    metrics.start_generation()

    # 2. RAG Retrieval Stage (simulate cache hit vs DB lookup)
    metrics.mark_rag_start()
    is_cache_hit = (turn_id % 10 < (rag_cache_hit_prob * 10))
    if is_cache_hit:
        # In-memory Redis cache hit: 0.5ms - 2ms
        await asyncio.sleep(0.001)
    else:
        # Vector search query: 25ms - 45ms
        await asyncio.sleep(0.030)
    metrics.mark_rag_end()

    # 3. Prompt Assembly (string templating, in-memory context stitching)
    await asyncio.sleep(0.003)
    metrics.mark_prompt_ready()

    # 4. LLM Stream Request & TTFT
    metrics.mark_llm_request()
    # Simulated LLM TTFT (300ms + minor jitter)
    await asyncio.sleep(0.320)
    metrics.mark_llm_first_token()

    # 5. TTS Queue & Synthesis TTFA
    metrics.mark_first_tts_queued()
    # Simulated TTS TTFA (200ms)
    await asyncio.sleep(0.210)
    metrics.mark_tts_first_audio()

    # 6. First Audio Playback & Jitter Priming
    await asyncio.sleep(0.015)
    metrics.mark_first_playback()

    # 7. Turn Completion
    await asyncio.sleep(0.400)
    metrics.mark_turn_complete()

    latencies = metrics.calculate_latencies()
    return latencies


async def run_concurrency_batch(num_concurrent_calls: int, turns_per_call: int = 3) -> Dict[str, Any]:
    """Run a batch of concurrent calls and collect latency distributions."""
    all_latencies: Dict[str, List[float]] = {
        "acoustic_endpointing_ms": [],
        "stt_dispatch_ms": [],
        "rag_latency_ms": [],
        "prompt_assembly_latency_ms": [],
        "llm_ttft_ms": [],
        "tts_ttfa_ms": [],
        "stt_final_to_first_audio_ms": [],
        "acoustic_end_to_first_playback_ms": [],
        "total_turn_latency_ms": [],
    }

    loop_lags: List[float] = []

    async def call_worker(call_idx: int):
        for turn_idx in range(turns_per_call):
            lag = await measure_event_loop_lag()
            loop_lags.append(lag)
            turn_lat = await simulate_single_voice_turn(
                call_id=str(call_idx),
                turn_id=turn_idx,
            )
            for k, v in turn_lat.items():
                if v is not None and k in all_latencies:
                    all_latencies[k].append(v)
            await asyncio.sleep(0.05)  # user pause between turns

    t_start = time.perf_counter()
    tasks = [call_worker(i) for i in range(num_concurrent_calls)]
    await asyncio.gather(*tasks)
    total_elapsed = round(time.perf_counter() - t_start, 2)

    # Compute percentiles for each metric
    percentiles: Dict[str, Dict[str, float]] = {}
    for metric_name, values in all_latencies.items():
        if values:
            percentiles[metric_name] = {
                "P50": percentile(values, 50),
                "P90": percentile(values, 90),
                "P95": percentile(values, 95),
                "P99": percentile(values, 99),
                "mean": round(statistics.mean(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
            }

    percentiles["event_loop_lag_ms"] = {
        "P50": percentile(loop_lags, 50),
        "P90": percentile(loop_lags, 90),
        "P95": percentile(loop_lags, 95),
        "P99": percentile(loop_lags, 99),
        "max": round(max(loop_lags) if loop_lags else 0.0, 2),
    }

    return {
        "concurrent_calls": num_concurrent_calls,
        "total_turns_measured": num_concurrent_calls * turns_per_call,
        "total_duration_sec": total_elapsed,
        "metrics": percentiles,
    }


async def main():
    levels = [1, 5, 10, 25, 50]
    full_report = {}
    print(f"Starting Concurrency Benchmark across levels: {levels}...")
    for concurrency in levels:
        res = await run_concurrency_batch(num_concurrent_calls=concurrency, turns_per_call=3)
        full_report[f"concurrency_{concurrency}"] = res
        print(f"  -> Concurrency {concurrency:2d}: P95 Total Turnaround = {res['metrics']['total_turn_latency_ms']['P95']}ms, Loop Lag P95 = {res['metrics']['event_loop_lag_ms']['P95']}ms")

    with open("scripts/benchmarks/benchmark_results.json", "w") as f:
        json.dump(full_report, f, indent=2)
    print("\nBenchmark results saved to scripts/benchmarks/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
