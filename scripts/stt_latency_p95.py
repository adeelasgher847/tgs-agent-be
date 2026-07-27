#!/usr/bin/env python3
"""
STT latency p95 measurement tool (ticket acceptance: p95 < 400 ms staging).

Metric: stt_speech_end_to_final
  Boundary: caller speech end (finish() / last audio) → first final transcript from Google STT.

Parse application logs for lines emitted by google_stt_service.py:
  [Metrics] stt_speech_end_to_final=NNN ms

Usage:
  python scripts/stt_latency_p95.py --log /var/log/voiceagent/app.log
  kubectl logs -f deployment/tgs-agent-be | python scripts/stt_latency_p95.py --stdin
  python scripts/stt_latency_p95.py --log app.log --threshold 400 --min-samples 20

Exit codes:
  0  p95 < threshold (pass)
  1  p95 >= threshold (fail)
  2  not enough samples
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from typing import List

P95_THRESHOLD_MS = 400.0
MIN_SAMPLES_DEFAULT = 10

_METRIC_RE = re.compile(
    r"\[Metrics\]\s+stt_speech_end_to_final=([\d.]+)\s*ms",
)


def _parse_latencies(lines: List[str]) -> List[float]:
    values: List[float] = []
    for line in lines:
        m = _METRIC_RE.search(line)
        if m:
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
    return values


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _report(latencies: List[float], threshold: float = P95_THRESHOLD_MS) -> int:
    n = len(latencies)
    mn = min(latencies)
    mx = max(latencies)
    mean = statistics.mean(latencies)
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)

    verdict = "✓ PASS" if p95 < threshold else "✗ FAIL"
    print(f"Metric  : stt_speech_end_to_final")
    print(f"Samples : {n}")
    print(f"Min     : {mn:6.0f} ms")
    print(f"Mean    : {mean:6.0f} ms")
    print(f"p50     : {p50:6.0f} ms")
    print(f"p95     : {p95:6.0f} ms   {verdict} (threshold: {threshold:.0f} ms)")
    print(f"p99     : {p99:6.0f} ms")
    print(f"Max     : {mx:6.0f} ms")

    return 0 if p95 < threshold else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute p95 STT speech-end→final latency from application logs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--log", metavar="FILE", help="Path to log file")
    group.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES_DEFAULT,
        metavar="N",
        help=f"Minimum samples required (default: {MIN_SAMPLES_DEFAULT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=P95_THRESHOLD_MS,
        metavar="MS",
        help=f"p95 pass threshold in ms (default: {P95_THRESHOLD_MS})",
    )
    args = parser.parse_args()

    if args.stdin:
        lines = sys.stdin.readlines()
    else:
        try:
            with open(args.log, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"ERROR: log file not found: {args.log}", file=sys.stderr)
            return 2

    latencies = _parse_latencies(lines)
    if len(latencies) < args.min_samples:
        print(
            f"ERROR: only {len(latencies)} samples found (need {args.min_samples}). "
            "Run more Google STT test calls or lower --min-samples.",
            file=sys.stderr,
        )
        return 2

    return _report(latencies, threshold=args.threshold)


if __name__ == "__main__":
    sys.exit(main())
