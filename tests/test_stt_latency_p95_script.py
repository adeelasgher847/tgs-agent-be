"""Unit tests for scripts/stt_latency_p95.py log parser."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stt_latency_p95.py"


def test_stt_latency_p95_passes_under_400ms(tmp_path):
    log = tmp_path / "app.log"
    log.write_text(
        "\n".join(
            f"2026-06-08 INFO [Metrics] stt_speech_end_to_final={ms} ms"
            for ms in [120, 180, 200, 150, 190, 210, 170, 160, 140, 130]
        )
    )
    # S603: no shell=True; args are the fixed interpreter/script paths plus a path
    # from pytest's own tmp_path fixture (OS-generated, not external input).
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--log", str(log), "--min-samples", "5"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "p95" in proc.stdout
    assert "PASS" in proc.stdout


def test_stt_latency_p95_fails_over_400ms(tmp_path):
    log = tmp_path / "app.log"
    log.write_text(
        "\n".join(
            f"INFO [Metrics] stt_speech_end_to_final={ms} ms"
            for ms in [500, 520, 480, 510, 490, 505, 495, 515, 530, 600]
        )
    )
    # S603: no shell=True; args are the fixed interpreter/script paths plus a path
    # from pytest's own tmp_path fixture (OS-generated, not external input).
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--log", str(log), "--min-samples", "5"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "FAIL" in proc.stdout
