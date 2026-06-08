#!/usr/bin/env python3
"""
Generate tests/integration/fixtures/hello_world_16k_3s.raw (LINEAR16 16kHz, 3s).

Uses Google Cloud Text-to-Speech with the same credentials as STT (ADC / .env JSON).
Requires: google-cloud-texttospeech, GOOGLE_APPLICATION_CREDENTIALS in .env
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

creds = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
if creds.startswith("{"):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        f.write(creds)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name

from google.cloud import texttospeech

TARGET_BYTES = 16000 * 2 * 3  # 3 seconds mono LINEAR16
OUT = ROOT / "tests" / "integration" / "fixtures" / "hello_world_16k_3s.raw"


def main() -> None:
    client = texttospeech.TextToSpeechClient()
    resp = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text="hello world"),
        voice=texttospeech.VoiceSelectionParams(language_code="en-AU"),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
        ),
    )
    pcm = resp.audio_content
    if len(pcm) < TARGET_BYTES:
        pcm = pcm + b"\x00" * (TARGET_BYTES - len(pcm))
    else:
        pcm = pcm[:TARGET_BYTES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(pcm)
    print(f"Wrote {len(pcm)} bytes ({len(pcm) / 32000:.2f}s) → {OUT}")


if __name__ == "__main__":
    main()
