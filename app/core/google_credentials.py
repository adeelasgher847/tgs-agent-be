"""
Apply GOOGLE_APPLICATION_CREDENTIALS from app settings for Google Cloud ADC.

Used by Vertex AI (Gemini 2.5 Flash), Cloud TTS, and other GCP clients that read
credentials from the standard environment variable.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from app.core.config import settings
from app.core.logger import logger


def ensure_google_application_credentials_env() -> Optional[str]:
    """
    Normalize settings.GOOGLE_APPLICATION_CREDENTIALS into os.environ.

    Accepts either a filesystem path or inline JSON (same as Google TTS service).

    Returns the effective path set on os.environ, or None if not configured.
    """
    raw = (settings.GOOGLE_APPLICATION_CREDENTIALS or "").strip()
    if not raw:
        return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or None

    is_json = False
    try:
        json.loads(raw)
        is_json = True
    except (json.JSONDecodeError, ValueError):
        is_json = False

    if is_json:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as f:
                f.write(raw)
                path = f.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
            logger.debug(
                "Google ADC: using credentials from inline JSON (temp file)"
            )
            return path
        except Exception as exc:
            logger.error("Google ADC: failed to write temp credentials file: %s", exc)
            return None

    if os.path.exists(raw):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = raw
        logger.debug("Google ADC: using credentials file %s", raw)
        return raw

    logger.warning("Google ADC: credentials path does not exist: %s", raw)
    return None
