"""
Regression test: the ARQ worker must register every SQLAlchemy model before
any query can run against a model with a string-based relationship() to a
class the worker doesn't otherwise import.

Unlike the FastAPI process — where importing the router tree transitively
imports nearly every model — app/workers/batch_call_worker.py imports models
lazily, function-by-function. Its on_startup hook queries CallbackSchedule
before any job has run, and CallbackSchedule.original_call resolves the
string "CallSession" against SQLAlchemy's class registry at mapper-configure
time. If CallSession was never imported in-process, that lookup fails with
InvalidRequestError, and the worker's on_startup hook (and every subsequent
job touching CallbackSchedule) crashes.

This can only be verified in a genuinely fresh interpreter — an in-process
pytest run already has every model imported via conftest.py — so this test
spawns a subprocess that imports nothing but the worker module itself.
"""
from __future__ import annotations

import subprocess
import sys

_PROBE = """
import os, sys
os.environ.setdefault("RATE_LIMIT_ENABLED", "False")
os.environ.setdefault("RIME_API_KEY", "test-rime-key")
os.environ.setdefault("ELEVENLABS_ENCRYPTION_KEY", "test-elevenlabs-key-32-chars-long!!")
os.environ.setdefault("WEBHOOK_SECRET_ENCRYPTION_KEY", "test-webhook-key")
os.environ.setdefault("HUBSPOT_TOKEN_ENCRYPTION_KEY", "test-hubspot-key")

from unittest.mock import MagicMock
for name in (
    "google", "google.genai", "google.oauth2", "google.auth",
    "google.auth.transport", "google.auth.transport.requests",
    "google.cloud", "google.cloud.speech_v1p1beta1",
    "google.api_core", "google.api_core.client_options",
    "google.api_core.exceptions",
):
    sys.modules[name] = MagicMock()

# Import ONLY the worker module, exactly as `arq app.workers.batch_call_worker.WorkerSettings`
# would — nothing else has imported CallSession into the registry yet.
import app.workers.batch_call_worker  # noqa: F401
from app.models.callback_schedule import CallbackSchedule
from sqlalchemy.orm import configure_mappers

# This is exactly what fails inside startup_recover_callbacks if CallSession
# was never registered: SQLAlchemy resolves relationship() string targets
# lazily, the first time the mapper is configured.
configure_mappers()
assert CallbackSchedule.original_call.property.mapper.class_.__name__ == "CallSession"
print("OK")
"""


def test_worker_module_registers_all_models_before_startup_query():
    """Importing the worker module alone must be enough to resolve every
    string-based relationship() used by models it queries in on_startup."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Worker module import failed to register all models.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout
