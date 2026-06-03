"""Export the FastAPI OpenAPI spec to docs/api/openapi.yaml.

Usage:
    python scripts/export_openapi.py

Run from the project root. Commit the output so API clients can diff contract changes.
"""
from __future__ import annotations

import os
import sys

import yaml

# Make sure project root is on sys.path when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure required test env vars are set so app startup doesn't fail.
os.environ.setdefault("RIME_API_KEY", "export-placeholder")
os.environ.setdefault("ELEVENLABS_ENCRYPTION_KEY", "export-placeholder")

from app.main import app  # noqa: E402 — after sys.path patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "docs", "api", "openapi.yaml")


def main() -> None:
    schema = app.openapi()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(
            schema,
            fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    print(f"Exported OpenAPI spec → {OUT_PATH}")


if __name__ == "__main__":
    main()
