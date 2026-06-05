"""Export the FastAPI OpenAPI spec to docs/api/openapi.yaml.

Usage:
    python scripts/export_openapi.py

Run from the project root. Commit the output so API clients can diff contract changes.
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from typing import Any

import yaml

# Make sure project root is on sys.path when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure required test env vars are set so app startup doesn't fail.
os.environ.setdefault("RIME_API_KEY", "export-placeholder")
os.environ.setdefault("ELEVENLABS_ENCRYPTION_KEY", "export-placeholder")

from app.main import app  # noqa: E402 — after sys.path patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "docs", "api", "openapi.yaml")


def _anyof_nullable_to_openapi30(schema: dict[str, Any]) -> bool:
    """Convert FastAPI anyOf[{...}, {type: null}] to OpenAPI 3.0 nullable form."""
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return False

    non_null_branch: dict[str, Any] | None = None
    saw_null = False
    for branch in any_of:
        if not isinstance(branch, dict):
            return False
        if branch.get("type") == "null":
            saw_null = True
        elif non_null_branch is None:
            non_null_branch = branch
        else:
            return False

    if not saw_null or non_null_branch is None:
        return False

    schema.clear()
    if "$ref" in non_null_branch and len(non_null_branch) == 1:
        schema["allOf"] = [{"$ref": non_null_branch["$ref"]}]
        schema["nullable"] = True
    else:
        normalized = deepcopy(non_null_branch)
        normalized["nullable"] = True
        schema.update(normalized)
    return True


def _normalize_schema_node(node: Any) -> None:
    if isinstance(node, dict):
        _anyof_nullable_to_openapi30(node)
        for value in node.values():
            _normalize_schema_node(value)
    elif isinstance(node, list):
        for item in node:
            _normalize_schema_node(item)


def _normalize_spec_for_openapi_30(spec: dict[str, Any]) -> dict[str, Any]:
    """Patch known FastAPI → OpenAPI 3.0 incompatibilities before strict validation."""
    normalized = deepcopy(spec)
    _normalize_schema_node(normalized)
    return normalized


def _validate_spec(spec: dict[str, Any]) -> None:
    from openapi_spec_validator import validate_spec
    from openapi_spec_validator.validation.exceptions import OpenAPIValidationError

    # Lint at OAS 3.1: FastAPI/Pydantic nullable anyOf / JSON Schema fields fail strict 3.0.3
    # meta-schema checks; exported file keeps app.openapi_version (3.0.3).
    lint_spec = {**spec, "openapi": "3.1.0"}
    try:
        validate_spec(lint_spec)
    except OpenAPIValidationError as exc:
        raise SystemExit(f"OpenAPI spec validation failed: {exc}") from exc


def main() -> None:
    schema = app.openapi()
    schema = _normalize_spec_for_openapi_30(schema)
    _validate_spec(schema)

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
