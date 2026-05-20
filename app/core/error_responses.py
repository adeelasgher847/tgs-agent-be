"""Structured API error payloads — no raw user PII in responses."""

from __future__ import annotations

from typing import Any

from app.core.pii_redactor import safe_error_message, status_to_error_code


def build_api_error_payload(
    status_code: int,
    detail: Any = None,
    *,
    error_code: str | None = None,
    request_id: str = "",
) -> dict[str, Any]:
    """
    Standard error JSON: { error: { code, message, requestId } }.

    Never embeds raw validation field values or unredacted exception text.
    The ``request_id`` should come from ``request.state.request_id`` at the call site.
    """
    code = error_code or status_to_error_code(status_code)
    message = safe_error_message(detail, status_code=status_code)
    return {
        "error": {
            "code": code,
            "message": message,
            "requestId": request_id,
        }
    }


def build_call_initiate_error_payload(
    status_code: int,
    detail: Any,
    call_request: Any,
    *,
    error_code: str | None = None,
    request_id: str = "",
) -> dict[str, Any]:
    """
    PII-safe call-initiate error with CRM echo fields for n8n workflows.

    Wraps the standard envelope and appends CRM fields at the top level for
    backward compatibility with n8n consumers that read those fields directly.
    """
    payload = build_api_error_payload(
        status_code, detail, error_code=error_code, request_id=request_id
    )
    # Alias used by some n8n workflow nodes.
    payload["detail"] = payload["error"]["message"]
    crm_fields = {
        "board_id": getattr(call_request, "board_id", None),
        "monday_item_id": getattr(call_request, "monday_item_id", None),
        "status_column_id": getattr(call_request, "status_column_id", None),
        "call_session_id_column_id": getattr(call_request, "call_session_id_column_id", None),
        "crm_container_id": getattr(call_request, "crm_container_id", None)
        or getattr(call_request, "board_id", None),
        "crm_item_id": getattr(call_request, "crm_item_id", None)
        or getattr(call_request, "monday_item_id", None),
        "status_field_id": getattr(call_request, "status_field_id", None)
        or getattr(call_request, "status_column_id", None),
        "call_session_id_field_id": getattr(call_request, "call_session_id_field_id", None)
        or getattr(call_request, "call_session_id_column_id", None),
        "crm_type": getattr(call_request, "crm_type", None),
    }
    payload.update({k: v for k, v in crm_fields.items() if v is not None})
    return payload
