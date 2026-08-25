"""Double-brace `{{ }}` templating for System Webhooks (Call Flow).

Deliberately distinct syntax from the three existing single-brace
interpolation mechanisms already in this codebase
(`flow_pipeline_mixin._interpolate_variables`,
`hubspot_service.apply_field_mapping_values`, and `conversation_orchestrator`'s
reuse of the latter) — `{{...}}` never collides with those.

Pure functions only — no I/O, no DB access. This runs inline in hot call
paths (pre-inbound webhook URL/header/query-param rendering happens while an
inbound Twilio call is waiting on TwiML), so lookups never raise: any
unresolved token silently renders as an empty string and is logged at DEBUG.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Matches {{ name }}, {{name.path}}, {{ name.deep.path }} — word chars and dots only.
_TOKEN_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _resolve_token(path: str, context: dict[str, Any]) -> str:
    """Resolve one `{{path}}` token against *context*.

    - No dot in *path* → flat top-level lookup (`customer_name` → `context["customer_name"]`).
    - Dot(s) in *path* → walk each segment as a nested dict key
      (`_system.phoneNumber` → `context["_system"]["phoneNumber"]`).
    - Any failed step (missing key, non-dict intermediate) or a resolved value
      of ``None`` → empty string, logged at DEBUG. Never raises.
    - A resolvable non-string leaf is coerced with ``str()`` rather than dropped.
    """
    parts = path.split(".")
    value: Any = context
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            logger.debug(
                "webhook_templating: unresolved token %r (failed at segment %r)",
                path,
                part,
            )
            return ""
        value = value[part]

    if value is None:
        logger.debug("webhook_templating: token %r resolved to None", path)
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def render_template(text: str, context: dict[str, Any]) -> str:
    """Replace every `{{name}}` / `{{a.b.c}}` token in *text* using *context*.

    Never raises — unresolved tokens become empty strings. `text` is
    returned unchanged if it has no tokens or is falsy.
    """
    if not text:
        return text
    return _TOKEN_RE.sub(lambda m: _resolve_token(m.group(1), context), text)


def render_json_template(template: Any, context: dict[str, Any]) -> Any:
    """Deep-walk a JSON-like structure (dict/list/str/other), applying
    :func:`render_template` to every string leaf found. Dict keys and
    non-string values (numbers, bools, None) are left untouched.

    Used for the Post-Call Webhook's optional custom payload template.
    """
    if isinstance(template, str):
        return render_template(template, context)
    if isinstance(template, dict):
        return {
            key: render_json_template(value, context) for key, value in template.items()
        }
    if isinstance(template, list):
        return [render_json_template(item, context) for item in template]
    return template
