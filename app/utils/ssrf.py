"""SSRF guard for outbound webhook URLs.

Called at two points in the webhook lifecycle:
1. Schema validation (create) — raises SSRFBlockedError (a ValueError) so Pydantic
   surfaces it as a 400 RequestValidationError.
2. Immediately before every HTTP delivery attempt inside _attempt_delivery() —
   creation-time validation is insufficient because DNS records can change after
   registration (TOCTOU risk).

Detection covers:
- Private ranges   (RFC 1918: 10/8, 172.16/12, 192.168/16)
- Loopback         (127.0.0.0/8, ::1)
- Link-local       (169.254/16 — AWS/GCP metadata endpoint lives here)
- Reserved ranges  (per ipaddress library definitions)
- Multicast        (224.0.0.0/4, ff00::/8)
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class SSRFBlockedError(ValueError):
    """Raised when a webhook URL resolves to a non-public address."""


def assert_public_url(url: str) -> None:
    """Resolve *url*'s hostname and reject any private/reserved/loopback IP.

    Raises :exc:`SSRFBlockedError` (a ``ValueError`` subclass) on violation.
    Because SSRFBlockedError is a ValueError, Pydantic field validators
    automatically convert it to a validation error message.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise SSRFBlockedError(f"Invalid webhook URL — no hostname found: {url!r}")

    try:
        results = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise SSRFBlockedError(
            f"Webhook URL hostname {hostname!r} could not be resolved: {exc}"
        ) from exc

    for _family, _type, _proto, _canonname, sockaddr in results:
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise SSRFBlockedError(
                f"Webhook URL {hostname!r} resolves to a blocked address: {addr}"
            )
