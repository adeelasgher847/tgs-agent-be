"""
Production entrypoint (see Dockerfile). Not used for local dev — the
documented dev command (`uvicorn app.main:app --reload`) runs uvicorn
directly and is unaffected by anything here.

Enables ProxyHeadersMiddleware so `request.url.scheme` (and therefore any
redirect Location header, SAML/OIDC callback URLs, etc.) reflects the
original client-facing scheme/host from X-Forwarded-* rather than the
scheme of the local connection from the reverse proxy. FORWARDED_ALLOW_IPS
controls which peer IPs are trusted to set those headers; default is
loopback-only, since only a same-host reverse proxy should ever connect
directly to this process. Deployments where the proxy connects from a
different, still-trusted address (e.g. FORWARDED_ALLOW_IPS="*" when the
port is otherwise firewalled to only that proxy) must opt in explicitly via
env var — see docker-compose.yml.
"""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8001")),
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )


if __name__ == "__main__":
    main()
