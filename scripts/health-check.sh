#!/usr/bin/env bash
# Checks every service's health endpoint and exits non-zero if any is
# unhealthy. Two layers: (1) each service's own Docker Compose `healthcheck:`
# status (covers postgres/redis/livekit/worker, which have no public HTTP
# health route of their own), and (2) an end-to-end HTTP smoke check through
# nginx-proxy, which is what actually proves the whole request path works.
#
# Usage:
#   ./scripts/health-check.sh
#   HEALTH_CHECK_BASE_URL=https://my-host ./scripts/health-check.sh

set -uo pipefail

cd "$(dirname "$0")/.."

SERVICES=(postgres redis livekit fastapi-v2 worker nginx-proxy)
BASE_URL="${HEALTH_CHECK_BASE_URL:-https://localhost}"
FAILED=0

echo "== Docker Compose service health =="
for svc in "${SERVICES[@]}"; do
    cid="$(docker compose -f docker-compose.yml ps -q "$svc" 2>/dev/null)"
    if [ -z "$cid" ]; then
        echo "FAIL  $svc: container not running"
        FAILED=1
        continue
    fi

    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid" 2>/dev/null)"
    if [ "$status" = "healthy" ] || [ "$status" = "running" ]; then
        echo "OK    $svc: $status"
    else
        echo "FAIL  $svc: ${status:-unknown}"
        FAILED=1
    fi
done

echo
echo "== HTTP smoke checks (through nginx-proxy at $BASE_URL) =="

check_url() {
    local name="$1" url="$2"
    if curl -fsSk --max-time 5 "$url" >/dev/null 2>&1; then
        echo "OK    $name ($url)"
    else
        echo "FAIL  $name ($url)"
        FAILED=1
    fi
}

check_url "nginx"          "$BASE_URL/nginx-health"
check_url "fastapi-v2 API" "$BASE_URL/api/v2/health"
check_url "app /health"    "$BASE_URL/health"

echo
if [ "$FAILED" -ne 0 ]; then
    echo "RESULT: one or more services are unhealthy."
    exit 1
fi

echo "RESULT: all services healthy."
exit 0
