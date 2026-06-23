#!/usr/bin/env bash
# Builds the on-premise deliverable: a .tar.gz containing exactly
# docker-compose.yml + .env.example + /scripts/ + /docs/on-premise/, per the
# on-premise packaging ticket. No source code — the images it references are
# pulled from Artifact Registry (APP_IMAGE/NGINX_IMAGE in .env.example).
#
# Usage:
#   ./scripts/package-onprem.sh [version]
#   ./scripts/package-onprem.sh 1.4.0   # -> dist/tgs-agent-onprem-1.4.0.tar.gz

set -euo pipefail

cd "$(dirname "$0")/.."

VERSION="${1:-$(date +%Y%m%d%H%M%S)}"
STAGE_NAME="tgs-agent-onprem-${VERSION}"
DIST_DIR="dist"
STAGE_DIR="${DIST_DIR}/${STAGE_NAME}"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

cp docker-compose.yml "$STAGE_DIR/"
cp .env.example "$STAGE_DIR/"

# Only the on-premise-relevant script ships — scripts/ also holds unrelated
# internal dev/ops tooling (KB ingestion, Jira provisioning, RAG eval, ...)
# that has no place in a customer-facing deliverable.
mkdir -p "$STAGE_DIR/scripts"
cp scripts/health-check.sh "$STAGE_DIR/scripts/"

mkdir -p "$STAGE_DIR/docs"
cp -R docs/on-premise "$STAGE_DIR/docs/on-premise"

mkdir -p "$DIST_DIR"
tar -czf "${DIST_DIR}/${STAGE_NAME}.tar.gz" -C "$DIST_DIR" "$STAGE_NAME"
rm -rf "$STAGE_DIR"

echo "Built ${DIST_DIR}/${STAGE_NAME}.tar.gz"
tar -tzf "${DIST_DIR}/${STAGE_NAME}.tar.gz"
