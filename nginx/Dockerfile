# Thin custom layer on top of the official nginx image: bakes in our proxy
# config plus a self-signed TLS cert generated at build time (per ticket's
# technical notes — "use self-signed cert in the Compose file"). To replace
# with a real certificate, see docs/on-premise/README.md.
#
# nginx/postgres/redis/livekit stay on their official upstream images in this
# stack (only the FastAPI app has custom Python code, which is why it alone
# gets the full multi-stage build -> distroless treatment in ../Dockerfile).

FROM alpine:3.20 AS cert-builder

RUN apk add --no-cache openssl

ARG NGINX_SERVER_NAME=localhost

RUN mkdir -p /ssl && \
    openssl req -x509 -nodes -newkey rsa:2048 -days 825 \
        -keyout /ssl/self-signed.key \
        -out /ssl/self-signed.crt \
        -subj "/CN=${NGINX_SERVER_NAME}" \
        -addext "subjectAltName=DNS:${NGINX_SERVER_NAME},DNS:localhost,IP:127.0.0.1"

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=cert-builder /ssl/self-signed.crt /etc/nginx/ssl/self-signed.crt
COPY --from=cert-builder /ssl/self-signed.key /etc/nginx/ssl/self-signed.key

RUN chmod 600 /etc/nginx/ssl/self-signed.key
