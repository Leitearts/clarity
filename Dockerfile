# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN groupadd -r clarity && useradd -r -g clarity clarity

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY data/ ./data/

RUN mkdir -p /app/logs && chown -R clarity:clarity /app/logs

USER clarity

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" \
    || exit 1

ENV CLARITY_DEBUG=false \
    CLARITY_AUDIT_LOG_PATH=/app/logs/audit.jsonl \
    CLARITY_RISK_WEIGHT_PRESET=default

CMD ["sh", "-c", \
     "uvicorn app.main:app \
      --host 0.0.0.0 \
      --port 8000 \
      --workers ${WORKERS:-4} \
      --log-level ${LOG_LEVEL:-info} \
      --no-access-log"]
