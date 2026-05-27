# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Cérebro Python — FastAPI + LangGraph
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Performance + logs limpos
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Deps de sistema necessárias por algumas wheels (cffi, cryptography, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Camada de deps cacheada: muda só quando requirements.txt muda
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Código da aplicação
COPY . .

# Porto interno do uvicorn. O EasyPanel injecta $PORT — usamos esse no CMD.
EXPOSE 8001

# Healthcheck nativo Docker (independente do EasyPanel)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8001}/healthz" || exit 1

# Em runtime, EasyPanel define $PORT (e Vercel, Render, etc.). Default 8001.
# `sh -c` permite expansão de $PORT — não use exec form aqui.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}"]
