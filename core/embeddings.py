"""Embeddings via HuggingFace Inference API (httpx, sem torch local).

Modelo default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(384d) — mesmo usado no JS para indexar `bot_kb_documents`. Override via
env `HF_EMBEDDING_MODEL`.

API (paridade api/bot/_lib/embeddings.js): HF migrou em 2024 de
`api-inference.huggingface.co` (deprecado) para o gateway Inference Providers.
URL: POST https://router.huggingface.co/hf-inference/models/<model>/pipeline/feature-extraction
Header: `Authorization: Bearer <HF_API_TOKEN>`
Body:   `{"inputs": "<text>", "options": {"wait_for_model": true}}`
Return: list[float] (pooled) ou list[list[float]] (token-level) — normalizamos.

Cold-start handling:
  - `options.wait_for_model=true` faz o HF bloquear até o modelo estar quente
    (até ~20s). Evita o 503 com `estimated_time`.
  - Mesmo assim, fazemos 1 retry com backoff em falha transiente (503/504/429).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_BASE = "https://router.huggingface.co/hf-inference/models"
_TIMEOUT = 30.0
_RETRY_DELAY = 3.0


def _model() -> str:
    return os.getenv("HF_EMBEDDING_MODEL", _DEFAULT_MODEL)


def _token() -> str:
    tok = os.getenv("HF_API_TOKEN")
    if not tok:
        raise RuntimeError("HF_API_TOKEN não está definido no ambiente")
    return tok


def _normalize_vector(payload: Any) -> list[float]:
    """HF devolve list[float] OU list[list[float]] (token-level).

    Para sentence-transformers via pipeline, normalmente vem já pooled
    (list[float]). Defesa: se for nested, tomamos a média element-wise dos
    tokens (mean pooling) — equivalente ao que o transformer faz internamente.
    """
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"HF embedding payload inválido: {type(payload).__name__}")

    if isinstance(payload[0], (int, float)):
        return [float(x) for x in payload]

    if isinstance(payload[0], list):
        # Token-level: mean pooling
        rows = [[float(x) for x in row] for row in payload]
        if not rows:
            raise ValueError("HF embedding payload vazio (nested)")
        dim = len(rows[0])
        sums = [0.0] * dim
        for row in rows:
            if len(row) != dim:
                raise ValueError(f"HF embedding inconsistent dims: {len(row)} vs {dim}")
            for i, v in enumerate(row):
                sums[i] += v
        n = len(rows)
        return [s / n for s in sums]

    raise ValueError(f"HF embedding payload inesperado: first={type(payload[0]).__name__}")


async def get_hf_embedding(text: str, *, model: str | None = None) -> list[float]:
    """Devolve o vector 384d (`paraphrase-multilingual-MiniLM-L12-v2`).

    Retry 1x em 503/504/429 (cold start). Levanta `RuntimeError` se falhar
    definitivamente — o specialist trata como `kb_used=0` + reply de fallback.
    """
    if not text or not text.strip():
        raise ValueError("get_hf_embedding: texto vazio")

    url = f"{_BASE}/{model or _model()}/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Content-Type": "application/json",
    }
    body = {"inputs": text.strip(), "options": {"wait_for_model": True}}

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in (1, 2):
            try:
                r = await client.post(url, headers=headers, json=body)
            except httpx.RequestError as e:
                log.warning("[embed] attempt=%d network error: %s", attempt, e)
                if attempt == 2:
                    raise RuntimeError(f"HF embedding falhou (network): {e}") from e
                await asyncio.sleep(_RETRY_DELAY)
                continue

            if r.status_code == 200:
                try:
                    return _normalize_vector(r.json())
                except Exception as e:
                    raise RuntimeError(f"HF embedding payload inválido: {e}") from e

            # Cold start ou rate-limit: retry curto
            if r.status_code in (503, 504, 429) and attempt == 1:
                log.warning("[embed] attempt=%d transient status=%d body=%s — retry", attempt, r.status_code, r.text[:200])
                await asyncio.sleep(_RETRY_DELAY)
                continue

            raise RuntimeError(f"HF embedding HTTP {r.status_code}: {r.text[:300]}")

    raise RuntimeError("HF embedding: esgotou retries (não deveria chegar aqui)")
