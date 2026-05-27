"""Wrapper LLM — espelha api/bot/_lib/llm.js.

Two helpers:
  - `get_llm(...)`            → `ChatOpenAI` cliente configurado.
  - `invoke_structured(...)`  → invocação async com output Pydantic forçado.

Provider-agnostic: usa `OPENAI_API_KEY` do ambiente. Se trocares de provider,
basta apontar `OPENAI_BASE_URL` ao endpoint compatível.

Retry minimal: 1 retry curto em timeout / empty json (paridade JS).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)


def get_llm(
    *,
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    timeout: int = 25,
) -> ChatOpenAI:
    """Constrói `ChatOpenAI` para um turno. API key vem de `OPENAI_API_KEY`.

    Override `OPENAI_BASE_URL` se quiseres apontar a um endpoint compatível
    (DeepSeek, Groq, Gemini OpenAI-compat).
    """
    effective_model = model or os.getenv("BOT_DEFAULT_MODEL", "gpt-4o-mini")
    kwargs: dict[str, Any] = {
        "model": effective_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _to_lc_messages(messages: list[dict[str, str]]) -> list:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content))
        else:
            out.append(HumanMessage(content))
    return out


async def invoke_structured(
    schema,
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 600,
    schema_name: str = "output",
):
    """Invoca o LLM forçando output válido pelo `schema` Pydantic.

    Em LangChain Python, `with_structured_output(schema, method="json_mode")`
    é o equivalente directo a `withStructuredOutput({method: 'jsonMode'})` do JS.
    """
    lc_messages = _to_lc_messages(messages)

    def _build(m: str | None, mt: int):
        llm = get_llm(model=m, temperature=temperature, max_tokens=mt)
        return llm.with_structured_output(schema, method="json_mode")

    structured = _build(model, max_tokens)
    try:
        return await structured.ainvoke(lc_messages)
    except Exception as e:
        msg = str(e).lower()
        is_timeout = "timeout" in msg or "timed out" in msg or "abort" in msg
        is_empty = "unexpected end of json" in msg or 'text: ""' in msg
        # 5xx do upstream OpenAI são transientes — vale a pena 1 retry.
        is_server = "500" in msg or "502" in msg or "503" in msg or "server_error" in msg or "internalservererror" in msg
        if not (is_timeout or is_empty or is_server):
            raise
        reason = "timeout" if is_timeout else ("empty_json" if is_empty else "server_5xx")
        log.warning("[llm] retry triggered schema=%s reason=%s", schema_name, reason)
        # Fallback estável: gpt-4o-mini com max_tokens menor
        retry_structured = _build("gpt-4o-mini", 800 if is_empty else 600)
        return await retry_structured.ainvoke(lc_messages)
