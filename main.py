"""
Cérebro Python — FastAPI entrypoint.

Expõe um único endpoint conversacional (POST /v1/chat/invoke) que será
consumido pelo Gateway (Vercel/Node.js) via HTTP. Esta fase entrega o
esqueleto: validação Pydantic + healthcheck + stub. A invocação real do
StateGraph entra nas fases seguintes.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core import LockBusyError, create_pool, release_lock
from graph import BotState, build_graph

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("cerebro")


# -------- Pydantic models ---------------------------------------------------


class ChatInvokeRequest(BaseModel):
    phone_id: str = Field(..., description="WABA phone_number_id (origem do bot)")
    contact_phone: str = Field(..., description="MSISDN do lead (E.164 sem '+')")
    message: str = Field(..., description="Texto agregado do lead após debounce")
    channel: str = Field(default="whatsapp", description="Canal de origem")
    context_id: str | None = Field(
        default=None,
        description="wamid da mensagem citada (ex.: template original do CRM)",
    )
    session_meta: dict[str, Any] | None = Field(
        default=None,
        description="Metadados extra opcionais (gateway pode passar nome do lead, etc.)",
    )


class ChatInvokeResponse(BaseModel):
    reply: list[str] = Field(
        default_factory=list,
        description="Mensagens a enviar (uma ou mais — split natural em frases)",
    )
    transfer: bool = Field(default=False, description="Marcar para handoff humano")
    transfer_reason: str | None = Field(default=None)
    qualification_state: dict[str, Any] = Field(default_factory=dict)
    pop_step: int = Field(default=0, description="Step actual do POP rastreado (0–10)")
    agent_used: str | None = Field(
        default=None,
        description="Nó terminal que produziu a resposta (triage|specialist|scheduling|bypass|...)",
    )


# -------- Lifespan (pool DB futuro) ----------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup: cérebro_python a arrancar")
    app.state.pool = await create_pool()
    app.state.graph = build_graph()
    app.state.ready = True
    log.info("startup: pool asyncpg + StateGraph prontos")
    try:
        yield
    finally:
        log.info("shutdown: a fechar recursos")
        app.state.ready = False
        app.state.graph = None
        if app.state.pool is not None:
            await app.state.pool.close()
            app.state.pool = None
        log.info("shutdown: pool fechado")


# -------- App ---------------------------------------------------------------


app = FastAPI(
    title="Cérebro Conversacional (Instituto Areluna)",
    version="0.1.0",
    description="Backend isolado do bot. Body (WhatsApp/Vercel) chama este cérebro por HTTP.",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "ready": getattr(app.state, "ready", False)}


@app.post("/v1/chat/invoke", response_model=ChatInvokeResponse)
async def chat_invoke(req: ChatInvokeRequest) -> ChatInvokeResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message vazia")

    log.info(
        "invoke phone_id=%s contact=%s channel=%s context_id=%s msg_len=%d",
        req.phone_id,
        req.contact_phone,
        req.channel,
        req.context_id,
        len(req.message),
    )

    graph = getattr(app.state, "graph", None)
    pool = getattr(app.state, "pool", None)
    if graph is None or pool is None:
        raise HTTPException(status_code=503, detail="service not ready")

    initial = BotState(
        phone_id=req.phone_id,
        contact_phone=req.contact_phone,
        channel=req.channel,
        incoming_message=req.message,
        context_id=req.context_id,
    )

    # TODO (fase seguinte): adicionar "thread_id" para checkpoint nativo LangGraph.
    config = {"configurable": {"pool": pool}}

    try:
        final_raw = await graph.ainvoke(initial, config=config)
    except LockBusyError as exc:
        log.info("[invoke] lock busy session=%s", exc.session_id)
        raise HTTPException(
            status_code=423,
            detail={"code": "lock_busy", "session_id": exc.session_id, "retry_after_ms": 500},
        )
    except Exception as exc:
        # Best-effort: se enter_node já adquiriu o lock e um nó intermédio crashou,
        # respond_node não corre — o lock fica órfão até TTL. Forçamos release aqui.
        log.exception("[invoke] graph failed — releasing lock as safety: %s", exc)
        try:
            await release_lock(pool, req.phone_id, req.contact_phone)
        except Exception:
            log.exception("[invoke] safety release_lock também falhou")
        raise HTTPException(status_code=500, detail="graph_error")

    final = BotState.model_validate(final_raw)

    return ChatInvokeResponse(
        reply=final.reply,
        transfer=final.transfer,
        transfer_reason=final.transfer_reason,
        qualification_state=final.qualification_state.model_dump(exclude_none=True),
        pop_step=final.pop_step,
        agent_used=final.agent_used,
    )
