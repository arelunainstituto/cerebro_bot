"""BotState — espelho do State JS actual (`api/bot/_state.js`).

Mantém-se compatível com qualification_state heterogéneo do Supabase (extra='allow'),
para que rows existentes em `bot_sessions` carreguem sem fricção quando ligarmos o DB.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QualificationState(BaseModel):
    """Campos conhecidos do JS; permite extras (CRM / templates podem injectar mais)."""

    model_config = ConfigDict(extra="allow")

    nome: str | None = None
    area_interesse: str | None = None  # implantes|alinhador|faceta|capilar|estetica|...
    ja_e_paciente: str | None = None   # "sim" | "não"
    motivacao: str | None = None
    cidade: str | None = None
    pais: str | None = None
    melhor_horario: str | None = None
    template_origem: str | None = None  # nome do template inferido via context.id


class BotState(BaseModel):
    """Estado completo do grafo. Persistido em `bot_sessions` ao fim do turno."""

    model_config = ConfigDict(extra="allow")

    # --- Input do turno ---
    phone_id: str
    contact_phone: str
    channel: str = "whatsapp"
    incoming_message: str
    context_id: str | None = None

    # --- Identidade da sessão ---
    session_id: str = ""  # f"{phone_id}:{contact_phone}" (preenchido em enter)

    # --- Memória curta + qualificação ---
    short_memory: list[Turn] = Field(default_factory=list)
    qualification_state: QualificationState = Field(default_factory=QualificationState)

    # --- POP rastreado ---
    pop_step: int = 0
    step_history: list[dict[str, Any]] = Field(default_factory=list)

    # --- Flags internas underscore-prefixed (espelha bot_sessions.state JSONB) ---
    # Ex.: _pending_confirm, _proposed_slots, _slots_fetched_at, _cases_sent,
    # _step_history. Mantém-se separado do qualification_state por convenção JS.
    state_extras: dict[str, Any] = Field(default_factory=dict)

    # --- Decisões de routing (preenchidas durante o grafo) ---
    next_agent: str | None = None     # "triage" | "specialist" | "scheduling"
    bypass_reason: str | None = None  # se != None, salta para respond
    agent_used: str | None = None     # quem efectivamente respondeu (telemetria)
    current_namespace: str | None = None  # namespace activo de RAG (specialist)

    # --- Output do turno ---
    reply: list[str] = Field(default_factory=list)
    transfer: bool = False
    transfer_reason: str | None = None

    # --- Diagnóstico (opcional) ---
    debug: dict[str, Any] = Field(default_factory=dict)
