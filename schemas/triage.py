"""Triage agent output — qualificação de lead (POP rastreado, identidade da Rosa)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TriageOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    reply: list[str] = Field(
        default_factory=list,
        description="Mensagens a enviar (uma ou mais — split natural em frases).",
    )
    qualification_state_patch: dict[str, Any] = Field(
        default_factory=dict,
        description="Campos a actualizar em bot_sessions.qualification_state (merge).",
    )
    pop_step: int | None = Field(
        default=None,
        description="Novo step do POP rastreado (0–10), se houver avanço.",
    )
    transfer: bool = Field(default=False)
    transfer_reason: str | None = None
