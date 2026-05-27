"""Router agent output — decide qual nó terminal lida com o turno."""

from typing import Literal

from pydantic import BaseModel, Field


AgentName = Literal["triage", "specialist", "scheduling"]


class RouterDecision(BaseModel):
    agent: AgentName = Field(
        ...,
        description="Nó que deve processar o turno: triage (qualificar), specialist (RAG/dúvidas), scheduling (marcar).",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(..., description="Justificação curta em PT-PT.")
