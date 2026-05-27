"""Specialist agent output — responde dúvidas com base no RAG da clínica."""

from pydantic import BaseModel, ConfigDict, Field


class SpecialistOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    reply: list[str] = Field(default_factory=list)
    kb_used: int = Field(
        default=0,
        description="Número de chunks RAG efectivamente citados na resposta.",
    )
    transfer: bool = Field(default=False, description="True se o specialist não conseguir responder.")
    transfer_reason: str | None = None
