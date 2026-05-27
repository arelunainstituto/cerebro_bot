"""Intent classification — chamado em bypass para detectar atalhos (emergency, no_interest, ...)."""

from typing import Literal

from pydantic import BaseModel, Field


IntentType = Literal[
    "emergency",
    "human_request",
    "wrong_number",
    "no_interest",
    "already_patient",
    "reception_service",
    "new_lead",
    "unknown",
]


class IntentClassification(BaseModel):
    type: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(..., description="Justificação curta em PT-PT.")
