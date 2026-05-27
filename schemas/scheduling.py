"""Scheduling agent output — propõe/confirma slot e (via tool) reserva no Google Calendar."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SchedulingOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    reply: list[str] = Field(default_factory=list)
    action: Literal["propose", "confirm", "booked", "nothing"] = Field(
        default="nothing",
        description="propose=oferecer slots; confirm=pedir confirmação; booked=já marcado; nothing=conversa lateral.",
    )
    slot_iso: str | None = Field(
        default=None,
        description="Slot proposto/confirmado em ISO 8601 com timezone Europe/Lisbon.",
    )
    calendar_event_id: str | None = Field(
        default=None,
        description="ID do evento no Google Calendar (preenchido após booking real).",
    )
    transfer: bool = Field(default=False)
    transfer_reason: str | None = None
