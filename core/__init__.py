"""Camada de infraestrutura: pool asyncpg, queries SQL, LLM e excepções."""

from .calendar import create_gcal_event
from .database import (
    book_slot_db,
    create_pool,
    fetch_available_slots,
    fetch_bot_kb,
    hold_slot,
    insert_turn,
    load_opt_out,
    load_recent_turns,
    load_session,
    release_hold,
    release_lock,
    try_acquire_lock,
    upsert_opt_out,
    upsert_session,
)
from .embeddings import get_hf_embedding
from .errors import DBError, LockBusyError, SlotUnavailableError
from .llm import get_llm, invoke_structured

__all__ = [
    "create_pool",
    "try_acquire_lock",
    "release_lock",
    "load_session",
    "upsert_session",
    "load_recent_turns",
    "insert_turn",
    "load_opt_out",
    "upsert_opt_out",
    "fetch_bot_kb",
    "fetch_available_slots",
    "hold_slot",
    "release_hold",
    "book_slot_db",
    "create_gcal_event",
    "get_hf_embedding",
    "get_llm",
    "invoke_structured",
    "LockBusyError",
    "DBError",
    "SlotUnavailableError",
]
