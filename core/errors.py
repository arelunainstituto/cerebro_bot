"""Excepções de domínio do cérebro."""

from __future__ import annotations


class LockBusyError(Exception):
    """Outra invocação detém o lock activo (TTL não expirou).

    O endpoint FastAPI converte isto em HTTP 423 (Locked) para o gateway
    saber que deve retransmitir após o debounce.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"lock busy for session={session_id}")
        self.session_id = session_id


class DBError(Exception):
    """Falha de DB não esperada (envolve excepções asyncpg)."""


class SlotUnavailableError(Exception):
    """`slot_start` não está disponível para booking (race ou TTL expirado).

    Lançada por `book_slot_db` quando o UPDATE atómico em `bot_calendar_slots`
    afecta 0 linhas — outro lead apanhou o slot OU o hold caducou.
    """

    def __init__(self, slot_iso: str) -> None:
        super().__init__(f"slot indisponível: {slot_iso}")
        self.slot_iso = slot_iso
