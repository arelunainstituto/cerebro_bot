"""Acesso ao Supabase Postgres via asyncpg.

Todas as queries são SQL puro. Mantemos paridade semântica com o JS em
`api/bot/_lib/state.js`, com **uma divergência deliberada**: o `short_memory`
é materializado a partir de `bot_turns` (não da coluna JSONB
`bot_sessions.short_memory`, que fica `[]`).

Padrões importantes:
- Pool criado com `init` callback que regista codec JSONB ↔ dict.
- `try_acquire_lock` faz UPSERT atómico (uma tentativa, sem retry — o gateway
  retransmite se receber 423).
- `release_lock` faz UPDATE para `locked_until = NOW()` (não DELETE) — mantém
  a row para reuso futuro.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import asyncpg


# ----------------------------------------------------------------------------
# Pool
# ----------------------------------------------------------------------------


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Codec JSONB → dict/list (sem isto, asyncpg devolve string crua)."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def create_pool(dsn: str | None = None) -> asyncpg.Pool:
    """Cria o pool de ligações. Lê `DATABASE_URL` se `dsn` não for fornecido.

    O DSN deve apontar ao Connection Pooler do Supabase (porta 6543, modo
    transaction). Min/max ajustáveis via `DB_POOL_MIN` / `DB_POOL_MAX`.

    `statement_cache_size=0` é obrigatório com PgBouncer em modo `transaction`
    (o caso do Supabase pooler): o pooler reaproveita conexões entre
    transacções diferentes, e os prepared statements nomeados não sobrevivem
    a esse switch — sem isto, há collision `__asyncpg_stmt_N__ already exists`.
    """
    dsn = dsn or os.environ["DATABASE_URL"]
    return await asyncpg.create_pool(
        dsn,
        min_size=int(os.getenv("DB_POOL_MIN", "1")),
        max_size=int(os.getenv("DB_POOL_MAX", "10")),
        command_timeout=10,
        statement_cache_size=0,
        init=_init_connection,
    )


# ----------------------------------------------------------------------------
# Locks
# ----------------------------------------------------------------------------


_ACQUIRE_LOCK_SQL = """
INSERT INTO bot_locks (phone_id, contact_phone, locked_until)
VALUES ($1, $2, NOW() + ($3 || ' seconds')::interval)
ON CONFLICT (phone_id, contact_phone) DO UPDATE
   SET locked_until = NOW() + ($3 || ' seconds')::interval
   WHERE bot_locks.locked_until <= NOW()
RETURNING phone_id;
"""

_RELEASE_LOCK_SQL = """
UPDATE bot_locks
SET locked_until = NOW()
WHERE phone_id = $1 AND contact_phone = $2;
"""


async def try_acquire_lock(
    pool: asyncpg.Pool,
    phone_id: str,
    contact_phone: str,
    ttl_seconds: int = 10,
) -> bool:
    """Tenta adquirir o lock atomicamente. Devolve True se obtido, False se ocupado."""
    row = await pool.fetchrow(_ACQUIRE_LOCK_SQL, phone_id, contact_phone, str(ttl_seconds))
    return row is not None


async def release_lock(pool: asyncpg.Pool, phone_id: str, contact_phone: str) -> None:
    """Liberta o lock (UPDATE locked_until = NOW(); preserva row)."""
    await pool.execute(_RELEASE_LOCK_SQL, phone_id, contact_phone)


# ----------------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------------


@dataclass
class SessionRow:
    qualification_state: dict[str, Any]
    state: dict[str, Any]
    current_namespace: str | None
    last_agent: str | None


_LOAD_SESSION_SQL = """
SELECT qualification_state, state, current_namespace, last_agent
FROM bot_sessions
WHERE phone_id = $1 AND contact_phone = $2
LIMIT 1;
"""

_UPSERT_SESSION_SQL = """
INSERT INTO bot_sessions
  (phone_id, contact_phone, qualification_state, state, current_namespace, last_agent, short_memory)
VALUES ($1, $2, $3, $4, $5, $6, '[]'::jsonb)
ON CONFLICT (phone_id, contact_phone) DO UPDATE SET
  qualification_state = EXCLUDED.qualification_state,
  state               = EXCLUDED.state,
  current_namespace   = EXCLUDED.current_namespace,
  last_agent          = EXCLUDED.last_agent;
"""


async def load_session(pool: asyncpg.Pool, phone_id: str, contact_phone: str) -> SessionRow:
    """Carrega a sessão. Se não existir, devolve defaults (sem INSERT)."""
    row = await pool.fetchrow(_LOAD_SESSION_SQL, phone_id, contact_phone)
    if row is None:
        return SessionRow(
            qualification_state={},
            state={},
            current_namespace=None,
            last_agent=None,
        )
    return SessionRow(
        qualification_state=row["qualification_state"] or {},
        state=row["state"] or {},
        current_namespace=row["current_namespace"],
        last_agent=row["last_agent"],
    )


async def upsert_session(
    pool: asyncpg.Pool,
    phone_id: str,
    contact_phone: str,
    *,
    qualification_state: dict[str, Any],
    state: dict[str, Any],
    current_namespace: str | None,
    last_agent: str | None,
) -> None:
    """UPSERT atómico. NÃO escreve `short_memory` — autoridade é `bot_turns`."""
    await pool.execute(
        _UPSERT_SESSION_SQL,
        phone_id,
        contact_phone,
        qualification_state or {},
        state or {},
        current_namespace,
        last_agent,
    )


# ----------------------------------------------------------------------------
# Turns
# ----------------------------------------------------------------------------


_LOAD_TURNS_SQL = """
SELECT role, content, ts
FROM bot_turns
WHERE phone_id = $1 AND contact_phone = $2
  AND role IN ('user', 'assistant')
  AND content IS NOT NULL
ORDER BY ts DESC
LIMIT $3;
"""

_INSERT_TURN_SQL = """
INSERT INTO bot_turns
  (phone_id, contact_phone, role, content, agent_used, tools_called, latency_ms)
VALUES ($1, $2, $3, $4, $5, $6, $7);
"""

# Paridade JS (`api/bot/_lib/state.js:482`): trunca content a 4000 chars.
_MAX_CONTENT_CHARS = 4000


async def load_recent_turns(
    pool: asyncpg.Pool,
    phone_id: str,
    contact_phone: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Últimos N turnos (DESC), devolvidos em ordem cronológica (reversed).

    Exclui roles `system`/`tool` — ruído para a memória do agente.
    """
    rows = await pool.fetch(_LOAD_TURNS_SQL, phone_id, contact_phone, limit)
    # rows vêm em ts DESC; reverter para ASC (cronológico).
    return [
        {"role": r["role"], "content": r["content"], "ts": r["ts"]}
        for r in reversed(rows)
    ]


async def insert_turn(
    pool: asyncpg.Pool,
    phone_id: str,
    contact_phone: str,
    *,
    role: str,
    content: str | None,
    agent_used: str | None = None,
    tools_called: dict[str, Any] | None = None,
    latency_ms: int | None = None,
) -> None:
    """Insere um turno append-only em `bot_turns`."""
    safe_content = content[:_MAX_CONTENT_CHARS] if content else None
    await pool.execute(
        _INSERT_TURN_SQL,
        phone_id,
        contact_phone,
        role,
        safe_content,
        agent_used,
        tools_called,
        latency_ms,
    )


# ----------------------------------------------------------------------------
# Opt-outs (lista global por contact_phone, não por phone_id)
# ----------------------------------------------------------------------------


_LOAD_OPT_OUT_SQL = """
SELECT opt_out_type
FROM bot_opt_outs
WHERE contact_phone = $1
LIMIT 1;
"""

_UPSERT_OPT_OUT_SQL = """
INSERT INTO bot_opt_outs (contact_phone, opt_out_type, reason, matched_text, source)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (contact_phone) DO UPDATE SET
  opt_out_type = EXCLUDED.opt_out_type,
  reason       = EXCLUDED.reason,
  matched_text = EXCLUDED.matched_text,
  source       = EXCLUDED.source,
  updated_at   = NOW();
"""


def _digits(s: str | None) -> str:
    """Normaliza para apenas dígitos. Paridade JS: `String(x).replace(/\\D/g, '')`."""
    if not s:
        return ""
    return "".join(c for c in s if c.isdigit())


async def load_opt_out(pool: asyncpg.Pool, contact_phone: str) -> str | None:
    """Devolve `opt_out_type` (`wrong_number` | `no_interest` | `already_patient`) ou `None`."""
    row = await pool.fetchrow(_LOAD_OPT_OUT_SQL, _digits(contact_phone))
    return row["opt_out_type"] if row else None


async def upsert_opt_out(
    pool: asyncpg.Pool,
    contact_phone: str,
    opt_out_type: str,
    *,
    reason: str | None = None,
    matched_text: str | None = None,
    source: str = "bot",
) -> None:
    """UPSERT em `bot_opt_outs`. Persistência usada em fases seguintes."""
    await pool.execute(
        _UPSERT_OPT_OUT_SQL,
        _digits(contact_phone),
        opt_out_type,
        reason,
        matched_text,
        source,
    )


# ----------------------------------------------------------------------------
# RAG — pgvector retrieval via RPC `match_bot_kb`
# ----------------------------------------------------------------------------


_MATCH_KB_SQL = """
SELECT id, content, source, metadata, similarity
FROM match_bot_kb($1::vector, $2, $3, $4);
"""


def _vector_literal(embedding: list[float]) -> str:
    """Serializa vector Python → literal pgvector `[v1,v2,…]`.

    Necessário porque o `statement_cache_size=0` (PgBouncer transaction mode)
    desactiva prepared statements e o codec automático de pgvector. Passamos
    como string + cast `::vector`.
    """
    if not embedding:
        raise ValueError("vector embedding vazio")
    # Float bruto sem notação científica para evitar parsing edge-cases no PG.
    return "[" + ",".join(format(float(x), ".7g") for x in embedding) + "]"


async def fetch_bot_kb(
    pool: asyncpg.Pool,
    embedding: list[float],
    phone_id: str,
    namespace: str,
    match_count: int = 4,
) -> list[dict[str, Any]]:
    """Top-K chunks por similaridade cosine. Devolve lista de dicts.

    Cada dict: `id`, `content`, `source`, `metadata`, `similarity`.
    """
    vec = _vector_literal(embedding)
    rows = await pool.fetch(_MATCH_KB_SQL, vec, phone_id, namespace, match_count)
    return [
        {
            "id": r["id"],
            "content": r["content"],
            "source": r["source"],
            "metadata": r["metadata"] or {},
            "similarity": float(r["similarity"]),
        }
        for r in rows
    ]


# ----------------------------------------------------------------------------
# Calendar — state machine de bot_calendar_slots + bot_appointments
# ----------------------------------------------------------------------------


_FETCH_SLOTS_SQL = """
SELECT slot_start, slot_end
FROM bot_calendar_slots
WHERE status = 'available'
  AND slot_start > NOW()
  AND slot_start < NOW() + ($1 || ' days')::interval
ORDER BY slot_start
LIMIT $2;
"""

_HOLD_SLOT_SQL = """
UPDATE bot_calendar_slots
SET status = 'held',
    reserved_by_phone = $2,
    reserved_until = NOW() + ($3 || ' minutes')::interval
WHERE slot_start = $1
  AND status = 'available'
RETURNING slot_start;
"""

_RELEASE_HOLD_SQL = """
UPDATE bot_calendar_slots
SET status = 'available',
    reserved_by_phone = NULL,
    reserved_until = NULL
WHERE slot_start = $1
  AND status = 'held'
  AND reserved_by_phone = $2
RETURNING slot_start;
"""

_BOOK_UPDATE_SLOT_SQL = """
UPDATE bot_calendar_slots
SET status = 'booked', gcal_event_id = $2
WHERE slot_start = $1
  AND status IN ('held', 'available')
RETURNING slot_end;
"""

_BOOK_INSERT_APPOINTMENT_SQL = """
INSERT INTO bot_appointments
  (phone_id, contact_phone, contact_name, treatment,
   slot_start, slot_end, gcal_event_id, status)
VALUES ($1, $2, $3, $4, $5, $6, $7, 'scheduled')
RETURNING id;
"""


def _parse_iso(iso: str):
    """Converte ISO 8601 → `datetime` aware (asyncpg precisa de datetime, não string)."""
    from datetime import datetime
    return datetime.fromisoformat(iso)


async def fetch_available_slots(
    pool: asyncpg.Pool,
    days: int = 7,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Slots `available` nos próximos N dias. Devolve lista ordenada por hora.

    Cada item: `{slot_start_dt, slot_start_iso, slot_end_dt, slot_end_iso}`.
    `*_dt` são `datetime` aware (com tz da DB).
    `*_iso` são strings ISO 8601 com tz (formato usado para passar de volta
    como argumento para `$1::timestamptz`).
    """
    rows = await pool.fetch(_FETCH_SLOTS_SQL, str(days), limit)
    return [
        {
            "slot_start_dt": r["slot_start"],
            "slot_start_iso": r["slot_start"].isoformat(),
            "slot_end_dt": r["slot_end"],
            "slot_end_iso": r["slot_end"].isoformat() if r["slot_end"] else None,
        }
        for r in rows
    ]


async def hold_slot(
    pool: asyncpg.Pool,
    slot_start_iso: str,
    phone_id: str,
    ttl_minutes: int = 30,
) -> bool:
    """Tenta prender slot atomicamente. True se conseguiu, False se outro foi mais rápido."""
    row = await pool.fetchrow(_HOLD_SLOT_SQL, _parse_iso(slot_start_iso), phone_id, str(ttl_minutes))
    return row is not None


async def release_hold(
    pool: asyncpg.Pool,
    slot_start_iso: str,
    phone_id: str,
) -> bool:
    """Liberta hold próprio. True se libertou, False se não era nosso ou já não está held."""
    row = await pool.fetchrow(_RELEASE_HOLD_SQL, _parse_iso(slot_start_iso), phone_id)
    return row is not None


async def book_slot_db(
    pool: asyncpg.Pool,
    *,
    slot_start_iso: str,
    phone_id: str,
    contact_phone: str,
    contact_name: str,
    treatment: str,
    gcal_event_id: str,
) -> int:
    """Booking atómico (UPDATE slot → 'booked' + INSERT appointment).

    Lança `SlotUnavailableError` se o slot não estava em `held`/`available`
    no momento do UPDATE — outro lead apanhou-o ou TTL expirou.

    Devolve o `id` (`BIGSERIAL`) do appointment criado.

    O unique partial index em `bot_appointments` (status='scheduled') protege
    contra duplo booking ao nível do PG — ConstraintViolation é propagada se
    raça acontecer entre dois turnos paralelos.
    """
    from .errors import SlotUnavailableError  # import tardio evita ciclo

    slot_start_dt = _parse_iso(slot_start_iso)
    async with pool.acquire() as conn:
        async with conn.transaction():
            slot_end = await conn.fetchval(
                _BOOK_UPDATE_SLOT_SQL,
                slot_start_dt,
                gcal_event_id,
            )
            if slot_end is None:
                raise SlotUnavailableError(slot_start_iso)
            appt_id = await conn.fetchval(
                _BOOK_INSERT_APPOINTMENT_SQL,
                phone_id,
                contact_phone,
                contact_name,
                treatment,
                slot_start_dt,
                slot_end,
                gcal_event_id,
            )
    return int(appt_id)
