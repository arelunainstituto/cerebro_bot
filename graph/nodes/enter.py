"""enter — primeiro nó. Adquire lock, carrega sessão e turnos, regista user.

Sequência:
  1. Calcula `session_id` (`phone_id:contact_phone`) se vazio.
  2. Tenta lock atómico em `bot_locks` (uma tentativa, TTL 10s).
     Se ocupado, lança `LockBusyError` → endpoint mapeia para HTTP 423.
  3. Carrega `qualification_state` e `state_extras` de `bot_sessions`.
  4. Materializa `short_memory` a partir dos últimos 12 turnos de `bot_turns`
     (autoridade única; coluna JSONB `bot_sessions.short_memory` é ignorada).
  5. Regista o turno do user em `bot_turns` (paridade `api/bot/graph.js:327-328`).
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from core.database import (
    insert_turn,
    load_recent_turns,
    load_session,
    try_acquire_lock,
)
from core.errors import LockBusyError
from graph.state import BotState, QualificationState, Turn

log = logging.getLogger(__name__)

_LOCK_TTL_SECONDS = 10
_MEMORY_LIMIT = 12


async def enter_node(state: BotState, config: RunnableConfig) -> BotState:
    pool = config["configurable"]["pool"]

    if not state.session_id:
        state.session_id = f"{state.phone_id}:{state.contact_phone}"

    # 1) Lock atómico (sem retry — gateway retransmite se 423)
    acquired = await try_acquire_lock(
        pool, state.phone_id, state.contact_phone, ttl_seconds=_LOCK_TTL_SECONDS
    )
    if not acquired:
        raise LockBusyError(state.session_id)

    # 2) Carregar sessão (defaults se inexistente — sem INSERT)
    sess = await load_session(pool, state.phone_id, state.contact_phone)
    state.qualification_state = QualificationState.model_validate(sess.qualification_state)
    state.state_extras = sess.state
    state.current_namespace = sess.current_namespace

    # 3) Memória curta dos últimos 12 turnos
    turns = await load_recent_turns(
        pool, state.phone_id, state.contact_phone, limit=_MEMORY_LIMIT
    )
    state.short_memory = [Turn(role=t["role"], content=t["content"]) for t in turns]

    # 4) Auditoria do turno do user (antes de qualquer decisão do grafo)
    await insert_turn(
        pool,
        state.phone_id,
        state.contact_phone,
        role="user",
        content=state.incoming_message,
    )

    log.info(
        "[enter] session=%s memory=%d qual_keys=%d",
        state.session_id,
        len(state.short_memory),
        len(state.qualification_state.model_dump(exclude_none=True)),
    )
    return state
