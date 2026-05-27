"""transfer_guard — sanidade antes de chamar o agente terminal.

Valida invariantes (POP coerente, qualificação mínima, sem flags de transfer pendentes).
Se algo estiver errado, marca state.transfer=True e o edge encaminha para respond.

TODO (fase seguinte): portar guards do `transferGuardNode` em `api/bot/graph.js`.
"""

from __future__ import annotations

import logging

from graph.state import BotState

log = logging.getLogger(__name__)


async def transfer_guard_node(state: BotState) -> BotState:
    log.info("[transfer_guard] (stub) transfer=%s next_agent=%s", state.transfer, state.next_agent)
    return state
