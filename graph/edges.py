"""Transições condicionais do StateGraph.

Cada função recebe BotState e devolve o nome do próximo nó (str).
LangGraph faz o mapping via dict no add_conditional_edges.
"""

from __future__ import annotations

from graph.state import BotState


def route_from_bypass(state: BotState) -> str:
    """Se o bypass marcou shortcut (bypass_reason ou transfer), salta para respond.
    Caso contrário, segue para o router para decidir o agente terminal.
    """
    if state.bypass_reason or state.transfer:
        return "respond"
    return "router"


def route_from_router(state: BotState) -> str:
    """Após o router, sempre passa pelo transfer_guard antes do agente terminal."""
    return "transfer_guard"


def route_from_guard(state: BotState) -> str:
    """Decide o agente terminal. Se guard marcou transfer, vai direto a respond."""
    if state.transfer:
        return "respond"
    agent = state.next_agent
    if agent in {"triage", "specialist", "scheduling"}:
        return agent
    # default seguro
    return "triage"
