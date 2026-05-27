"""StateGraph: 8 nós + transições condicionais. Espelho do `api/bot/graph.js`.

Topologia:
    START → enter → bypass ─┬─► (bypass_reason|transfer) ──► respond → END
                            └─► router → transfer_guard ─┬─► (transfer) ──► respond
                                                         ├─► triage    ──► respond
                                                         ├─► specialist ──► respond
                                                         └─► scheduling ──► respond
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.edges import route_from_bypass, route_from_guard, route_from_router
from graph.nodes import (
    bypass_node,
    enter_node,
    respond_node,
    router_node,
    scheduling_node,
    specialist_node,
    transfer_guard_node,
    triage_node,
)
from graph.state import BotState


def build_graph():
    """Constrói e compila o grafo. Devolve o `CompiledGraph` pronto a `ainvoke`."""
    g: StateGraph = StateGraph(BotState)

    # --- Nós ---
    g.add_node("enter", enter_node)
    g.add_node("bypass", bypass_node)
    g.add_node("router", router_node)
    g.add_node("transfer_guard", transfer_guard_node)
    g.add_node("triage", triage_node)
    g.add_node("specialist", specialist_node)
    g.add_node("scheduling", scheduling_node)
    g.add_node("respond", respond_node)

    # --- Edges fixos ---
    g.set_entry_point("enter")
    g.add_edge("enter", "bypass")
    g.add_edge("triage", "respond")
    g.add_edge("specialist", "respond")
    g.add_edge("scheduling", "respond")
    g.add_edge("respond", END)

    # --- Edges condicionais ---
    g.add_conditional_edges(
        "bypass",
        route_from_bypass,
        {"router": "router", "respond": "respond"},
    )
    g.add_conditional_edges(
        "router",
        route_from_router,
        {"transfer_guard": "transfer_guard"},
    )
    g.add_conditional_edges(
        "transfer_guard",
        route_from_guard,
        {
            "triage": "triage",
            "specialist": "specialist",
            "scheduling": "scheduling",
            "respond": "respond",
        },
    )

    return g.compile()
