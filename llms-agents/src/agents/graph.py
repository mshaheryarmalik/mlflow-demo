"""
LangGraph StateGraph definition for the multi-agent pipeline.

Pipeline flow:
  START → researcher → writer → critic → [should_revise?]
                                              ├─ revise → writer (loop)
                                              └─ end → END
"""

from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    critic_node,
    researcher_node,
    reviser_node,
    should_revise,
    writer_node,
)
from agents.state import AgentState


def build_graph() -> StateGraph:
    """
    Construct and compile the multi-agent StateGraph.

    Returns:
        A compiled LangGraph runnable ready for .invoke() or .stream().
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reviser", reviser_node)

    # ── Add edges ─────────────────────────────────────────────────
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")

    # Conditional: either loop back through reviser→writer or finish
    graph.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "reviser",
            "end": END,
        },
    )
    graph.add_edge("reviser", "writer")

    return graph.compile()


# Module-level compiled graph
pipeline = build_graph()
