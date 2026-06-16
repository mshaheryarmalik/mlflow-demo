"""
AgentState — the single shared state object passed between all LangGraph nodes.
"""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state flowing through the multi-agent pipeline.

    Attributes:
        topic:      The original user query / research topic.
        messages:   Full message history (auto-merged by LangGraph).
        research:   Raw research notes produced by the Researcher node.
        draft:      Article draft produced by the Writer node.
        feedback:   Critique produced by the Critic node.
        final:      Final polished output after revisions.
        iterations: How many research→write→critique cycles have occurred.
    """

    topic: str
    messages: Annotated[list, add_messages]
    research: str
    draft: str
    feedback: str
    final: str
    iterations: int
