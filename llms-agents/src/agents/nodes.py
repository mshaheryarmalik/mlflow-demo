"""
Agent nodes for the LangGraph multi-agent pipeline.

Each node:
  - Receives the shared AgentState
  - Calls an Ollama-backed LLM (via LangChain ChatOllama)
  - Is automatically traced by MLflow via mlflow.langchain.autolog()
  - Returns a dict with the AgentState keys it mutates

Node roles:
  researcher  → gathers information on the topic
  writer      → turns research notes into a structured draft
  critic      → reviews the draft and provides actionable feedback
  reviser     → applies critic feedback and produces the final output
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.config import get_ollama_base_url, get_ollama_model
from agents.state import AgentState
from agents.tools import AVAILABLE_TOOLS

# ── LLM Factory ─────────────────────────────────────────────────────────────


def _build_llm(temperature: float = 0.7) -> ChatOllama:
    """Return a ChatOllama instance pointing at the local Ollama server."""
    return ChatOllama(
        model=get_ollama_model(),
        base_url=get_ollama_base_url(),
        temperature=temperature,
    )


# ── System Prompts ────────────────────────────────────────────────────────────
# These are also registered in MLflow Prompt Registry (see scripts/run_pipeline.py).

RESEARCHER_SYSTEM_PROMPT = """You are a thorough research assistant.
Given a topic, produce concise but comprehensive research notes covering:
- Key concepts and definitions
- Current industry trends and adoption
- Real-world use cases and examples
- Relevant statistics or benchmarks (cite as approximate if unsure)

Be factual. Use bullet points for clarity. Aim for 200-300 words."""

WRITER_SYSTEM_PROMPT = """You are a skilled technical writer.
Given research notes on a topic, write a clear and engaging article draft.
Structure: Introduction → Key Concepts → Industry Impact → Practical Tips → Conclusion.
Target audience: software engineers and data scientists.
Aim for 400-500 words. Use markdown formatting."""

CRITIC_SYSTEM_PROMPT = """You are a senior technical editor.
Review the article draft and provide specific, actionable feedback on:
1. Accuracy — are all claims correct?
2. Clarity — is the language accessible?
3. Completeness — are important points missing?
4. Structure — does the flow make sense?
5. Engagement — is the content clear and compelling?

Be constructive. Output numbered feedback points only."""

REVISER_SYSTEM_PROMPT = """You are a technical writer performing a revision pass.
Apply the editor's feedback to improve the draft.
Output only the final polished article (no meta-commentary).
Preserve markdown formatting."""


# ── Nodes ─────────────────────────────────────────────────────────────────────


def researcher_node(state: AgentState) -> dict:
    """Research node: gather information about the topic."""
    llm = _build_llm(temperature=0.5)
    llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"Research this topic thoroughly: {state['topic']}"),
    ]

    response = llm_with_tools.invoke(messages)
    research_notes = response.content

    return {
        "research": research_notes,
        "messages": [response],
    }


def writer_node(state: AgentState) -> dict:
    """Writer node: turn research notes into a structured draft."""
    llm = _build_llm(temperature=0.8)

    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Research notes:\n{state['research']}\n\n"
                "Write the article draft now."
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        "draft": response.content,
        "messages": [response],
    }


def critic_node(state: AgentState) -> dict:
    """Critic node: review the draft and produce actionable feedback."""
    llm = _build_llm(temperature=0.3)

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Article draft to review:\n{state['draft']}"
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        "feedback": response.content,
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1,
    }


def reviser_node(state: AgentState) -> dict:
    """Reviser node: apply feedback and produce the final polished article."""
    llm = _build_llm(temperature=0.6)

    messages = [
        SystemMessage(content=REVISER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original draft:\n{state['draft']}\n\n"
                f"Editor feedback:\n{state['feedback']}\n\n"
                "Produce the final revised article."
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        "final": response.content,
        "messages": [response],
    }


# ── Routing ───────────────────────────────────────────────────────────────────


def should_revise(state: AgentState) -> str:
    """
    Conditional edge: decide whether to revise or finish.
    Cap at 2 iterations to avoid infinite loops.
    """
    if state.get("iterations", 0) < 2 and state.get("feedback"):
        return "revise"
    return "end"
