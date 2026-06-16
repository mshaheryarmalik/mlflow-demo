"""
Tests for the LangGraph multi-agent pipeline.
These tests mock the Ollama LLM to run without a live server.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agents.nodes import critic_node, researcher_node, should_revise, writer_node
from agents.state import AgentState
from agents.tools import calculate, web_search_stub


# ── Tool tests ────────────────────────────────────────────────────────────────


def test_calculate_basic():
    result = calculate.invoke({"expression": "2 + 2"})
    assert result == "4"


def test_calculate_complex():
    result = calculate.invoke({"expression": "2 ** 10"})
    assert result == "1024"


def test_calculate_invalid():
    result = calculate.invoke({"expression": "import os"})
    assert "Error" in result


def test_web_search_stub_returns_string():
    result = web_search_stub.invoke({"query": "MLflow tracing"})
    assert isinstance(result, str)
    assert len(result) > 0


# ── Routing logic tests ───────────────────────────────────────────────────────


def test_should_revise_under_limit():
    state: AgentState = {
        "topic": "test",
        "messages": [],
        "research": "",
        "draft": "",
        "feedback": "Some feedback",
        "final": "",
        "iterations": 1,
    }
    assert should_revise(state) == "revise"


def test_should_revise_over_limit():
    state: AgentState = {
        "topic": "test",
        "messages": [],
        "research": "",
        "draft": "",
        "feedback": "Some feedback",
        "final": "",
        "iterations": 2,
    }
    assert should_revise(state) == "end"


def test_should_revise_no_feedback():
    state: AgentState = {
        "topic": "test",
        "messages": [],
        "research": "",
        "draft": "",
        "feedback": "",
        "final": "",
        "iterations": 0,
    }
    assert should_revise(state) == "end"


# ── Node tests (mocked LLM) ───────────────────────────────────────────────────


@pytest.fixture
def base_state() -> AgentState:
    return {
        "topic": "MLflow overview",
        "messages": [],
        "research": "MLflow is an open-source platform for the ML lifecycle.",
        "draft": "# MLflow\nMLflow tracks experiments and registers models.",
        "feedback": "1. Add more detail about tracing features.",
        "final": "",
        "iterations": 0,
    }


@patch("agents.nodes.ChatOllama")
def test_researcher_node(mock_ollama_cls, base_state):
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Research notes about MLflow.")
    mock_ollama_cls.return_value = mock_llm

    result = researcher_node(base_state)

    assert "research" in result
    assert result["research"] == "Research notes about MLflow."
    assert len(result["messages"]) == 1


@patch("agents.nodes.ChatOllama")
def test_writer_node(mock_ollama_cls, base_state):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="# Draft article about MLflow")
    mock_ollama_cls.return_value = mock_llm

    result = writer_node(base_state)

    assert "draft" in result
    assert result["draft"] == "# Draft article about MLflow"


@patch("agents.nodes.ChatOllama")
def test_critic_node(mock_ollama_cls, base_state):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="1. Good structure\n2. Add examples")
    mock_ollama_cls.return_value = mock_llm

    result = critic_node(base_state)

    assert "feedback" in result
    assert result["iterations"] == 1
