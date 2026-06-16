"""
Custom tools available to agent nodes.
All tools are plain Python functions wrapped with @tool from LangChain.
"""

from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a safe arithmetic expression and return the result as a string.

    Args:
        expression: A Python arithmetic expression, e.g. "2 ** 10 + 42"

    Returns:
        String representation of the result, or an error message.
    """
    allowed_names: dict = {"__builtins__": {}}
    try:
        result = eval(expression, allowed_names)  # noqa: S307 — intentionally sandboxed
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def web_search_stub(query: str) -> str:
    """
    Stub for a web search tool.  Returns a canned response so the demo
    works fully offline. Replace the body with a real search API call
    (e.g. Tavily, SerpAPI) when you have credentials.

    Args:
        query: The search query string.

    Returns:
        A placeholder paragraph simulating search results.
    """
    return (
        f"[STUB] Top search results for '{query}':\n"
        "1. MLflow is the most popular open-source AI engineering platform with "
        "30M+ downloads/month.\n"
        "2. LangGraph enables cyclical, stateful multi-agent workflows built on "
        "LangChain primitives.\n"
        "3. Ollama allows running large language models locally with a simple REST API.\n"
        "Source: stub (replace with real search API for production use)."
    )


AVAILABLE_TOOLS = [calculate, web_search_stub]
