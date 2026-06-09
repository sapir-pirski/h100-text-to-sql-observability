"""LangGraph wiring for the text-to-SQL agent."""
from __future__ import annotations

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agent.llm_client import (
    FAST_VERIFY,
    LLM_API_KEY,
    LLM_GENERATE_MAX_TOKENS,
    LLM_MAX_TOKENS,
    LLM_REVISE_MAX_TOKENS,
    LLM_VERIFY_MAX_TOKENS,
    MAX_ITERATIONS,
    PROMPT_SET,
    VLLM_BASE_URL,
    VLLM_MODEL,
    generate_llm,
    llm,
    revise_llm,
    verify_llm,
)
from agent.logging_config import configure_logging
from agent.nodes import (
    _attach_schema,
    _coerce_bool,
    _duration_ms,
    _execution_error_type,
    _extract_json_object,
    _extract_sql,
    _fast_verify_issue,
    execute_node,
    generate_sql_node,
    revise_node,
    verify_node,
)
from agent.routing import route_after_verify
from agent.state import AgentState

load_dotenv()
configure_logging()


def build_graph():
    """Build and compile the LangGraph execution graph."""
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()

__all__ = [
    "AgentState",
    "FAST_VERIFY",
    "LLM_API_KEY",
    "LLM_GENERATE_MAX_TOKENS",
    "LLM_MAX_TOKENS",
    "LLM_REVISE_MAX_TOKENS",
    "LLM_VERIFY_MAX_TOKENS",
    "MAX_ITERATIONS",
    "PROMPT_SET",
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "_attach_schema",
    "_coerce_bool",
    "_duration_ms",
    "_execution_error_type",
    "_extract_json_object",
    "_extract_sql",
    "_fast_verify_issue",
    "build_graph",
    "execute_node",
    "generate_sql_node",
    "generate_llm",
    "graph",
    "llm",
    "revise_llm",
    "revise_node",
    "route_after_verify",
    "verify_llm",
    "verify_node",
]
