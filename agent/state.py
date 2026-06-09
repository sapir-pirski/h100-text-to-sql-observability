"""Shared graph state for the text-to-SQL agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.execution import ExecutionResult


@dataclass
class AgentState:
    """State threaded through the LangGraph agent."""

    question: str
    db_id: str
    request_id: str = ""
    schema: str = ""
    grounded_values: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
