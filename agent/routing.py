"""Conditional routing for the text-to-SQL LangGraph loop."""
from __future__ import annotations

import logging
import os

from agent.llm_client import MAX_ITERATIONS
from agent.logging_config import log_event, truncate_text
from agent.state import AgentState

logger = logging.getLogger(__name__)


def route_after_verify(state: AgentState) -> str:
    """Route to a revision pass or terminate after successful verification/cap."""
    if state.verify_ok:
        route = "end"
        reason = "verified"
    elif _execution_timed_out(state):
        route = "end"
        reason = "sql_timeout"
    elif _repair_stalled(state):
        route = "end"
        reason = "repair_stalled"
    elif state.iteration >= REVISION_STOP_AFTER_ITERATION:
        route = "end"
        reason = "revision_budget"
    elif state.iteration >= MAX_ITERATIONS:
        route = "end"
        reason = "max_iterations"
    else:
        route = "revise"
        reason = "needs_revision"
    log_event(
        logger,
        logging.INFO,
        "agent.route_decided",
        request_id=state.request_id,
        db_id=state.db_id,
        iteration=state.iteration,
        route=route,
        reason=reason,
        verify_ok=state.verify_ok,
        issue=truncate_text(state.verify_issue),
    )
    return route


def _execution_timed_out(state: AgentState) -> bool:
    execution = state.execution
    return bool(
        execution is not None
        and execution.error
        and "sqlitetimeouterror" in execution.error.casefold()
    )


def _repair_stalled(state: AgentState) -> bool:
    """Stop when a repair pass did not change the SQL or verifier issue."""
    if state.iteration < 2:
        return False

    sql_attempts = [
        _normalize_sql(item.get("sql", ""))
        for item in state.history
        if item.get("node") in {"generate_sql", "revise"} and item.get("sql")
    ]
    if len(sql_attempts) >= 2 and sql_attempts[-1] == sql_attempts[-2]:
        return True

    issues = [
        str(item.get("issue", "")).strip().casefold()
        for item in state.history
        if item.get("node") == "verify" and item.get("issue")
    ]
    return len(issues) >= 2 and issues[-1] == issues[-2]


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().casefold().rstrip(";").split())


REVISION_STOP_AFTER_ITERATION = int(
    os.environ.get("AGENT_REVISION_STOP_AFTER_ITERATION", str(MAX_ITERATIONS))
)
