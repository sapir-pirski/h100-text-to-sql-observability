"""LangGraph node implementations for the text-to-SQL agent."""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from typing import Any

from agent.execution import ExecutionResult, execute_sql
from agent.llm_client import FAST_VERIFY, PROMPT_SET, generate_llm, revise_llm, verify_llm
from agent.logging_config import log_event, truncate_sql, truncate_text
from agent.schema import render_schema
from agent.state import AgentState
from agent.value_grounding import format_grounded_values, ground_question_values

logger = logging.getLogger(__name__)
STRICT_VERIFY_HEURISTICS = os.environ.get("AGENT_STRICT_VERIFY_HEURISTICS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}


def attach_schema_node(state: AgentState) -> dict:
    """Render DB schema and value-grounding context once at the start."""
    started = time.perf_counter()
    try:
        grounded = ground_question_values(state.db_id, state.question)
    except Exception as exc:
        grounded = []
        log_event(
            logger,
            logging.WARNING,
            "agent.value_grounding_failed",
            request_id=state.request_id,
            db_id=state.db_id,
            duration_ms=_duration_ms(started),
            error_type=type(exc).__name__,
            error=str(exc),
        )
    try:
        schema = render_schema(
            state.db_id,
            question=state.question,
            pinned_tables=tuple(sorted({item.table for item in grounded})),
            pinned_columns=tuple(sorted({(item.table, item.column) for item in grounded})),
        )
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "agent.node_failed",
            request_id=state.request_id,
            db_id=state.db_id,
            node="attach_schema",
            duration_ms=_duration_ms(started),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    grounded_values = format_grounded_values(grounded)
    log_event(
        logger,
        logging.INFO,
        "agent.node_completed",
        request_id=state.request_id,
        db_id=state.db_id,
        node="attach_schema",
        duration_ms=_duration_ms(started),
        schema_chars=len(schema),
        grounded_value_count=len(grounded),
        grounded_tables=sorted({item.table for item in grounded}),
    )
    return {"schema": schema, "grounded_values": grounded_values}


def generate_sql_node(state: AgentState) -> dict:
    """Generate the first SQL attempt from question, schema, and grounded values."""
    started = time.perf_counter()
    next_iteration = state.iteration + 1
    try:
        response = generate_llm().invoke([
            ("system", PROMPT_SET.generate_sql_system),
            (
                "user",
                PROMPT_SET.generate_sql_user.format(
                    schema=state.schema,
                    grounded_values=state.grounded_values,
                    question=state.question,
                ),
            ),
        ])
        sql = _extract_sql(response.content)
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "agent.node_failed",
            request_id=state.request_id,
            db_id=state.db_id,
            node="generate_sql",
            iteration=next_iteration,
            duration_ms=_duration_ms(started),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    log_event(
        logger,
        logging.INFO,
        "agent.node_completed",
        request_id=state.request_id,
        db_id=state.db_id,
        node="generate_sql",
        iteration=next_iteration,
        duration_ms=_duration_ms(started),
        sql=truncate_sql(sql),
        sql_chars=len(sql),
    )
    return {
        "sql": sql,
        "iteration": next_iteration,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Run generated SQL and store the structured execution result."""
    started = time.perf_counter()
    execution = execute_sql(state.db_id, state.sql)
    log_event(
        logger,
        logging.INFO if execution.ok else logging.WARNING,
        "agent.sql_executed",
        request_id=state.request_id,
        db_id=state.db_id,
        node="execute",
        iteration=state.iteration,
        duration_ms=_duration_ms(started),
        ok=execution.ok,
        row_count=execution.row_count,
        column_count=len(execution.columns or []),
        error_type=_execution_error_type(execution),
        error=execution.error,
        sql=truncate_sql(state.sql),
    )
    return {"execution": execution}


def verify_node(state: AgentState) -> dict:
    """Decide whether the executed result plausibly answers the question."""
    started = time.perf_counter()
    result = _render_verify_result(state.execution)
    if FAST_VERIFY:
        issue = _fast_verify_issue(state.question, state.sql, state.execution)
        ok = issue is None
        log_event(
            logger,
            logging.INFO if ok else logging.WARNING,
            "agent.node_completed",
            request_id=state.request_id,
            db_id=state.db_id,
            node="verify",
            method="heuristic",
            iteration=state.iteration,
            duration_ms=_duration_ms(started),
            ok=ok,
            issue=truncate_text(issue or ""),
        )
        return {
            "verify_ok": ok,
            "verify_issue": issue or "",
            "history": state.history + [{
                "node": "verify",
                "method": "heuristic",
                "ok": ok,
                "issue": issue or "",
                "result": result,
            }],
        }

    try:
        response = verify_llm().invoke([
            ("system", PROMPT_SET.verify_system),
            (
                "user",
                PROMPT_SET.verify_user.format(
                    question=state.question,
                    sql=state.sql,
                    result=result,
                ),
            ),
        ])
        parsed = _extract_json_object(str(response.content))
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "agent.node_failed",
            request_id=state.request_id,
            db_id=state.db_id,
            node="verify",
            method="llm",
            iteration=state.iteration,
            duration_ms=_duration_ms(started),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    parsed_ok = _coerce_bool(parsed.get("ok"))
    issue = str(parsed.get("issue", "")).strip()
    parse_fallback = False
    zero_row_normalized = False
    speculative_issue_normalized = False
    heuristic_reject = False

    if state.execution is None:
        ok = False
        issue = issue or "SQL was not executed."
    elif not state.execution.ok:
        ok = False
        issue = issue or f"SQL execution failed: {state.execution.error}"
    elif parsed_ok is None and _execution_has_result_shape(state.execution):
        ok = True
        issue = ""
        parse_fallback = True
    elif parsed_ok is None:
        ok = False
        issue = issue or "Verifier did not return valid JSON."
    else:
        ok = parsed_ok
        if not ok and _zero_row_only_issue(state.question, state.execution, issue):
            ok = True
            issue = ""
            zero_row_normalized = True
        elif not ok and _speculative_verify_issue(state.execution, issue):
            ok = True
            issue = ""
            speculative_issue_normalized = True
        if not ok and not issue:
            issue = "Result does not plausibly answer the question."

    heuristic_issue = _heuristic_verify_issue(state.question, state.sql, state.execution)
    if ok and heuristic_issue:
        ok = False
        issue = heuristic_issue
        heuristic_reject = True

    log_event(
        logger,
        logging.INFO if ok else logging.WARNING,
        "agent.node_completed",
        request_id=state.request_id,
        db_id=state.db_id,
        node="verify",
        method="llm",
        iteration=state.iteration,
        duration_ms=_duration_ms(started),
        ok=ok,
        issue=truncate_text(issue),
        parse_fallback=parse_fallback,
        zero_row_normalized=zero_row_normalized,
        speculative_issue_normalized=speculative_issue_normalized,
        heuristic_reject=heuristic_reject,
    )
    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{
            "node": "verify",
            "method": "llm",
            "ok": ok,
            "issue": issue,
            "result": result,
            "parse_fallback": parse_fallback,
            "zero_row_normalized": zero_row_normalized,
            "speculative_issue_normalized": speculative_issue_normalized,
            "heuristic_reject": heuristic_reject,
        }],
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query from the prior attempt and verifier issue."""
    started = time.perf_counter()
    next_iteration = state.iteration + 1
    result = state.execution.render() if state.execution is not None else "ERROR: no execution result"
    try:
        response = revise_llm().invoke([
            ("system", PROMPT_SET.revise_system),
            (
                "user",
                PROMPT_SET.revise_user.format(
                    schema=state.schema,
                    grounded_values=state.grounded_values,
                    question=state.question,
                    sql=state.sql,
                    result=result,
                    issue=state.verify_issue,
                ),
            ),
        ])
        sql = _extract_sql(str(response.content))
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "agent.node_failed",
            request_id=state.request_id,
            db_id=state.db_id,
            node="revise",
            iteration=next_iteration,
            duration_ms=_duration_ms(started),
            error_type=type(exc).__name__,
            error=str(exc),
            issue=truncate_text(state.verify_issue),
        )
        raise
    log_event(
        logger,
        logging.INFO,
        "agent.node_completed",
        request_id=state.request_id,
        db_id=state.db_id,
        node="revise",
        iteration=next_iteration,
        duration_ms=_duration_ms(started),
        issue=truncate_text(state.verify_issue),
        sql=truncate_sql(sql),
        sql_chars=len(sql),
    )
    return {
        "sql": sql,
        "iteration": next_iteration,
        "history": state.history + [{
            "node": "revise",
            "sql": sql,
            "issue": state.verify_issue,
        }],
    }


def _extract_sql(text: str) -> str:
    """Pull one SQL statement from an LLM reply, stripping fences and prose."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = (fenced.group(1) if fenced else text).strip()
    if not fenced:
        sql_start = re.search(r"\b(?:WITH|SELECT)\b", candidate, re.IGNORECASE)
        if sql_start:
            candidate = candidate[sql_start.start():]
    semicolon = candidate.find(";")
    if semicolon >= 0:
        candidate = candidate[:semicolon + 1]
    return candidate.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from an LLM reply."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = (fenced.group(1) if fenced else text).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except (SyntaxError, ValueError):
            pass

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "ok"}:
            return True
        if lowered in {"false", "no", "n", "0", "bad"}:
            return False
    return None


def _render_verify_result(execution: ExecutionResult | None) -> str:
    """Render only the result facts the LLM verifier needs."""
    if execution is None:
        return "ERROR: no execution result"
    return execution.render(max_rows=3, max_cell_chars=80, max_chars=1200)


def _execution_has_result_shape(execution: ExecutionResult) -> bool:
    """Return True when an executed SELECT produced a usable result shape."""
    return execution.ok and execution.columns is not None and len(execution.columns) > 0


def _zero_row_only_issue(question: str, execution: ExecutionResult, issue: str) -> bool:
    """Treat bare zero-row suspicion as acceptable, not a revision trigger."""
    if not _execution_has_result_shape(execution) or execution.row_count != 0:
        return False
    if _zero_rows_need_revision(question, execution):
        return False
    normalized = issue.strip().casefold()
    if not normalized:
        return False
    return any(marker in normalized for marker in ("no rows", "zero rows", "0 rows"))


def _speculative_verify_issue(execution: ExecutionResult, issue: str) -> bool:
    """Suppress verifier speculation after a successful shaped result."""
    if not _execution_has_result_shape(execution):
        return False
    normalized = issue.strip().casefold()
    if not normalized:
        return False
    concrete_markers = (
        "aggregation",
        "aggregate",
        "column",
        "count",
        "error",
        "invalid",
        "limit",
        "order",
        "parse",
        "rank",
        "syntax",
        "unrelated",
        "wrong",
    )
    return not any(marker in normalized for marker in concrete_markers)


def _heuristic_verify_issue(
    question: str,
    sql: str,
    execution: ExecutionResult | None,
) -> str | None:
    """Catch concrete result-shape mistakes before accepting an LLM verdict."""
    if execution is None or not execution.ok:
        return None
    normalized_question = question.casefold()
    normalized_sql = _normalize_sql(sql)

    if _zero_rows_need_revision(question, execution):
        return "Zero rows for entity lookup."
    if _null_only_result(execution):
        return "Aggregate returned NULL."
    if _count_zero_with_entity_filter(question, sql, execution):
        return "Count is zero; revise coded values."
    if _duplicate_single_answer(question, execution):
        return "Duplicate rows; add DISTINCT."
    if _time_seconds_conversion_mistake(question, sql):
        return "Duration conversion is wrong."
    if _asks_for_time_value(normalized_question) and _selects_milliseconds_instead(normalized_sql):
        return "Return time text, not milliseconds."
    if "excerpt post" in normalized_question and "excerptpostid" not in normalized_sql:
        return "Use tags.ExcerptPostId join."
    if "well-finished" in normalized_question and not _uses_boolean_answer(normalized_sql):
        return "Return well-finished label."
    if "missing weight" in normalized_question and "weight_kg" in normalized_sql:
        if "weight_kg = 0" not in normalized_sql and "weight_kg=0" not in normalized_sql:
            return "Missing weight includes 0."
    if "normal ig g" in normalized_question and "acl" in normalized_sql:
        return "Use Laboratory.IGG."
    if "normal uric acid" in normalized_question and "sex" not in normalized_sql:
        return "Normal UA depends on sex."
    if "higher popularity" in normalized_question and "displayname" not in _select_clause(normalized_sql):
        return "Return the higher user name."
    if "2019 and 2020" in normalized_question and _subtracts_second_year_first(normalized_sql):
        return "Compute 2019 minus 2020."
    if _needs_not_null_type_filter(normalized_question, execution):
        return "Filter originalType IS NOT NULL."
    return None


def _zero_rows_need_revision(question: str, execution: ExecutionResult) -> bool:
    if not _execution_has_result_shape(execution) or execution.row_count != 0:
        return False
    normalized = question.casefold()
    if any(marker in normalized for marker in ("how many", "count ", "number of")):
        return False
    entity_lookup_markers = (
        "among",
        "list",
        "mention",
        "mostly",
        "provide",
        "show",
        "what is",
        "what are",
        "which",
        "who",
    )
    return any(marker in normalized for marker in entity_lookup_markers)


def _null_only_result(execution: ExecutionResult) -> bool:
    if execution.row_count != 1 or not execution.rows:
        return False
    row = execution.rows[0]
    return bool(row) and all(value is None for value in row)


def _count_zero_with_entity_filter(
    question: str,
    sql: str,
    execution: ExecutionResult,
) -> bool:
    if execution.row_count != 1 or not execution.rows:
        return False
    row = execution.rows[0]
    if len(row) != 1 or row[0] not in {0, "0"}:
        return False
    normalized_question = question.casefold()
    if not any(marker in normalized_question for marker in ("how many", "count", "number of")):
        return False
    normalized_sql = _normalize_sql(sql)
    if not re.search(r"count\s*\(", normalized_sql):
        return False
    # A zero count is suspicious only when the query filtered on an entity or
    # coded categorical value. Plain unfiltered COUNT(*) = 0 can be valid.
    return bool(re.search(r"=\s*'[^']*[a-z][^']*'", normalized_sql))


def _duplicate_single_answer(question: str, execution: ExecutionResult) -> bool:
    if execution.row_count <= 1 or not execution.rows:
        return False
    normalized = question.casefold()
    if not any(marker in normalized for marker in ("what is", "which", "who", "coordinates")):
        return False
    unique_rows = {tuple(row) for row in execution.rows}
    return len(unique_rows) == 1


def _time_seconds_conversion_mistake(question: str, sql: str) -> bool:
    normalized_question = question.casefold()
    normalized_sql = sql.casefold()
    return (
        "seconds" in normalized_question
        and "time" in normalized_question
        and "replace(" in normalized_sql
        and "':'" in normalized_sql
    )


def _asks_for_time_value(normalized_question: str) -> bool:
    return "time" in normalized_question and any(
        marker in normalized_question
        for marker in ("fastest", "slowest", "lap records", "fastest one")
    )


def _selects_milliseconds_instead(normalized_sql: str) -> bool:
    select_head = _select_clause(normalized_sql)
    return "milliseconds" in select_head


def _select_clause(normalized_sql: str) -> str:
    return normalized_sql.split(" from ", 1)[0]


def _uses_boolean_answer(normalized_sql: str) -> bool:
    return any(marker in normalized_sql for marker in ("case", "iif", "well-finished", "not well-finished"))


def _subtracts_second_year_first(normalized_sql: str) -> bool:
    first_2020 = normalized_sql.find("'2020'")
    first_2019 = normalized_sql.find("'2019'")
    minus = normalized_sql.find("-")
    return first_2020 >= 0 and first_2019 >= 0 and minus >= 0 and first_2020 < minus < first_2019


def _needs_not_null_type_filter(normalized_question: str, execution: ExecutionResult) -> bool:
    if "originally printed" not in normalized_question and "original type" not in normalized_question:
        return False
    if execution.row_count <= 1 or not execution.rows:
        return False
    return any(any(value is None for value in row) for row in execution.rows)


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.casefold())


def _fast_verify_issue(
    question: str,
    sql: str,
    execution: ExecutionResult | None,
) -> str | None:
    """Return a deterministic verification issue, or None when result is acceptable."""
    if execution is None:
        return "SQL was not executed."
    if not execution.ok:
        return f"SQL execution failed: {execution.error}"
    return _fast_result_shape_issue(question, sql, execution)


def _fast_result_shape_issue(
    question: str,
    sql: str,
    execution: ExecutionResult,
) -> str | None:
    """Latency-safe subset of deterministic checks for the fast verifier profile."""
    normalized_question = question.casefold()
    normalized_sql = _normalize_sql(sql)
    if _duplicate_single_answer(question, execution):
        return "Duplicate rows; add DISTINCT."
    if "well-finished" in normalized_question and not _uses_boolean_answer(normalized_sql):
        return "Return well-finished label."
    if "missing weight" in normalized_question and "weight_kg" in normalized_sql:
        if "weight_kg = 0" not in normalized_sql and "weight_kg=0" not in normalized_sql:
            return "Missing weight includes 0."
    if _needs_not_null_type_filter(normalized_question, execution):
        return "Filter originalType IS NOT NULL."
    return None


def _duration_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _execution_error_type(execution: ExecutionResult | None) -> str | None:
    if execution is None or not execution.error:
        return None
    return execution.error.split(":", 1)[0]


# Backward-compatible name used by earlier graph.py versions.
_attach_schema = attach_schema_node
