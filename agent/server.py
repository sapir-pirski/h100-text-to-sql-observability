"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

load_dotenv()

from agent.logging_config import (  # noqa: E402
    configure_logging,
    log_event,
    question_hash,
    truncate_sql,
)

configure_logging()
logger = logging.getLogger(__name__)

from agent.graph import AgentState, graph  # noqa: E402

# Langfuse callback handler. If keys are set we initialize it; failures
# are NOT swallowed - a misconfigured Langfuse should not silently
# produce zero traces.
_lf_handler: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse.langchain import CallbackHandler

    _lf_handler = CallbackHandler()
    log_event(logger, logging.INFO, "agent.langfuse_enabled")


app = FastAPI()


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    request_id: str
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, request: Request) -> AnswerResponse:
    started = time.perf_counter()
    request_id = _request_id(request)
    q_hash = question_hash(req.question)
    metadata = {
        **req.tags,
        "request_id": request_id,
        "db_id": req.db,
        "question_hash": q_hash,
    }
    log_event(
        logger,
        logging.INFO,
        "agent.request_started",
        request_id=request_id,
        db_id=req.db,
        question_hash=q_hash,
        client_host=request.client.host if request.client else "",
        tags=req.tags,
    )

    state = AgentState(question=req.question, db_id=req.db, request_id=request_id)
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": metadata,
        "tags": _trace_tags(req),
    }
    try:
        final = graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        log_event(
            logger,
            logging.ERROR,
            "agent.request_failed",
            request_id=request_id,
            db_id=req.db,
            question_hash=q_hash,
            duration_ms=_duration_ms(started),
            error_type=type(e).__name__,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    if execution is None:
        log_event(
            logger,
            logging.ERROR,
            "agent.request_completed",
            request_id=request_id,
            db_id=req.db,
            question_hash=q_hash,
            duration_ms=_duration_ms(started),
            iterations=iteration,
            ok=False,
            error_type="MissingExecutionResult",
            error="agent produced no execution result",
            final_sql=truncate_sql(sql),
        )
        response = AnswerResponse(
            request_id=request_id,
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
        return response
    if not execution.ok:
        log_event(
            logger,
            logging.WARNING,
            "agent.request_completed",
            request_id=request_id,
            db_id=req.db,
            question_hash=q_hash,
            duration_ms=_duration_ms(started),
            iterations=iteration,
            ok=False,
            row_count=execution.row_count,
            error_type=_execution_error_type(execution.error),
            error=execution.error,
            final_sql=truncate_sql(sql),
        )
        response = AnswerResponse(
            request_id=request_id,
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )
        return response

    log_event(
        logger,
        logging.INFO,
        "agent.request_completed",
        request_id=request_id,
        db_id=req.db,
        question_hash=q_hash,
        duration_ms=_duration_ms(started),
        iterations=iteration,
        ok=True,
        row_count=execution.row_count,
        column_count=len(execution.columns or []),
        final_sql=truncate_sql(sql),
    )
    response = AnswerResponse(
        request_id=request_id,
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
    return response


def _request_id(request: Request) -> str:
    existing = request.headers.get("x-request-id", "").strip()
    if existing and len(existing) <= 128:
        return existing
    return uuid.uuid4().hex


def _trace_tags(req: AnswerRequest) -> list[str]:
    tags = ["agent", f"db:{req.db}"]
    for key in ("phase", "runner"):
        value = req.tags.get(key)
        if value:
            tags.append(f"{key}:{value}")
    return tags


def _duration_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _execution_error_type(error: str | None) -> str | None:
    if not error:
        return None
    return error.split(":", 1)[0]
