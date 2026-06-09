"""Structured logging helpers for the agent service."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

DEFAULT_LOG_FILE = "logs/agent.log"
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 5
DEFAULT_TEXT_MAX_CHARS = 500
DEFAULT_SQL_MAX_CHARS = 1000

SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "hf_token",
    "key",
    "password",
    "secret",
    "token",
)

_STANDARD_RECORD_ATTRS = set(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__
)
_STANDARD_RECORD_ATTRS.update({"asctime", "message"})


class JsonFormatter(logging.Formatter):
    """Format records as one compact JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            payload.update(structured)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_ATTRS
            and key not in {"event", "structured"}
            and not key.startswith("_")
        }
        if extras:
            payload.update(_sanitize(extras))

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, default=str, separators=(",", ":"))


def configure_logging() -> None:
    """Configure structured logging for all `agent.*` loggers.

    The default writes JSON lines to logs/agent.log with rotation. Console
    logging is opt-in to avoid duplicating lines when Uvicorn stdout is already
    redirected to the same file by scripts/run-full-project.sh.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE).strip()
    max_bytes = _env_int("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES)
    backup_count = _env_int("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT)
    to_stdout = os.environ.get("LOG_TO_STDOUT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    logger = logging.getLogger("agent")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_agent_json_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = JsonFormatter()
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler._agent_json_handler = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    if to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        stream_handler._agent_json_handler = True  # type: ignore[attr-defined]
        logger.addHandler(stream_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit one sanitized structured log event."""
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "structured": _sanitize(fields),
        },
    )


def question_hash(question: str) -> str:
    """Return a stable hash for correlation without logging question text."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def truncate_text(value: str | None, limit: int | None = None) -> str:
    """Truncate free text for logs."""
    if value is None:
        return ""
    max_chars = limit if limit is not None else _env_int("LOG_TEXT_MAX_CHARS", DEFAULT_TEXT_MAX_CHARS)
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def truncate_sql(sql: str | None, limit: int | None = None) -> str:
    """Truncate SQL for logs while keeping enough detail for debugging."""
    max_chars = limit if limit is not None else _env_int("LOG_SQL_MAX_CHARS", DEFAULT_SQL_MAX_CHARS)
    return truncate_text(sql, max_chars)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                sanitized[key_text] = "[REDACTED]"
            elif key_text in {"sql", "final_sql"}:
                sanitized[key_text] = truncate_sql(str(item))
            else:
                sanitized[key_text] = _sanitize(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value[:50]]
    if isinstance(value, str):
        return truncate_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return truncate_text(str(value))
