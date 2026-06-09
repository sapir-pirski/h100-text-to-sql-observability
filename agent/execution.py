"""SQL execution helper (provided complete).

execute_sql() runs the agent's SQL against the target DB in read-only mode
and returns a structured ExecutionResult. The verify node consumes this
to decide whether the answer looks plausible.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass

from agent.schema import db_path
from agent.sql_validation import SQLValidationError, validate_read_only_select


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    row_count: int = 0

    def render(
        self,
        max_rows: int = 5,
        max_cell_chars: int = 120,
        max_chars: int = 3000,
    ) -> str:
        """Compact text rendering for prompt context."""
        if not self.ok:
            return f"ERROR: {self.error}"
        if self.row_count == 0:
            return "OK: 0 rows returned."
        cols = ", ".join(self.columns or [])

        def fmt_cell(value: object) -> str:
            text = str(value).replace("\n", " ")
            if len(text) > max_cell_chars:
                return text[: max_cell_chars - 3] + "..."
            return text

        preview = "\n".join(
            " | ".join(fmt_cell(c) for c in row) for row in (self.rows or [])[:max_rows]
        )
        more = f"\n... ({self.row_count - max_rows} more rows)" if self.row_count > max_rows else ""
        rendered = f"OK: {self.row_count} rows.\nCOLUMNS: {cols}\nFIRST ROWS:\n{preview}{more}"
        if len(rendered) > max_chars:
            return rendered[:max_chars] + "\n... (render truncated)"
        return rendered


def execute_sql(db_id: str, sql: str, timeout_seconds: float | None = None) -> ExecutionResult:
    """Run SQL against db_id's sqlite, return result or error."""
    if timeout_seconds is None:
        timeout_seconds = _env_float("SQLITE_QUERY_TIMEOUT_SECONDS", 2.0)
    try:
        validated = validate_read_only_select(sql)
        path = db_path(db_id)
        deadline = time.monotonic() + timeout_seconds
        with sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=timeout_seconds,
        ) as conn:
            conn.set_progress_handler(
                lambda: 1 if time.monotonic() > deadline else 0,
                1000,
            )
            cur = conn.execute(validated.sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return ExecutionResult(ok=True, rows=rows, columns=cols, row_count=len(rows))
    except SQLValidationError as e:
        return ExecutionResult(ok=False, error=f"SQLValidationError: {e}")
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).casefold():
            return ExecutionResult(
                ok=False,
                error=f"SQLiteTimeoutError: query exceeded {timeout_seconds:g}s",
            )
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default
