"""SQLGlot-based guardrails for model-generated SQL."""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp


class SQLValidationError(ValueError):
    """Raised when generated SQL is not safe to execute."""


@dataclass(frozen=True)
class SQLValidationResult:
    sql: str


DISALLOWED_EXPRESSIONS = tuple(
    cls
    for cls in (
        getattr(exp, "Alter", None),
        getattr(exp, "Attach", None),
        getattr(exp, "Command", None),
        getattr(exp, "Create", None),
        getattr(exp, "Delete", None),
        getattr(exp, "Detach", None),
        getattr(exp, "Drop", None),
        getattr(exp, "Insert", None),
        getattr(exp, "Pragma", None),
        getattr(exp, "Update", None),
    )
    if cls is not None
)

READ_ONLY_ROOTS: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Subquery,
    exp.Union,
    exp.Except,
    exp.Intersect,
)


def validate_read_only_select(sql: str) -> SQLValidationResult:
    """Validate that SQL is one read-only SQLite SELECT/WITH query.

    The agent only needs to answer questions by reading benchmark SQLite DBs.
    Rejecting non-query statements before sqlite3 sees them prevents accidental
    writes, schema changes, attachment of external files, and PRAGMA/VACUUM-like
    commands from model output.
    """
    candidate = sql.strip()
    if not candidate:
        raise SQLValidationError("empty SQL")

    try:
        statements = sqlglot.parse(candidate, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise SQLValidationError(f"parse failed: {exc}") from exc

    statements = [statement for statement in statements if statement is not None]
    if len(statements) != 1:
        raise SQLValidationError("only one SQL statement is allowed")

    statement = statements[0]
    if not _is_read_only_query(statement):
        raise SQLValidationError("only SELECT or WITH queries are allowed")

    disallowed = next(statement.find_all(*DISALLOWED_EXPRESSIONS), None)
    if disallowed is not None:
        raise SQLValidationError(f"disallowed SQL expression: {disallowed.key.upper()}")

    return SQLValidationResult(sql=candidate)


def _is_read_only_query(statement: exp.Expression) -> bool:
    if isinstance(statement, READ_ONLY_ROOTS):
        return True
    if isinstance(statement, exp.With):
        return True
    if statement.args.get("with") is not None and isinstance(statement, READ_ONLY_ROOTS):
        return True
    return False
