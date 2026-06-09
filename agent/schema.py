"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"
TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type_name: str
    notnull: bool
    pk: bool


@dataclass(frozen=True)
class ForeignKeyInfo:
    from_col: str
    ref_table: str
    ref_col: str


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[ColumnInfo, ...]
    foreign_keys: tuple[ForeignKeyInfo, ...]


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def schema_pruning_enabled() -> bool:
    return os.environ.get("SCHEMA_PRUNING", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def render_schema(
    db_id: str,
    question: str | None = None,
    pinned_tables: tuple[str, ...] = (),
) -> str:
    """Render a task schema, pruning to question-relevant tables when possible."""
    question_key = _normalize_question(question or "")
    pinned_key = tuple(sorted(set(pinned_tables)))
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    tables = _schema_metadata(db_id)
    selected = _select_tables(tables, question_key, pinned_key)
    parts: list[str] = [f"-- Database: {db_id}"]
    if schema_pruning_enabled() and len(selected) < len(tables):
        parts.append(
            f"-- Pruned schema: {len(selected)} of {len(tables)} tables selected for the question."
        )
    for table in selected:
        parts.append(_render_table(table))
    return "\n".join(parts)


def _schema_metadata(db_id: str) -> tuple[TableInfo, ...]:
    path = db_path(db_id)
    tables: list[TableInfo] = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        table_names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for table_name in table_names:
            columns = tuple(
                ColumnInfo(name=name, type_name=ctype, notnull=bool(notnull), pk=bool(pk))
                for _cid, name, ctype, notnull, _dflt, pk in conn.execute(
                    f"PRAGMA table_info({_q(table_name)})"
                )
            )
            foreign_keys = []
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(table_name)})"):
                from_col = fk[3]
                ref_table = fk[2]
                ref_col = fk[4]
                if from_col and ref_table:
                    foreign_keys.append(ForeignKeyInfo(from_col, ref_table, ref_col or ""))
            tables.append(TableInfo(table_name, columns, tuple(foreign_keys)))
    return tuple(tables)


def _select_tables(
    tables: tuple[TableInfo, ...],
    question_key: str,
    pinned_tables: tuple[str, ...],
) -> tuple[TableInfo, ...]:
    if not schema_pruning_enabled():
        return tables

    max_tables = _env_int("SCHEMA_MAX_TABLES", 8)
    if len(tables) <= max_tables:
        return tables

    tokens = set(TOKEN_RE.findall(question_key))
    table_by_name = {table.name: table for table in tables}
    pinned = {name for name in pinned_tables if name in table_by_name}
    scored: list[tuple[int, str, TableInfo]] = []
    for table in tables:
        score = 0
        table_tokens = _identifier_tokens(table.name)
        score += 8 * len(tokens & table_tokens)
        for column in table.columns:
            column_tokens = _identifier_tokens(column.name)
            overlap = len(tokens & column_tokens)
            score += 4 * overlap
            if column.pk and overlap:
                score += 2
        if table.name in pinned:
            score += 100
        scored.append((score, table.name, table))

    selected_names = {name for score, name, _table in scored if score > 0}
    if not selected_names:
        selected_names = {name for _score, name, _table in sorted(scored, reverse=True)[:max_tables]}

    # Include direct FK neighbors so joins remain possible.
    expanded = set(selected_names)
    for table in tables:
        if table.name in selected_names:
            expanded.update(fk.ref_table for fk in table.foreign_keys)
        if any(fk.ref_table in selected_names for fk in table.foreign_keys):
            expanded.add(table.name)

    ranked_names = [name for _score, name, _table in sorted(scored, reverse=True)]
    ordered = []
    for name in ranked_names:
        if name in expanded and name not in ordered:
            ordered.append(name)
    if len(ordered) < max_tables:
        for name in ranked_names:
            if name not in ordered:
                ordered.append(name)
            if len(ordered) >= max_tables:
                break
    else:
        ordered = ordered[: max(max_tables, len(pinned))]
    for name in pinned:
        if name not in ordered:
            ordered.append(name)

    return tuple(table_by_name[name] for name in sorted(ordered) if name in table_by_name)


def _render_table(table: TableInfo) -> str:
    col_lines: list[str] = []
    for column in table.columns:
        line = f"  {_q(column.name)} {column.type_name}"
        if column.pk:
            line += " PRIMARY KEY"
        if column.notnull and not column.pk:
            line += " NOT NULL"
        col_lines.append(line)
    for fk in table.foreign_keys:
        reference = f"REFERENCES {_q(fk.ref_table)}"
        if fk.ref_col:
            reference += f"({_q(fk.ref_col)})"
        col_lines.append(f"  FOREIGN KEY ({_q(fk.from_col)}) {reference}")
    return f"\nCREATE TABLE {_q(table.name)} (\n" + ",\n".join(col_lines) + "\n);"


def _identifier_tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(value.casefold().replace("_", " ")))


def _normalize_question(question: str) -> str:
    return " ".join(TOKEN_RE.findall(question.casefold()))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
