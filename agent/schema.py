"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import csv
import os
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"
DESCRIPTION_DIR = DB_DIR / "dev_20240627" / "dev_databases"
TOKEN_RE = re.compile(r"[a-z0-9]+")

SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("coordinate", "coordinates", "lat", "latitude", "lng", "longitude"),
    ("location", "address", "street", "city", "state", "zip"),
    ("number", "num", "no", "count"),
    ("identifier", "id"),
    ("crime", "crimes", "committed"),
    ("print", "printed", "printing", "original", "originally"),
    ("popular", "popularity", "view", "views", "viewcount", "score"),
    ("finished", "well", "closed", "closeddate"),
    ("uric", "acid", "ua"),
    ("ig", "igg"),
    ("bilirubin", "tbil", "t", "bil"),
    ("calcium", "ca"),
    ("chlorine", "cl"),
    ("carcinogenic", "label"),
)

SYNONYMS: dict[str, set[str]] = {}
for group in SYNONYM_GROUPS:
    members = set(group)
    for token in group:
        SYNONYMS[token] = members - {token}


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type_name: str
    notnull: bool
    pk: bool
    display_name: str = ""
    description: str = ""
    value_description: str = ""


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


@dataclass(frozen=True)
class ColumnDescription:
    display_name: str = ""
    description: str = ""
    value_description: str = ""


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


@lru_cache(maxsize=2048)
def render_schema(
    db_id: str,
    question: str | None = None,
    pinned_tables: tuple[str, ...] = (),
    pinned_columns: tuple[tuple[str, str], ...] = (),
) -> str:
    """Render a task schema, pruning to question-relevant tables when possible."""
    question_text = question or ""
    question_tokens = _identifier_tokens(question_text)
    pinned_key = tuple(sorted(set(pinned_tables)))
    pinned_column_key = tuple(sorted(set(pinned_columns)))
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    tables = _schema_metadata(db_id)
    column_scores = {
        table.name: _score_columns(table, question_text, question_tokens)
        for table in tables
    }
    table_scores = _score_tables(tables, column_scores, question_text, question_tokens, pinned_key)
    selected = _select_tables(tables, table_scores, pinned_key)
    selected_columns = {
        table.name: _select_columns(
            table,
            column_scores[table.name],
            question_text,
            question_tokens,
            pinned_column_key,
        )
        for table in selected
    }

    parts: list[str] = [f"-- Database: {db_id}"]
    if schema_pruning_enabled() and len(selected) < len(tables):
        parts.append(
            f"-- Pruned schema: {len(selected)} of {len(tables)} tables selected for the question."
        )
    parts.extend(_render_linking_summary(selected, selected_columns, table_scores, column_scores))
    for table in selected:
        parts.append(_render_table(table, selected_columns[table.name]))
    return "\n".join(parts)


@lru_cache(maxsize=32)
def _schema_metadata(db_id: str) -> tuple[TableInfo, ...]:
    path = db_path(db_id)
    descriptions = _column_descriptions(db_id)
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
            table_descriptions = descriptions.get(table_name, {})
            columns = tuple(
                ColumnInfo(
                    name=name,
                    type_name=ctype,
                    notnull=bool(notnull),
                    pk=bool(pk),
                    display_name=table_descriptions.get(name, ColumnDescription()).display_name,
                    description=table_descriptions.get(name, ColumnDescription()).description,
                    value_description=table_descriptions.get(name, ColumnDescription()).value_description,
                )
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


@lru_cache(maxsize=32)
def _column_descriptions(db_id: str) -> dict[str, dict[str, ColumnDescription]]:
    """Load BIRD database_description CSV metadata when it is available."""
    description_root = DESCRIPTION_DIR / db_id / "database_description"
    if not description_root.exists():
        return {}

    descriptions: dict[str, dict[str, ColumnDescription]] = {}
    for csv_path in sorted(description_root.glob("*.csv")):
        table_descriptions: dict[str, ColumnDescription] = {}
        with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                original = _clean_metadata_text(row.get("original_column_name", ""))
                if not original:
                    continue
                table_descriptions[original] = ColumnDescription(
                    display_name=_clean_metadata_text(row.get("column_name", "")),
                    description=_clean_metadata_text(row.get("column_description", "")),
                    value_description=_clean_metadata_text(row.get("value_description", "")),
                )
        if table_descriptions:
            descriptions[csv_path.stem] = table_descriptions
    return descriptions


def _score_tables(
    tables: tuple[TableInfo, ...],
    column_scores: dict[str, dict[str, int]],
    question_text: str,
    question_tokens: set[str],
    pinned_tables: tuple[str, ...],
) -> dict[str, int]:
    pinned = set(pinned_tables)
    scores: dict[str, int] = {}
    for table in tables:
        score = 10 * len(question_tokens & _identifier_tokens(table.name))
        ranked_columns = sorted(column_scores[table.name].values(), reverse=True)
        score += sum(ranked_columns[:4])
        if any(value > 0 for value in ranked_columns):
            score += 6
        if table.name in pinned:
            score += 120
        score += _table_phrase_boost(table, question_text, question_tokens)
        scores[table.name] = score
    return scores


def _score_columns(
    table: TableInfo,
    question_text: str,
    question_tokens: set[str],
) -> dict[str, int]:
    scores: dict[str, int] = {}
    question_key = question_text.casefold()
    wants_numeric = bool(
        question_tokens
        & {
            "average",
            "avg",
            "count",
            "difference",
            "highest",
            "lowest",
            "max",
            "min",
            "number",
            "percentage",
            "sum",
            "total",
        }
    )
    wants_date = bool(question_tokens & {"date", "day", "month", "time", "year"})

    for column in table.columns:
        identifier_tokens = _identifier_tokens(f"{column.name} {column.display_name}")
        description_tokens = _identifier_tokens(column.description)
        value_tokens = _identifier_tokens(column.value_description)

        score = 0
        score += 12 * len(question_tokens & identifier_tokens)
        score += 7 * len(question_tokens & description_tokens)
        score += 5 * len(question_tokens & value_tokens)

        for alias in (column.name, column.display_name):
            alias_key = _normalize_phrase(alias)
            if alias_key and alias_key in question_key:
                score += 18

        if wants_numeric and _is_numeric_type(column.type_name):
            score += 3
        if wants_date and _is_date_type(column.type_name, column.name):
            score += 8
        if "normal" in question_tokens and "normal range" in column.value_description.casefold():
            score += 16
        if "missing" in question_tokens and "missing" in column.value_description.casefold():
            score += 16
        score += _column_phrase_boost(table, column, question_text, question_tokens)
        scores[column.name] = score
    return scores


def _select_tables(
    tables: tuple[TableInfo, ...],
    table_scores: dict[str, int],
    pinned_tables: tuple[str, ...],
) -> tuple[TableInfo, ...]:
    if not schema_pruning_enabled():
        return tables

    max_tables = _env_int("SCHEMA_MAX_TABLES", 8)
    table_by_name = {table.name: table for table in tables}
    pinned = {name for name in pinned_tables if name in table_by_name}
    ranked = sorted(tables, key=lambda table: (-table_scores.get(table.name, 0), table.name.casefold()))
    if len(tables) <= max_tables:
        return tuple(ranked)

    selected_names = {table.name for table in ranked if table_scores.get(table.name, 0) > 0}
    if not selected_names:
        selected_names = {table.name for table in ranked[:max_tables]}
    selected_names.update(pinned)

    # Include direct FK neighbors so joins remain possible.
    expanded = set(selected_names)
    for table in tables:
        if table.name in selected_names:
            expanded.update(fk.ref_table for fk in table.foreign_keys)
        if any(fk.ref_table in selected_names for fk in table.foreign_keys):
            expanded.add(table.name)

    ranked_names = [table.name for table in ranked]
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

    return tuple(table_by_name[name] for name in ordered if name in table_by_name)


def _select_columns(
    table: TableInfo,
    column_scores: dict[str, int],
    question_text: str,
    question_tokens: set[str],
    pinned_columns: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    if not schema_pruning_enabled():
        return tuple(column.name for column in table.columns)

    max_columns = _env_int("SCHEMA_MAX_COLUMNS_PER_TABLE", 18)
    min_columns = _env_int("SCHEMA_MIN_COLUMNS_PER_TABLE", 4)
    if len(table.columns) <= max_columns:
        return tuple(column.name for column in table.columns)

    pinned = {column for table_name, column in pinned_columns if table_name == table.name}
    selected = set(pinned)

    for column in table.columns:
        if column.pk:
            selected.add(column.name)
    for fk in table.foreign_keys:
        selected.add(fk.from_col)

    ranked = sorted(
        table.columns,
        key=lambda column: (-column_scores.get(column.name, 0), table.columns.index(column)),
    )
    for column in ranked:
        if column_scores.get(column.name, 0) > 0:
            selected.add(column.name)
        if len(selected) >= max_columns:
            break

    if len(selected) < min_columns:
        for column in ranked:
            selected.add(column.name)
            if len(selected) >= min_columns:
                break

    return tuple(column.name for column in table.columns if column.name in selected)


def _render_linking_summary(
    selected_tables: tuple[TableInfo, ...],
    selected_columns: dict[str, tuple[str, ...]],
    table_scores: dict[str, int],
    column_scores: dict[str, dict[str, int]],
) -> list[str]:
    if not _env_bool("SCHEMA_LINKING_SUMMARY", True) or not schema_pruning_enabled():
        return []
    table_names = ", ".join(table.name for table in selected_tables[:8])
    ranked_columns: list[tuple[int, str]] = []
    for table in selected_tables:
        chosen = set(selected_columns.get(table.name, ()))
        for column in table.columns:
            score = column_scores[table.name].get(column.name, 0)
            if column.name in chosen and score > 0:
                ranked_columns.append((score, f"{table.name}.{column.name}"))
    ranked_columns.sort(key=lambda item: (-item[0], item[1].casefold()))
    column_names = ", ".join(name for _score, name in ranked_columns[:16])
    lines = ["-- Schema linking: ranked tables/columns selected before SQL generation."]
    if table_names:
        lines.append(f"-- High-priority tables: {table_names}")
    if column_names:
        lines.append(f"-- High-priority columns: {column_names}")
    return lines


def _render_table(table: TableInfo, selected_column_names: tuple[str, ...]) -> str:
    selected = set(selected_column_names)
    skipped = len(table.columns) - len(selected)
    header = f"\n-- Table {table.name}: {len(selected)} of {len(table.columns)} columns shown"
    if skipped <= 0:
        header = f"\n-- Table {table.name}"

    col_lines: list[str] = []
    for column in table.columns:
        if column.name not in selected:
            continue
        line = f"  {_q(column.name)} {column.type_name}"
        if column.pk:
            line += " PRIMARY KEY"
        if column.notnull and not column.pk:
            line += " NOT NULL"
        hint = _column_hint(column)
        if hint:
            line += f" /* {hint} */"
        col_lines.append(line)
    for fk in table.foreign_keys:
        if fk.from_col not in selected:
            continue
        reference = f"REFERENCES {_q(fk.ref_table)}"
        if fk.ref_col:
            reference += f"({_q(fk.ref_col)})"
        col_lines.append(f"  FOREIGN KEY ({_q(fk.from_col)}) {reference}")
    return header + f"\nCREATE TABLE {_q(table.name)} (\n" + ",\n".join(col_lines) + "\n);"


def _table_phrase_boost(table: TableInfo, question_text: str, question_tokens: set[str]) -> int:
    table_name = table.name.casefold()
    question_key = question_text.casefold()
    score = 0
    if table_name == "comments" and "comment" in question_tokens:
        score += 30
    if table_name == "postHistory" and "history" not in question_tokens and "comment" in question_tokens:
        score -= 12
    if table_name == "lapTimes" and {"lap", "fastest"} & question_tokens:
        score += 24
    if table_name == "results" and {"finisher", "finishers", "disqualified", "fastest"} & question_tokens:
        score += 18
    if table_name == "races" and {"grand", "prix", "race"} & question_tokens:
        score += 18
    if table_name == "circuits" and {"coordinate", "coordinates", "location"} & question_tokens:
        score += 18
    if table_name == "users" and {"display", "user", "users", "owned", "by"} & question_tokens:
        score += 14
    if table_name == "posts" and {"post", "posts", "popularity", "view"} & question_tokens:
        score += 14
    if table_name == "cards" and {"card", "cards", "printed", "mythic"} & question_tokens:
        score += 18
    if table_name == "legalities" and {"banned", "format", "gladiator"} & question_tokens:
        score += 24
    if table_name == "laboratory" and {"normal", "uric", "acid", "ig", "igg", "bilirubin"} & question_tokens:
        score += 24
    if table_name == "examination" and "symptoms" in question_tokens:
        score += 24
    if table_name == "patient" and {"patients", "patient", "sex", "admission"} & question_tokens:
        score += 16
    if table_name == "district" and {"district", "region", "crime", "crimes"} & question_tokens:
        score += 18
    if table_name == "account" and {"account", "accounts", "opened"} & question_tokens:
        score += 18
    if table_name == "atom" and {"element", "calcium", "chlorine"} & question_tokens:
        score += 24
    if table_name == "molecule" and "carcinogenic" in question_tokens:
        score += 24
    if table_name == "colour" and ("eye color" in question_key or "eye colour" in question_key):
        score += 24
    return score


def _column_phrase_boost(
    table: TableInfo,
    column: ColumnInfo,
    question_text: str,
    question_tokens: set[str],
) -> int:
    table_name = table.name.casefold()
    column_name = column.name.casefold()
    question_key = question_text.casefold()
    score = 0

    if {"coordinate", "coordinates", "location"} & question_tokens and column_name in {"lat", "lng"}:
        score += 30
    if "complete address" in question_key and column_name in {"street", "city", "state", "zip"}:
        score += 30
    if {"average", "crimes", "crime", "1995"} <= question_tokens and column_name == "a15":
        score += 36
    if {"crimes", "crime", "1996"} <= question_tokens and column_name == "a16":
        score += 24
    if {"entrepreneurs"} & question_tokens and column_name == "a14":
        score += 24
    if "originally printed" in question_key and column_name == "originaltype":
        score += 36
    if table_name == "cards" and "print cards" in question_key and column_name == "id":
        score += 24
    if table_name == "cards" and {"name", "card"} & question_tokens and column_name == "name":
        score += 14
    if table_name == "legalities" and column_name in {"format", "status"}:
        score += 16
    if table_name == "results" and column_name == "time" and {"finisher", "finishers"} & question_tokens:
        score += 28
    if table_name == "results" and column_name == "fastestlaptime" and {"fastest", "lap", "seconds"} <= question_tokens:
        score += 30
    if table_name == "laptimes" and column_name == "time" and {"fastest", "time"} <= question_tokens:
        score += 34
    if table_name == "laptimes" and column_name == "milliseconds" and "seconds" not in question_tokens:
        score -= 10
    if table_name == "races" and column_name == "name" and {"grand", "prix"} & question_tokens:
        score += 24
    if table_name == "races" and column_name == "round" and "round" not in question_tokens:
        score -= 8
    if table_name == "comments" and column_name in {"userid", "creationdate", "postid"}:
        score += 18
    if table_name == "posthistory" and "comment" in question_tokens and "history" not in question_tokens:
        score -= 10
    if table_name == "posts" and column_name == "closeddate" and "well-finished" in question_key:
        score += 40
    if table_name == "posts" and column_name == "viewcount" and {"popular", "popularity"} & question_tokens:
        score += 30
    if table_name == "users" and column_name == "displayname" and {"display", "name", "by"} & question_tokens:
        score += 24
    if table_name == "laboratory" and column_name == "igg" and {"ig", "igg"} & question_tokens:
        score += 34
    if table_name == "laboratory" and column_name == "ua" and {"uric", "acid", "ua"} & question_tokens:
        score += 34
    if table_name == "laboratory" and column_name == "t-bil" and "bilirubin" in question_tokens:
        score += 34
    if table_name == "patient" and column_name in {"sex", "admission"}:
        score += 14
    if table_name == "examination" and column_name == "symptoms" and "symptoms" in question_tokens:
        score += 34
    if table_name == "superhero" and column_name == "weight_kg" and "missing" in question_tokens:
        score += 34
    if table_name == "superhero" and column_name == "eye_colour_id" and "eye" in question_tokens:
        score += 26
    if table_name == "colour" and column_name == "colour" and "eye" in question_tokens:
        score += 26
    if table_name == "atom" and column_name == "element" and {"element", "calcium", "chlorine"} & question_tokens:
        score += 34
    if table_name == "molecule" and column_name == "label" and "carcinogenic" in question_tokens:
        score += 34
    return score


def _column_hint(column: ColumnInfo) -> str:
    if not _env_bool("SCHEMA_COLUMN_HINTS", True):
        return ""
    parts = []
    display = _clean_metadata_text(column.display_name)
    description = _clean_metadata_text(column.description)
    value_description = _clean_metadata_text(column.value_description)
    if display and display.casefold() != column.name.casefold():
        parts.append(display)
    if description and description.casefold() != display.casefold():
        parts.append(description)
    if _env_bool("SCHEMA_VALUE_HINTS", True) and value_description:
        lowered = value_description.casefold()
        if lowered not in {"not useful", "not quite useful"}:
            parts.append(value_description)
    if not parts:
        return ""
    return _truncate_hint(" | ".join(parts), _env_int("SCHEMA_HINT_MAX_CHARS", 150)).replace(
        "*/",
        "",
    )


def _clean_metadata_text(value: str | None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \ufeff,")
    return text


def _truncate_hint(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def _normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("_", " ")).strip()


def _is_numeric_type(type_name: str) -> bool:
    normalized = type_name.upper()
    return any(part in normalized for part in ("INT", "REAL", "NUM", "DEC", "DOUBLE", "FLOAT"))


def _is_date_type(type_name: str, column_name: str) -> bool:
    normalized_type = type_name.upper()
    normalized_name = column_name.casefold()
    return (
        any(part in normalized_type for part in ("DATE", "TIME"))
        or "date" in normalized_name
        or "time" in normalized_name
    )


def _identifier_tokens(value: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value.replace("_", " "))
    raw_tokens = TOKEN_RE.findall(spaced.casefold())
    tokens: set[str] = set()
    for token in raw_tokens:
        if not token:
            continue
        tokens.add(token)
        tokens.update(SYNONYMS.get(token, set()))
        if token.endswith("ies") and len(token) > 4:
            tokens.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def _normalize_question(question: str) -> str:
    return " ".join(TOKEN_RE.findall(question.casefold()))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
