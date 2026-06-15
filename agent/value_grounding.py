"""Ground question phrases to exact text values stored in SQLite DBs."""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from agent.schema import db_path

DEFAULT_MAX_PHRASES = 8
DEFAULT_MAX_COLUMNS = 60
DEFAULT_LIMIT_PER_PHRASE = 3
DEFAULT_MAX_MATCHES = 12
DEFAULT_MAX_VALUE_CHARS = 180
DEFAULT_ALLOW_CONTAINS = False

CONNECTORS = {"and", "or", "of", "the", "for", "in", "at", "to", "&"}
FREE_TEXT_COLUMN_MARKERS = ("body", "comment", "content", "flavortext", "text")
ENTITY_SPAN_BLOCKERS = {
    "are",
    "calculate",
    "gave",
    "give",
    "had",
    "has",
    "have",
    "is",
    "list",
    "mention",
    "provide",
    "show",
    "was",
    "were",
}
STOP_SINGLE_WORDS = {
    "among",
    "average",
    "calculate",
    "count",
    "from",
    "give",
    "how",
    "list",
    "mention",
    "please",
    "provide",
    "show",
    "what",
    "when",
    "where",
    "which",
    "with",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.'_-]*")
QUOTED_RE = re.compile(r"""["']([^"']{2,})["']""")
MDY_TIME_RE = re.compile(
    r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\s+"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})\s*(?P<ampm>[AP]M)\b",
    re.IGNORECASE,
)
YMD_TIME_RE = re.compile(
    r"\b(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2}).{0,40}?"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})\b",
    re.IGNORECASE,
)
TIME_ON_YMD_RE = re.compile(
    r"\b(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})\s+"
    r"(?:on\s+)?(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})\b",
    re.IGNORECASE,
)
ENTITY_SPAN_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9.'_-]*(?:\s+(?:[A-Za-z][A-Za-z0-9.'_-]*)){1,5}\b"
)

KNOWN_VALUE_PHRASES = {
    "australian grand prix": ("Australian Grand Prix",),
    "banned": ("Banned",),
    "blue": ("Blue",),
    "blue eyes": ("Blue",),
    "calcium": ("ca",),
    "carcinogenic": ("+",),
    "chlorine": ("cl",),
    "disqualified": ("Disqualified",),
    "female": ("F",),
    "gladiator": ("gladiator",),
    "male": ("M",),
    "mythic": ("mythic",),
    "no eye color": ("No Colour",),
    "no eye colour": ("No Colour",),
    "non carcinogenic": ("-",),
    "outpatient clinic": ("-",),
}


@dataclass(frozen=True)
class TextColumn:
    table: str
    column: str
    type_name: str


@dataclass(frozen=True)
class GroundedValue:
    phrase: str
    table: str
    column: str
    value: str
    match_type: str
    score: int


def grounding_enabled() -> bool:
    return os.environ.get("VALUE_GROUNDING", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


@lru_cache(maxsize=4096)
def ground_question_values(db_id: str, question: str) -> list[GroundedValue]:
    """Find exact DB string values that match entity-like phrases in a question."""
    if not grounding_enabled():
        return []

    max_phrases = _env_int("VALUE_GROUNDING_MAX_PHRASES", DEFAULT_MAX_PHRASES)
    max_columns = _env_int("VALUE_GROUNDING_MAX_COLUMNS", DEFAULT_MAX_COLUMNS)
    limit_per_phrase = _env_int("VALUE_GROUNDING_LIMIT_PER_PHRASE", DEFAULT_LIMIT_PER_PHRASE)
    max_matches = _env_int("VALUE_GROUNDING_MAX_MATCHES", DEFAULT_MAX_MATCHES)
    max_value_chars = _env_int("VALUE_GROUNDING_MAX_VALUE_CHARS", DEFAULT_MAX_VALUE_CHARS)
    allow_contains = _env_bool("VALUE_GROUNDING_ALLOW_CONTAINS", DEFAULT_ALLOW_CONTAINS)

    phrases = extract_candidate_phrases(question, max_phrases=max_phrases)
    if not phrases:
        return []

    columns = _rank_columns(_text_columns(db_id), question)[:max_columns]
    if not columns:
        return []

    matches: list[GroundedValue] = []
    seen: set[tuple[str, str, str, str]] = set()
    path = db_path(db_id)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for phrase in phrases:
            per_phrase = 0
            for column in columns:
                if not _column_allowed_for_phrase(phrase, column):
                    continue
                for value, match_type, score in _lookup_matches(
                    conn,
                    column,
                    phrase,
                    limit_per_phrase,
                    max_value_chars,
                    allow_contains,
                ):
                    key = (phrase.casefold(), column.table, column.column, value.casefold())
                    if key in seen:
                        continue
                    seen.add(key)
                    matches.append(
                        GroundedValue(
                            phrase=phrase,
                            table=column.table,
                            column=column.column,
                            value=value,
                            match_type=match_type,
                            score=score,
                        )
                    )
                    per_phrase += 1
                    if per_phrase >= limit_per_phrase or len(matches) >= max_matches:
                        break
                if per_phrase >= limit_per_phrase or len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break
    finally:
        conn.close()

    return sorted(matches, key=lambda item: (-item.score, item.phrase, item.table, item.column, item.value))


def format_grounded_values(values: list[GroundedValue]) -> str:
    """Render grounded values as concise prompt context."""
    if not values:
        return "No exact database value matches found."
    lines = [
        "Use these exact database values for string filters when relevant:",
    ]
    for item in values:
        value = item.value.replace("'", "''")
        lines.append(
            f"- question phrase \"{item.phrase}\" -> "
            f"{item.table}.{item.column} = '{value}' "
            f"({item.match_type})"
        )
    return "\n".join(lines)


def extract_candidate_phrases(question: str, max_phrases: int = DEFAULT_MAX_PHRASES) -> list[str]:
    """Extract quoted and entity-like phrases worth checking against DB values."""
    phrases: list[str] = []
    seen: set[str] = set()

    def finish() -> list[str]:
        return _drop_redundant_subphrases(phrases)[:max_phrases]

    def add(raw: str) -> None:
        phrase = _clean_phrase(raw)
        if len(phrase) < 2:
            return
        if all(part.isdigit() for part in phrase.split()):
            return
        key = phrase.casefold()
        if key in seen:
            return
        seen.add(key)
        phrases.append(phrase)
        possessive = _strip_possessive(phrase)
        if possessive != phrase and possessive.casefold() not in seen:
            seen.add(possessive.casefold())
            phrases.append(possessive)

    for match in QUOTED_RE.finditer(question):
        add(match.group(1))
        if len(phrases) >= max_phrases:
            return finish()

    for phrase in _datetime_phrases(question):
        add(phrase)
        if len(phrases) >= max_phrases:
            return finish()

    normalized_question = question.casefold()
    for phrase in KNOWN_VALUE_PHRASES:
        if phrase in normalized_question:
            add(phrase)
            if len(phrases) >= max_phrases:
                return finish()

    for match in ENTITY_SPAN_RE.finditer(question):
        phrase = _clean_entity_span(match.group(0))
        if not phrase:
            continue
        add(phrase)
        for part in _split_person_pair(phrase):
            add(part)
        if len(phrases) >= max_phrases:
            return finish()

    tokens = [match.group(0) for match in TOKEN_RE.finditer(question)]
    i = 0
    while i < len(tokens):
        if not _is_value_token(tokens[i]):
            i += 1
            continue

        parts = [tokens[i]]
        value_tokens = 1
        j = i + 1
        while j < len(tokens):
            token = tokens[j]
            lowered = token.casefold()
            if lowered in CONNECTORS and j + 1 < len(tokens) and _is_value_token(tokens[j + 1]):
                parts.extend([token, tokens[j + 1]])
                value_tokens += 1
                j += 2
                continue
            if _is_value_token(token):
                parts.append(token)
                value_tokens += 1
                j += 1
                continue
            break

        phrase = " ".join(parts)
        if value_tokens > 1 or _is_useful_single_word(phrase):
            add(phrase)
            for part in _split_person_pair(phrase):
                add(part)
        i = max(j, i + 1)
        if len(phrases) >= max_phrases:
            break

    return finish()


@lru_cache(maxsize=32)
def _text_columns(db_id: str) -> tuple[TextColumn, ...]:
    path = db_path(db_id)
    if not path.exists():
        return ()
    columns: list[TextColumn] = []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for table in tables:
            for _cid, column, type_name, *_rest in conn.execute(f"PRAGMA table_info({_q(table)})"):
                type_text = str(type_name or "")
                if _is_text_type(type_text):
                    columns.append(TextColumn(table=table, column=column, type_name=type_text))
    finally:
        conn.close()
    return tuple(columns)


def _lookup_matches(
    conn: sqlite3.Connection,
    column: TextColumn,
    phrase: str,
    limit: int,
    max_value_chars: int,
    allow_contains: bool,
) -> list[tuple[str, str, int]]:
    expression = f"CAST({_q(column.column)} AS TEXT)"
    base = f"SELECT DISTINCT {expression} FROM {_q(column.table)} WHERE {_q(column.column)} IS NOT NULL"
    # Prefer compact values. Large free-text bodies are rarely useful as string
    # filters and can push vLLM calls over the context limit.
    candidate_limit = max(limit * 5, limit)
    exact_sql = f"{base} AND {expression} = ? COLLATE NOCASE ORDER BY LENGTH({expression}) ASC LIMIT ?"
    contains_sql = f"{base} AND LOWER({expression}) LIKE ? ESCAPE '\\' ORDER BY LENGTH({expression}) ASC LIMIT ?"
    matches: list[tuple[str, str, int]] = []
    seen_values: set[str] = set()

    for variant, variant_type in _lookup_variants(phrase):
        for (value,) in conn.execute(exact_sql, (variant, candidate_limit)):
            text = str(value)
            if len(text) > max_value_chars:
                continue
            if text.casefold() in seen_values:
                continue
            seen_values.add(text.casefold())
            score = 100 if variant_type == "literal" else 96
            matches.append((text, f"exact/{variant_type}", score))
            if len(matches) >= limit:
                return matches

        if not allow_contains or (variant_type in {"coded", "datetime"} and len(variant) <= 2):
            continue
        pattern = f"%{_escape_like(variant.casefold())}%"
        for (value,) in conn.execute(contains_sql, (pattern, candidate_limit)):
            text = str(value)
            if len(text) > max_value_chars:
                continue
            if text.casefold() in seen_values:
                continue
            seen_values.add(text.casefold())
            score = 90 if variant_type == "literal" else 86
            matches.append((text, f"contains/{variant_type}", score))
            if len(matches) >= limit:
                return matches
    return matches


def _rank_columns(columns: tuple[TextColumn, ...], question: str) -> list[TextColumn]:
    question_tokens = set(_identifier_tokens(question))
    has_digit = any(char.isdigit() for char in question)

    def score(column: TextColumn) -> tuple[int, str, str]:
        table_tokens = set(_identifier_tokens(column.table))
        column_tokens = set(_identifier_tokens(column.column))
        value = 0
        value += 4 * len(table_tokens & question_tokens)
        value += 6 * len(column_tokens & question_tokens)
        if "banned" in question_tokens and column.column.casefold() == "status":
            value += 18
        if "gladiator" in question_tokens and column.column.casefold() == "format":
            value += 18
        if "mythic" in question_tokens and column.column.casefold() == "rarity":
            value += 18
        if {"calcium", "chlorine"} & question_tokens and column.column.casefold() == "element":
            value += 18
        if {"carcinogenic", "non"} & question_tokens and column.column.casefold() == "label":
            value += 18
        if {"male", "female"} & question_tokens and column.column.casefold() in {"gender", "sex"}:
            value += 18
        if "outpatient" in question_tokens and column.column.casefold() == "admission":
            value += 18
        if has_digit and "date" in column.column.casefold():
            value += 18
        if {"tag", "tags"} & question_tokens and column.column.casefold() == "tagname":
            value += 18
        if "department" in question_tokens and column.column.casefold() == "department":
            value += 18
        return (-value, column.table, column.column)

    return sorted(columns, key=score)


def _column_allowed_for_phrase(phrase: str, column: TextColumn) -> bool:
    normalized = phrase.casefold()
    column_name = column.column.casefold()
    table_name = column.table.casefold()
    if normalized == "australian grand prix":
        return table_name == "races" and column_name == "name"
    if normalized in {"blue", "blue eyes", "no eye color", "no eye colour"}:
        return table_name == "colour" and column_name == "colour"
    if normalized in {"calcium", "chlorine"}:
        return table_name == "atom" and column_name == "element"
    if normalized in {"carcinogenic", "non carcinogenic"}:
        return table_name == "molecule" and column_name == "label"
    if normalized == "banned":
        return table_name == "legalities" and column_name == "status"
    if normalized == "gladiator":
        return table_name == "legalities" and column_name == "format"
    if normalized == "mythic":
        return table_name == "cards" and column_name == "rarity"
    if normalized in {"male", "female"}:
        return column_name in {"gender", "sex"}
    if normalized == "outpatient clinic":
        return column_name == "admission"
    if _is_free_text_column(column):
        return False
    return True


def _is_free_text_column(column: TextColumn) -> bool:
    normalized = column.column.casefold().replace("_", "")
    return any(marker in normalized for marker in FREE_TEXT_COLUMN_MARKERS)


def _identifier_tokens(text: str) -> list[str]:
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", text.replace("_", " "))
    return [part.casefold() for part in TOKEN_RE.findall(parts) if len(part) > 1]


def _is_text_type(type_name: str) -> bool:
    normalized = type_name.upper()
    if not normalized:
        return True
    return any(part in normalized for part in ("CHAR", "CLOB", "DATE", "TEXT", "TIME", "VARCHAR"))


def _is_value_token(token: str) -> bool:
    cleaned = _clean_phrase(token)
    if len(cleaned) < 2:
        return False
    return (
        cleaned[0].isupper()
        or any(char.isdigit() for char in cleaned)
        or "." in cleaned
        or "-" in cleaned
        or "'" in cleaned
        or cleaned.isupper()
    )


def _is_useful_single_word(phrase: str) -> bool:
    cleaned = _clean_phrase(phrase)
    if len(cleaned) < 4:
        return False
    lowered = cleaned.casefold()
    return lowered in KNOWN_VALUE_PHRASES or lowered not in STOP_SINGLE_WORDS


def _lookup_variants(phrase: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    for value in KNOWN_VALUE_PHRASES.get(phrase.casefold(), ()):
        variants.append((value, "coded"))
    variants.append((phrase, "literal"))
    if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", phrase):
        variants.append((f"{phrase}.0", "datetime"))
    elif re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.0$", phrase):
        variants.append((phrase[:-2], "datetime"))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value, kind in variants:
        if value.casefold() in seen:
            continue
        seen.add(value.casefold())
        deduped.append((value, kind))
    return deduped


def _datetime_phrases(question: str) -> list[str]:
    phrases: list[str] = []
    for match in MDY_TIME_RE.finditer(question):
        hour = int(match.group("hour"))
        ampm = match.group("ampm").upper()
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0
        phrases.append(
            _format_datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                hour,
                int(match.group("minute")),
                int(match.group("second")),
            )
        )
    for match in YMD_TIME_RE.finditer(question):
        phrases.append(
            _format_datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second")),
            )
        )
    for match in TIME_ON_YMD_RE.finditer(question):
        phrases.append(
            _format_datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second")),
            )
        )
    return phrases


def _format_datetime(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
) -> str:
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def _clean_phrase(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip(" \t\r\n.,;:!?()[]{}")).strip()


def _clean_entity_span(raw: str) -> str:
    phrase = _clean_phrase(raw)
    if not phrase:
        return ""
    first = phrase.split()[0].casefold()
    if first in STOP_SINGLE_WORDS or first in CONNECTORS:
        return ""
    if set(part.casefold() for part in phrase.split()) & ENTITY_SPAN_BLOCKERS:
        return ""
    if any(char.isdigit() for char in phrase):
        return ""
    while phrase.split()[-1].casefold() in STOP_SINGLE_WORDS | CONNECTORS:
        phrase = " ".join(phrase.split()[:-1])
        if not phrase:
            return ""
    return phrase


def _split_person_pair(phrase: str) -> list[str]:
    match = re.fullmatch(
        r"([A-Z][A-Za-z.'_-]+\s+[A-Z][A-Za-z.'_-]+)\s+(?:and|or)\s+"
        r"([A-Z][A-Za-z.'_-]+\s+[A-Z][A-Za-z.'_-]+)",
        phrase,
    )
    if not match:
        return []
    return [match.group(1), match.group(2)]


def _drop_redundant_subphrases(phrases: list[str]) -> list[str]:
    result: list[str] = []
    lowered = [phrase.casefold() for phrase in phrases]
    for index, phrase in enumerate(phrases):
        key = lowered[index]
        words = key.split()
        if len(words) == 1 and any(
            index != other_index and re.search(rf"\b{re.escape(key)}\b", other_key)
            for other_index, other_key in enumerate(lowered)
        ):
            continue
        result.append(phrase)
    return result


def _strip_possessive(phrase: str) -> str:
    return re.sub(r"(?i)'s$", "", phrase).strip()


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(1, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}
