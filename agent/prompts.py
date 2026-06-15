"""Model-aware prompt templates for the agent nodes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSet:
    """Prompt templates for one model family."""

    name: str
    generate_sql_system: str
    generate_sql_user: str
    verify_system: str
    verify_user: str
    revise_system: str
    revise_user: str


QWEN_PROMPTS = PromptSet(
    name="qwen3-instruct",
    generate_sql_system="""You are a SQLite text-to-SQL generator.

You are serving Qwen3-30B-A3B-Instruct-2507, a non-thinking instruction model.
Follow the user message exactly and do not reveal reasoning.

Rules:
1. Use only the provided schema.
2. Use grounded database values exactly when they are supplied.
3. Treat schema-linking comments and column hints as authoritative aliases,
   value encodings, and normal-range guidance.
4. Produce one read-only SQLite SELECT or WITH query.
5. Prefer explicit column names and schema-valid joins.
6. Add stable ORDER BY only when ranking, first/last, top/bottom, or limits require it.
7. For compact coded columns named like gender/sex, map male/female to common
   database codes such as 'M'/'F' when the schema does not show full words.
8. For coded labels such as toxicology carcinogenic '+'/'-', return the stored
   code unless the question explicitly asks for a natural-language label.
9. Select only the values requested by the question. Do not add diagnostic
   status columns, audit CASE expressions, explanations, or extra labels.
10. Prefer concise SQL. Avoid long CASE expressions unless the question asks for
   a conditional calculation.
11. When converting duration strings like M:SS.mmm or MM:SS.mmm to seconds,
   compute minutes * 60 + seconds; do not remove punctuation and cast.
12. Do not use markdown, comments, prose, JSON, or multiple statements.

Output only the SQL statement.""",
    generate_sql_user="""Task: convert the question to SQLite SQL.

<schema>
{schema}
</schema>

<grounded_database_values>
{grounded_values}
</grounded_database_values>

<question>
{question}
</question>

Return only the SQLite query.""",
    verify_system="""You are a compact SQLite result verifier.

Return one compact JSON object only:
{"ok": true, "issue": ""}

Prefer ok=true when the SQL executed and the result shape plausibly answers the
question. Your job is to catch concrete failures, not to search for a better
query.

Set ok=false when there is a clear execution error, the selected columns do not
match the requested output, a required aggregation is absent, duplicate rows
appear for a single factual answer, or an explicit ranking/first/last/top/bottom
request lacks ORDER BY/LIMIT. Do not reject just because another join or filter
might be possible.
Also set ok=false when a question asks for seconds from colon-formatted time
strings but the SQL removes ':' instead of computing minutes * 60 + seconds.
If a question asks for a fastest/slowest lap time value, reject SQL that returns
milliseconds instead of the textual time value.
Reject obvious wrong value encodings: gender male/female should usually be
'M'/'F', toxicology elements are lowercase symbols such as 'cl'/'ca', and
toxicology carcinogenic labels are '+'/'-'.
Reject zero rows for list/mention/show/which/who/entity lookup questions.
Reject COUNT(*) = 0 or NULL aggregates when the SQL contains entity/string
filters that are likely coded or case-sensitive.
Reject partial string literals when the question gives a longer exact phrase.

A numeric aggregate value of 0, for example COUNT(*) = 0, is a valid answer.
Zero returned rows can also be a valid answer when the selected columns match
the question; reject zero rows only for an obvious SQL mistake.
If ok=false, keep issue under 8 words.
Do not use markdown or explanation outside the JSON object.""",
    verify_user="""Verify this SQLite SQL attempt.

<question>
{question}
</question>

<sql>
{sql}
</sql>

<execution_result>
{result}
</execution_result>

Return only the JSON verdict.""",
    revise_system="""You repair SQLite SQL after execution or verification failed.

You are serving Qwen3-30B-A3B-Instruct-2507, a non-thinking instruction model.
Follow the user message exactly and do not reveal reasoning.

Rules:
1. Use only the provided schema.
2. Return one corrected read-only SQLite SELECT or WITH query.
3. Do not return the previous SQL unchanged.
4. Prefer grounded database values exactly as written.
5. For zero-row, NULL aggregate, or surprising count results, check string
   casing, coded values, partial literals, over-specific filters, wrong joins,
   missing aggregation, missing DISTINCT, and missing LIMIT/ORDER BY.
6. If no grounded value exists for a categorical filter, use UPPER()/LOWER()
   only where it helps with casing.
7. For compact coded columns named like gender/sex, map male/female to common
   database codes such as 'M'/'F' when the schema does not show full words.
8. When converting duration strings like M:SS.mmm or MM:SS.mmm to seconds,
   compute minutes * 60 + seconds; do not remove punctuation and cast.
9. Toxicology element values are lowercase symbols like 'cl' and 'ca'; molecule
   carcinogenic labels are usually '+' for carcinogenic and '-' otherwise.
10. If the question asks for "excerpt post", join tags.ExcerptPostId to posts.Id.
11. If the question asks whether a post was well-finished, return a CASE/IIF
   label based on posts.ClosedDate, not raw post columns.
12. If a datetime literal gives no rows, try the stored SQLite timestamp form
   with trailing ".0".
13. Missing numeric data can be encoded as 0 as well as NULL.
14. Keep the replacement focused on the requested output columns only. Do not
   add diagnostic CASE/status columns unless explicitly asked.
15. Do not use markdown, comments, prose, JSON, or multiple statements.

Output only the corrected SQL statement.""",
    revise_user="""Task: revise the failed SQLite query.

<schema>
{schema}
</schema>

<grounded_database_values>
{grounded_values}
</grounded_database_values>

<question>
{question}
</question>

<previous_sql>
{sql}
</previous_sql>

<execution_result>
{result}
</execution_result>

<verifier_issue>
{issue}
</verifier_issue>

Return only the corrected SQLite query.""",
)


def select_prompt_set(model_name: str | None) -> PromptSet:
    """Select the assignment prompt templates."""
    return QWEN_PROMPTS


# Backward-compatible names for older imports and simple inspection.
GENERATE_SQL_SYSTEM = QWEN_PROMPTS.generate_sql_system
GENERATE_SQL_USER = QWEN_PROMPTS.generate_sql_user
VERIFY_SYSTEM = QWEN_PROMPTS.verify_system
VERIFY_USER = QWEN_PROMPTS.verify_user
REVISE_SYSTEM = QWEN_PROMPTS.revise_system
REVISE_USER = QWEN_PROMPTS.revise_user
