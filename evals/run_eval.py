"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"
REQUEST_TIMEOUT_SECONDS = 180.0


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


def normalize_sql_text(sql: str | None) -> str:
    """Normalize SQL text for rough exact-match diagnostics."""
    if not sql:
        return ""
    normalized = re.sub(r"\s+", " ", sql.strip())
    return normalized[:-1].strip().lower() if normalized.endswith(";") else normalized.lower()


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question.get("db_id") or question.get("db")
    question_text = question["question"]
    gold_sql = question.get("gold_sql") or question.get("SQL")

    gold_ok, gold_rows, gold_error = run_sql(db_id, gold_sql)
    started = time.monotonic()
    agent_error: str | None = None
    agent_payload: dict | None = None

    try:
        response = httpx.post(
            agent_url,
            json={
                "question": question_text,
                "db": db_id,
                "tags": {"phase": "eval", "db_id": db_id},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        agent_payload = response.json()
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"

    latency = time.monotonic() - started
    history = (agent_payload or {}).get("history", [])
    attempts = [
        {"source": item.get("node", "unknown"), "sql": item.get("sql", "")}
        for item in history
        if item.get("node") in {"generate_sql", "revise"} and item.get("sql")
    ]
    final_sql = (agent_payload or {}).get("sql", "")
    if not attempts and final_sql:
        attempts = [{"source": "final", "sql": final_sql}]
    elif final_sql and (not attempts or attempts[-1]["sql"] != final_sql):
        attempts.append({"source": "final", "sql": final_sql})

    attempt_results = []
    for i, attempt in enumerate(attempts):
        pred_ok, pred_rows, pred_error = run_sql(db_id, attempt["sql"])
        correct = gold_ok and pred_ok and matches(gold_rows, pred_rows)
        attempt_results.append({
            "iteration": i,
            "source": attempt["source"],
            "sql": attempt["sql"],
            "sql_ok": pred_ok,
            "sql_error": pred_error,
            "row_count": len(pred_rows) if pred_rows is not None else None,
            "correct": correct,
        })

    final_correct = attempt_results[-1]["correct"] if attempt_results else False
    return {
        "question": question_text,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "gold_sql_ok": gold_ok,
        "gold_sql_error": gold_error,
        "agent_error": agent_error,
        "agent_latency_seconds": latency,
        "agent_iterations": (agent_payload or {}).get("iterations", 0),
        "final_sql": final_sql,
        "final_correct": final_correct,
        "attempts": attempt_results,
        "history": history,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    final_correct = sum(1 for r in results if r.get("final_correct"))
    max_attempts = max((len(r.get("attempts", [])) for r in results), default=0)

    per_iteration = []
    for i in range(max_attempts):
        correct_at_i = 0
        attempted_at_i = 0
        for r in results:
            attempts = r.get("attempts", [])
            if not attempts:
                continue
            attempted_at_i += 1
            carried = attempts[min(i, len(attempts) - 1)]
            correct_at_i += int(bool(carried.get("correct")))
        per_iteration.append({
            "iteration": i,
            "correct": correct_at_i,
            "total": total,
            "accuracy": (correct_at_i / total) if total else 0.0,
            "questions_with_sql": attempted_at_i,
        })

    latencies = [
        float(r["agent_latency_seconds"])
        for r in results
        if r.get("agent_error") is None and r.get("agent_latency_seconds") is not None
    ]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    avg_iterations = (
        sum(int(r.get("agent_iterations", 0)) for r in results) / total
        if total else 0.0
    )

    return {
        "total": total,
        "correct": final_correct,
        "accuracy": (final_correct / total) if total else 0.0,
        "per_iteration": per_iteration,
        "agent_errors": sum(1 for r in results if r.get("agent_error")),
        "gold_sql_errors": sum(1 for r in results if not r.get("gold_sql_ok")),
        "avg_latency_seconds": avg_latency,
        "avg_agent_iterations": avg_iterations,
        "diagnostics": diagnostic_metrics(results),
    }


def diagnostic_metrics(results: list[dict]) -> dict:
    """Secondary diagnostics. Execution accuracy remains the eval signal."""
    total = len(results)
    attempted = [r for r in results if r.get("attempts")]
    final_attempts = [r["attempts"][-1] for r in attempted]

    valid_sql = sum(1 for attempt in final_attempts if attempt.get("sql_ok"))
    non_empty = sum(
        1
        for attempt in final_attempts
        if attempt.get("sql_ok") and (attempt.get("row_count") or 0) > 0
    )
    exact_sql = sum(
        1
        for r in attempted
        if normalize_sql_text(r.get("final_sql")) == normalize_sql_text(r.get("gold_sql"))
    )

    error_categories: dict[str, int] = {}
    for attempt in final_attempts:
        if attempt.get("sql_ok"):
            continue
        category = _error_category(attempt.get("sql_error"))
        error_categories[category] = error_categories.get(category, 0) + 1

    attempt_counts: dict[str, int] = {}
    for r in results:
        key = str(len(r.get("attempts", [])))
        attempt_counts[key] = attempt_counts.get(key, 0) + 1

    return {
        "valid_sql": valid_sql,
        "valid_sql_rate": (valid_sql / total) if total else 0.0,
        "non_empty_result": non_empty,
        "non_empty_result_rate": (non_empty / total) if total else 0.0,
        "exact_sql_match": exact_sql,
        "exact_sql_match_rate": (exact_sql / total) if total else 0.0,
        "attempt_count_distribution": dict(sorted(attempt_counts.items(), key=lambda item: int(item[0]))),
        "sql_error_categories": dict(sorted(error_categories.items())),
    }


def _error_category(error: str | None) -> str:
    if not error:
        return "unknown"
    head = error.split(":", 1)[0].strip()
    return head or "unknown"


def summarize_by_db(results: list[dict]) -> dict[str, dict]:
    """Aggregate the same eval metrics independently for each database."""
    grouped: dict[str, list[dict]] = {}
    for result in results:
        db_id = str(result.get("db_id") or "unknown")
        grouped.setdefault(db_id, []).append(result)

    by_db: dict[str, dict] = {}
    for db_id in sorted(grouped):
        db_summary = summarize(grouped[db_id])
        db_summary["iteration_accuracy"] = {
            str(item["iteration"]): item["accuracy"]
            for item in db_summary.get("per_iteration", [])
        }
        by_db[db_id] = db_summary
    return by_db


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    by_db = summarize_by_db(results)
    out = {
        "summary": summary,
        "by_db": by_db,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))
    print(json.dumps({"by_db": by_db}, indent=2))


if __name__ == "__main__":
    main()
