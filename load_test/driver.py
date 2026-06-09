"""Async load driver for the agent endpoint.

Samples questions from load_test/perf_pool.jsonl and fires them at the
agent at the requested RPS for the requested duration, recording per-
request latency and outcome.

Run:
    uv run python load_test/driver.py --rps 8 --duration 300

Writes a JSON file (default results/load_test.json) with summary + raw
per-request data.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
PERF_POOL = ROOT / "load_test" / "perf_pool.jsonl"
DEFAULT_OUT = ROOT / "results" / "load_test.json"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"
RESPONSE_PREVIEW_CHARS = 500
QUESTION_PREVIEW_CHARS = 180


def preview_text(value: object, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    k = int(round(p * (len(sorted_values) - 1)))
    return sorted_values[k]


def latency_summary(results: Iterable[dict], statuses: set[str] | None = None) -> dict:
    latencies = sorted(
        float(r["latency_seconds"])
        for r in results
        if isinstance(r.get("latency_seconds"), int | float)
        and (statuses is None or r.get("status") in statuses)
    )
    return {
        "count": len(latencies),
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
        "p99": percentile(latencies, 0.99),
        "max": latencies[-1] if latencies else None,
    }


def non_ok_samples(results: list[dict], limit: int) -> list[dict]:
    fields = (
        "request_index",
        "db_id",
        "question_id",
        "question_preview",
        "status",
        "status_code",
        "error_type",
        "error",
        "response_body_preview",
        "latency_seconds",
    )
    samples: list[dict] = []
    for result in results:
        if result.get("status") == "ok":
            continue
        samples.append({field: result[field] for field in fields if result.get(field) is not None})
        if len(samples) >= limit:
            break
    return samples


def summarize_results(
    results: list[dict],
    target_rps: float,
    duration_seconds: int | float,
    wall_clock_seconds: float,
    non_ok_sample_limit: int = 20,
) -> dict:
    issued_requests = len(results)
    outcome_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        outcome_counts[status] = outcome_counts.get(status, 0) + 1

    ok = outcome_counts.get("ok", 0)
    non_ok = issued_requests - ok
    ok_latency = latency_summary(results, {"ok"})
    completed_latency = latency_summary(results)

    summary = {
        "target_rps": target_rps,
        "duration_seconds": duration_seconds,
        "wall_clock_seconds": wall_clock_seconds,
        "actual_wall_time_seconds": wall_clock_seconds,
        "issued_requests": issued_requests,
        "issued_rps": (issued_requests / duration_seconds) if duration_seconds > 0 else 0.0,
        "actual_issued_rps": (issued_requests / wall_clock_seconds) if wall_clock_seconds > 0 else 0.0,
        "ok": ok,
        "non_ok": non_ok,
        "success_rate": (ok / issued_requests) if issued_requests else 0.0,
        "non_ok_rate": (non_ok / issued_requests) if issued_requests else 0.0,
        "outcome_counts": outcome_counts,
        "latency_ok_only": ok_latency,
        "latency_all_completed": completed_latency,
        "non_ok_samples": non_ok_samples(results, non_ok_sample_limit),
    }

    # Backward-compatible aliases used by earlier reports and scripts.
    summary.update(
        {
            "requested_rps": target_rps,
            "total_requests": issued_requests,
            "achieved_rps": summary["actual_issued_rps"],
            "ok_rate": summary["success_rate"],
            "timeouts": outcome_counts.get("timeout", 0),
            "http_errors": outcome_counts.get("http_error", 0),
            "client_errors": outcome_counts.get("client_error", 0),
            "latency_p50": ok_latency["p50"],
            "latency_p95": ok_latency["p95"],
            "latency_p99": ok_latency["p99"],
            "latency_max": ok_latency["max"],
        }
    )
    return summary


def build_schedule(
    questions: Sequence[dict],
    rps: float,
    duration_seconds: int | float,
    seed: int,
) -> list[tuple[int, float, dict]]:
    if rps <= 0:
        raise ValueError("--rps must be greater than zero")
    if duration_seconds <= 0:
        return []

    rnd = random.Random(seed)
    interval = 1.0 / rps
    request_count = int(rps * duration_seconds)
    return [
        (index + 1, index * interval, rnd.choice(questions))
        for index in range(request_count)
    ]


def unique_scheduled_questions(schedule: Sequence[tuple[int, float, dict]]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    selected: list[dict] = []
    for _index, _scheduled_at, question in schedule:
        key = (str(question.get("db_id", "")), " ".join(str(question.get("question", "")).split()))
        if key in seen:
            continue
        seen.add(key)
        selected.append(question)
    return selected


async def fire_one(
    session: aiohttp.ClientSession,
    url: str,
    question: dict,
    results: list[dict],
    request_index: int,
    scheduled_at_relative_seconds: float,
    request_timeout_seconds: float,
    tags: dict[str, str] | None = None,
) -> None:
    payload = {"question": question["question"], "db": question["db_id"], "tags": tags or {}}
    t0 = time.monotonic()
    status = "ok"
    status_code: int | None = None
    error_type: str | None = None
    err: str | None = None
    response_body_preview: str | None = None
    try:
        timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            status_code = resp.status
            body = await resp.read()
            if resp.status < 200 or resp.status >= 300:
                status = "http_error"
                error_type = "HTTPStatusError"
                err = f"HTTP {resp.status}"
                response_body_preview = preview_text(
                    body.decode("utf-8", errors="replace"),
                    RESPONSE_PREVIEW_CHARS,
                )
    except TimeoutError:
        status = "timeout"
        error_type = "TimeoutError"
        err = f"request timed out after {request_timeout_seconds:g}s"
    except aiohttp.ClientError as e:
        status = "client_error"
        error_type = type(e).__name__
        err = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        status = "client_error"
        error_type = type(e).__name__
        err = f"{type(e).__name__}: {e}"
    results.append({
        "request_index": request_index,
        "db_id": question.get("db_id"),
        "question_id": question.get("question_id"),
        "question_preview": preview_text(question.get("question"), QUESTION_PREVIEW_CHARS),
        "scheduled_at_relative_seconds": scheduled_at_relative_seconds,
        "latency_seconds": time.monotonic() - t0,
        "status": status,
        "status_code": status_code,
        "error_type": error_type,
        "error": err,
        "response_body_preview": response_body_preview,
    })


async def warmup_unique_questions(
    session: aiohttp.ClientSession,
    url: str,
    questions: Sequence[dict],
    concurrency: int,
    request_timeout_seconds: float,
    retries: int,
    non_ok_sample_limit: int,
    tags: dict[str, str],
) -> dict:
    started = time.monotonic()
    all_results: list[dict] = []
    sem = asyncio.Semaphore(max(1, concurrency))

    remaining = list(questions)
    attempts = max(0, retries) + 1
    for attempt in range(1, attempts + 1):
        if not remaining:
            break
        attempt_results: list[dict] = []

        async def run_one(
            index: int,
            question: dict,
            target_results: list[dict] = attempt_results,
        ) -> None:
            async with sem:
                await fire_one(
                    session,
                    url,
                    question,
                    target_results,
                    index,
                    time.monotonic() - started,
                    request_timeout_seconds,
                    tags,
                )

        await asyncio.gather(*(run_one(index, q) for index, q in enumerate(remaining, start=1)))
        for result in attempt_results:
            result["warmup_attempt"] = attempt
        all_results.extend(attempt_results)

        status_by_index = {
            int(result["request_index"]): result.get("status")
            for result in attempt_results
            if isinstance(result.get("request_index"), int)
        }
        remaining = [
            question
            for index, question in enumerate(remaining, start=1)
            if status_by_index.get(index) != "ok"
        ]

    wall = time.monotonic() - started
    summary = summarize_results(
        all_results,
        target_rps=(len(all_results) / wall) if wall > 0 else 0.0,
        duration_seconds=wall,
        wall_clock_seconds=wall,
        non_ok_sample_limit=non_ok_sample_limit,
    )
    summary["unique_requests"] = len(questions)
    summary["warmup_concurrency"] = concurrency
    summary["warmup_retries"] = retries
    summary["warmup_attempts"] = min(attempts, max((r["warmup_attempt"] for r in all_results), default=0))
    summary["unresolved_requests"] = len(remaining)
    return summary


async def drive(args: argparse.Namespace) -> None:
    if not PERF_POOL.exists():
        raise SystemExit(f"{PERF_POOL} not found - run scripts/load_data.py first")
    questions = [json.loads(line) for line in PERF_POOL.read_text().splitlines() if line.strip()]
    if not questions:
        raise SystemExit(f"{PERF_POOL} is empty")

    schedule = build_schedule(questions, args.rps, args.duration, args.schedule_seed)
    results: list[dict] = []
    interval = 1.0 / args.rps
    warmup_summary: dict | None = None
    tags = {
        "phase": "load",
        "runner": "load_test",
        "target_rps": str(args.rps),
        "duration_seconds": str(args.duration),
    }

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        if args.warmup_unique:
            warmup_questions = unique_scheduled_questions(schedule)
            warmup_summary = await warmup_unique_questions(
                session,
                args.agent_url,
                warmup_questions,
                args.warmup_concurrency,
                args.warmup_timeout_seconds,
                args.warmup_retries,
                args.non_ok_sample_limit,
                {**tags, "phase": "load-warmup"},
            )

        start = time.monotonic()
        tasks: list[asyncio.Task] = []
        next_fire = start
        for request_index, scheduled_at, q in schedule:
            tasks.append(
                asyncio.create_task(
                    fire_one(
                        session,
                        args.agent_url,
                        q,
                        results,
                        request_index,
                        scheduled_at,
                        args.request_timeout_seconds,
                        tags,
                    )
                )
            )
            next_fire += interval
            sleep_for = next_fire - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        if tasks:
            await asyncio.gather(*tasks)
        wall = time.monotonic() - start

    summary = summarize_results(
        results,
        args.rps,
        args.duration,
        wall,
        non_ok_sample_limit=args.non_ok_sample_limit,
    )
    if warmup_summary is not None:
        summary["warmup"] = warmup_summary

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rps", type=float, default=8.0, help="target requests/second")
    p.add_argument("--duration", type=int, default=300, help="seconds to drive load")
    p.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--request-timeout-seconds", type=float, default=120.0)
    p.add_argument("--non-ok-sample-limit", type=int, default=20)
    p.add_argument("--schedule-seed", type=int, default=0)
    p.add_argument(
        "--warmup-unique",
        action="store_true",
        help="preload one request for every unique DB/question pair in the measured schedule",
    )
    p.add_argument("--warmup-concurrency", type=int, default=16)
    p.add_argument("--warmup-timeout-seconds", type=float, default=120.0)
    p.add_argument("--warmup-retries", type=int, default=0)
    args = p.parse_args()
    asyncio.run(drive(args))


if __name__ == "__main__":
    main()
