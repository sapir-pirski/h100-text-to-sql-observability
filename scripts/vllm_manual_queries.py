#!/usr/bin/env python3
"""Run Phase 1 manual vLLM checks against BIRD eval questions.

The script calls the OpenAI-compatible vLLM chat endpoint directly, writes JSON
evidence under results/, and renders screenshots/vllm_manual_query.png when
Playwright + Chromium are available.
"""
from __future__ import annotations

import html
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from agent.schema import render_schema

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCREENSHOTS = ROOT / "screenshots"

VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
CHROMIUM = os.environ.get("CHROMIUM_PATH", "/snap/bin/chromium")

SYSTEM_PROMPT = "/no_think You are a text-to-SQL generator. Return SQL only, no markdown."


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 90.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def load_cases(limit: int = 5) -> list[dict[str, str]]:
    eval_file = ROOT / "evals" / "eval_set.jsonl"
    rows = [json.loads(line) for line in eval_file.read_text().splitlines() if line.strip()]
    if len(rows) < limit:
        raise RuntimeError(f"Need at least {limit} eval rows in {eval_file}")
    return [
        {
            "db_id": row["db_id"],
            "question": row["question"],
            "gold_sql": row.get("gold_sql", ""),
        }
        for row in rows[:limit]
    ]


def render_case_schema(db_id: str, question: str) -> str:
    try:
        return render_schema(db_id, question)  # type: ignore[call-arg]
    except TypeError:
        return render_schema(db_id)


def build_prompt(db_id: str, question: str, schema: str) -> str:
    return f"""Task: convert this eval_set question to SQLite SQL.

Database: {db_id}

Schema:
{schema}

Question:
{question}

Return only one read-only SQLite SELECT or WITH query. Do not use markdown or explain."""


def build_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 160,
    }


def extract_content(response: dict[str, Any]) -> str:
    return (((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


def run_queries() -> dict[str, Any]:
    models = request_json(f"{VLLM_URL}/models", timeout=30.0)
    queries: list[dict[str, Any]] = []
    for index, case in enumerate(load_cases(), start=1):
        schema = render_case_schema(case["db_id"], case["question"])
        prompt = build_prompt(case["db_id"], case["question"], schema)
        payload = build_payload(prompt)
        started = time.monotonic()
        response = request_json(f"{VLLM_URL}/chat/completions", payload=payload, timeout=90.0)
        latency = time.monotonic() - started
        queries.append(
            {
                "index": index,
                "db_id": case["db_id"],
                "question": case["question"],
                "gold_sql": case["gold_sql"],
                "schema": schema,
                "request": payload,
                "response": response,
                "sql": extract_content(response),
                "latency_seconds": latency,
            }
        )
    return {
        "endpoint": f"{VLLM_URL}/chat/completions",
        "model": MODEL,
        "models": models,
        "manual_query_count": len(queries),
        "queries": queries,
    }


def render_html(evidence: dict[str, Any]) -> str:
    cards = []
    for item in evidence["queries"]:
        schema_excerpt = item["schema"][:2200]
        if len(item["schema"]) > len(schema_excerpt):
            schema_excerpt += "\n..."
        cards.append(
            f"""
  <section class="query">
    <h2>Manual eval input {item["index"]}: {html.escape(item["db_id"])}</h2>
    <div class="question">{html.escape(item["question"])}</div>
    <div class="latency">Direct vLLM latency: {item["latency_seconds"]:.3f}s</div>
    <h3>SQL returned by Qwen through vLLM</h3>
    <pre class="sql">{html.escape(item["sql"])}</pre>
    <details>
      <summary>Prompt schema excerpt</summary>
      <pre>{html.escape(schema_excerpt)}</pre>
    </details>
  </section>"""
        )

    model_blob = json.dumps(evidence["models"], indent=2)[:3500]
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>vLLM manual queries evidence</title>
  <style>
    body {{ font: 16px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 24px 0 8px; font-size: 18px; }}
    h3 {{ margin: 18px 0 8px; font-size: 15px; }}
    .meta {{ color: #4b5563; margin-bottom: 20px; }}
    .query {{ border-top: 1px solid #d1d5db; padding-top: 8px; margin-top: 22px; }}
    .question {{ font-weight: 650; }}
    .latency {{ color: #4b5563; margin: 6px 0 10px; }}
    pre {{ background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 6px; padding: 14px; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .sql {{ background: #ecfdf5; border-color: #86efac; font-size: 18px; }}
    summary {{ color: #374151; cursor: pointer; margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>vLLM serving + {evidence["manual_query_count"]} manual SQL queries</h1>
  <div class="meta">Live endpoint: {html.escape(VLLM_URL)} | model: {html.escape(MODEL)} | source: evals/eval_set.jsonl</div>
  <h2>/v1/models response</h2>
  <pre>{html.escape(model_blob)}</pre>
  {"".join(cards)}
</body>
</html>"""


def write_outputs(evidence: dict[str, Any], page_html: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    evidence_json = json.dumps(evidence, indent=2)
    (RESULTS / "vllm_manual_queries_evidence.json").write_text(evidence_json)
    (RESULTS / "vllm_manual_query_evidence.json").write_text(evidence_json)
    (RESULTS / "vllm_manual_query.html").write_text(page_html)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=CHROMIUM,
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
        page.set_content(page_html, wait_until="domcontentloaded")
        page.screenshot(path=str(SCREENSHOTS / "vllm_manual_query.png"), full_page=True)
        browser.close()


def main() -> None:
    evidence = run_queries()
    page_html = render_html(evidence)
    write_outputs(evidence, page_html)
    for query in evidence["queries"]:
        print(f"[{query['index']}] {query['db_id']} {query['latency_seconds']:.3f}s")
        print(query["sql"])
    screenshot = SCREENSHOTS / "vllm_manual_query.png"
    print(f"{screenshot.relative_to(ROOT)} {screenshot.stat().st_size if screenshot.exists() else 'missing'}")


if __name__ == "__main__":
    main()
