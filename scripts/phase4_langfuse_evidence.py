#!/usr/bin/env python3
"""Export and capture Phase 4 Langfuse evidence.

Run on the VM where Langfuse is reachable on LANGFUSE_HOST. The script uses the
public API to select the latest phase4 traces, then captures the actual
Langfuse UI trace and trace-list pages.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from playwright.sync_api import (
    Page,
    sync_playwright,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
)

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS = ROOT / "screenshots"
RESULTS = ROOT / "results"

CHROMIUM = os.environ.get("CHROMIUM_PATH", "/snap/bin/chromium")


def read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def langfuse_json(url: str, env: dict[str, str]) -> dict[str, Any]:
    raw = f"{env['LANGFUSE_PUBLIC_KEY']}:{env['LANGFUSE_SECRET_KEY']}".encode()
    headers = {
        "Accept": "application/json",
        "Authorization": "Basic " + base64.b64encode(raw).decode("ascii"),
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def trace_nodes(trace: dict[str, Any]) -> list[str]:
    history = ((trace.get("output") or {}).get("history") or [])
    nodes = [item.get("node") for item in history if isinstance(item, dict) and item.get("node")]
    if nodes:
        return nodes
    observations = trace.get("observations") or []
    return [
        obs.get("metadata", {}).get("langgraph_node") or obs.get("name")
        for obs in observations
        if obs.get("metadata", {}).get("langgraph_node") or obs.get("name")
    ]


def trace_sort_key(trace: dict[str, Any]) -> str:
    return trace.get("timestamp") or trace.get("createdAt") or ""


def export_evidence(env: dict[str, str]) -> tuple[str, str]:
    host = env["LANGFUSE_HOST"].rstrip("/")
    projects = langfuse_json(f"{host}/api/public/projects", env).get("data", [])
    if not projects:
        raise RuntimeError("No Langfuse projects found")
    project_id = projects[0]["id"]

    since = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    traces_response = langfuse_json(
        f"{host}/api/public/traces?limit=100&fromTimestamp={since}&orderBy=timestamp.desc",
        env,
    )
    traces = traces_response.get("data", [])
    phase_traces = [
        trace
        for trace in traces
        if (trace.get("metadata") or {}).get("phase") == "langfuse-evidence"
        or "phase:langfuse-evidence" in (trace.get("tags") or [])
    ]
    phase_traces.sort(key=trace_sort_key, reverse=True)
    latest_ten = phase_traces[:10]
    if len(latest_ten) < 10:
        raise RuntimeError(f"Expected at least 10 current Phase 4 traces, found {len(latest_ten)}")

    detailed = [langfuse_json(f"{host}/api/public/traces/{trace['id']}", env) for trace in latest_ten]
    selected = next((trace for trace in detailed if "revise" in trace_nodes(trace)), detailed[0])
    observations = selected.get("observations") or []

    normalized = []
    for trace in detailed:
        normalized.append(
            {
                "id": trace.get("id"),
                "timestamp": trace.get("timestamp"),
                "createdAt": trace.get("createdAt"),
                "name": trace.get("name"),
                "tags": trace.get("tags") or [],
                "metadata": trace.get("metadata") or {},
                "db_id": (trace.get("metadata") or {}).get("db_id"),
                "request_id": (trace.get("metadata") or {}).get("request_id"),
                "nodes": trace_nodes(trace),
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "langfuse_trace_evidence.json").write_text(
        json.dumps({"trace": selected, "observations": observations}, indent=2)
    )
    (RESULTS / "langfuse_trace_list_evidence.json").write_text(json.dumps(normalized, indent=2))
    return project_id, selected["id"]


def first_visible(page: Page, selectors: list[str]):
    for selector in selectors:
        matches = page.locator(selector)
        try:
            if matches.count() == 0:
                continue
            locator = matches.first
            locator.wait_for(state="visible", timeout=1000)
            return locator
        except PlaywrightTimeoutError:
            continue
    return None


def langfuse_login(page: Page, env: dict[str, str]) -> None:
    host = env["LANGFUSE_HOST"].rstrip("/")
    page.goto(f"{host}/auth/sign-in", wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(1500)
    email_box = first_visible(page, ['input[name="email"]', 'input[type="email"]', 'input[autocomplete="email"]'])
    password_box = first_visible(page, ['input[name="password"]', 'input[type="password"]'])
    if email_box is None or password_box is None:
        return
    email_box.fill(env.get("LANGFUSE_INIT_USER_EMAIL", ""))
    password_box.fill(env.get("LANGFUSE_INIT_USER_PASSWORD", ""))
    submit = first_visible(page, ['button[type="submit"]'])
    if submit is not None:
        submit.click()
    else:
        page.keyboard.press("Enter")
    page.wait_for_timeout(3500)


def capture_langfuse(project_id: str, trace_id: str, env: dict[str, str]) -> None:
    host = env["LANGFUSE_HOST"].rstrip("/")
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=CHROMIUM,
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
        langfuse_login(page, env)

        page.goto(f"{host}/project/{project_id}/traces/{trace_id}", wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(7000)
        page.screenshot(path=str(SCREENSHOTS / "langfuse_trace.png"), full_page=True)

        page.goto(f"{host}/project/{project_id}/traces", wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(5000)
        expand_tags_filter(page)
        page.wait_for_timeout(3000)
        page.screenshot(path=str(SCREENSHOTS / "langfuse_tags.png"), full_page=True)
        browser.close()


def expand_tags_filter(page: Page) -> None:
    tag_text = page.get_by_text("Tags", exact=True)
    try:
        tag_text.first.click(timeout=2000)
    except PlaywrightTimeoutError:
        pass
    try:
        page.get_by_text(re.compile(r"phase:langfuse-evidence")).first.wait_for(timeout=4000)
    except PlaywrightTimeoutError:
        try_enable_tags_column(page)


def try_enable_tags_column(page: Page) -> None:
    try:
        page.get_by_role("button", name=re.compile(r"Columns")).click(timeout=2000)
        page.wait_for_timeout(1000)
        page.get_by_text("Tags", exact=True).first.click(timeout=2000)
        page.keyboard.press("Escape")
    except PlaywrightTimeoutError:
        return


def main() -> None:
    env = read_env_file(ROOT / ".env")
    for key, value in env.items():
        os.environ.setdefault(key, value)
    project_id, trace_id = export_evidence(env)
    capture_langfuse(project_id, trace_id, env)
    for path in [
        RESULTS / "langfuse_trace_evidence.json",
        RESULTS / "langfuse_trace_list_evidence.json",
        SCREENSHOTS / "langfuse_trace.png",
        SCREENSHOTS / "langfuse_tags.png",
    ]:
        print(f"{path.relative_to(ROOT)} {path.stat().st_size}")


if __name__ == "__main__":
    main()
