#!/usr/bin/env python3
"""Capture the Grafana vLLM dashboard from the live UI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


DEFAULT_URL = "http://localhost:3000/d/vllm-serving/vllm-serving?orgId=1&from=now-2h&to=now&refresh=5s"
DEFAULT_CHROMIUM = "/snap/bin/chromium"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=Path("screenshots/grafana_eval_run.png"))
    parser.add_argument("--chromium", default=DEFAULT_CHROMIUM)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.chromium,
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
        page.context.request.post(
            "http://localhost:3000/login",
            data=json.dumps({"user": "admin", "password": "admin"}),
            headers={"Content-Type": "application/json"},
        )
        page.goto(args.url, wait_until="networkidle", timeout=60_000)
        page.get_by_text("Scheduler state", exact=False).wait_for(timeout=30_000)
        page.get_by_text("End-to-end request latency", exact=False).wait_for(timeout=30_000)
        page.wait_for_timeout(8_000)
        page.evaluate(
            """
            () => {
              for (const el of document.querySelectorAll('.react-grid-item,[data-testid*=panel]')) {
                el.style.background = '#181b1f';
                el.style.boxShadow = 'inset 0 0 0 1px #30343b';
              }
              window.dispatchEvent(new Event('resize'));
            }
            """
        )
        if args.debug:
            page.evaluate(
                """
                () => {
                  for (const el of document.querySelectorAll('.react-grid-item,[data-panelid]')) {
                    el.style.outline = '3px solid #f97316';
                    el.style.background = 'rgba(249, 115, 22, 0.12)';
                  }
                }
                """
            )
            for selector in ["text=Scheduler state", "[data-testid*=panel]", ".react-grid-item", "[data-panelid]", "main", "body"]:
                locator = page.locator(selector)
                print(selector, "count", locator.count())
                for idx in range(min(3, locator.count())):
                    print("  box", idx, locator.nth(idx).bounding_box())
            print(
                "layout",
                page.evaluate(
                    """
                    () => ({
                      scrollHeight: document.documentElement.scrollHeight,
                      bodyScrollHeight: document.body.scrollHeight,
                      bodyText: document.body.innerText.slice(0, 800),
                    })
                    """
                ),
            )
        page.screenshot(path=str(args.out), full_page=True)
        print(args.out, args.out.stat().st_size)
        browser.close()


if __name__ == "__main__":
    main()
