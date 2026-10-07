"""Optional real-browser smoke test; uses an isolated, explicitly extractive instance."""

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
TEST_KEY = "browser-test-only-key-with-32-characters"
BASE = "http://127.0.0.1:8011"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

with tempfile.TemporaryDirectory(prefix="support-ui-") as directory:
    env = {
        **os.environ,
        "RAG_DATA_DIR": directory,
        "RAG_SEARCH_MODE": "lexical",
        "RAG_GENERATION_MODE": "extractive",
        "RAG_API_KEY": TEST_KEY,
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8011"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(BASE + "/health/live", timeout=1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Test service did not start")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(BASE)
            page.locator("#api-key").fill(TEST_KEY)
            page.locator("#access-form button").click()
            expect(page.locator("#service-status")).to_contain_text("без LLM")
            expect(page.locator("#document-count")).to_have_text("0")
            page.locator('[data-tab="knowledge"]').click()
            page.locator("#document-title").fill("Порядок доставки")
            page.locator("#document-key").fill("browser-delivery")
            page.locator("#document-file").set_input_files(
                {
                    "name": "delivery.md",
                    "mimeType": "text/markdown",
                    "buffer": "Для доставки уточните город и способ получения. Срок подтвердит менеджер после проверки наличия.".encode(),
                }
            )
            page.locator("#upload-button").click()
            expect(page.locator("#document-count")).to_have_text("1")
            expect(page.locator("#documents")).to_contain_text("Версия 1")
            page.locator('[data-tab="assistant"]').click()
            page.locator("#question").fill("Какие сведения нужны для доставки?")
            page.locator("#generate").click()
            expect(page.locator("#result-label")).to_contain_text("без генерации")
            expect(page.locator("#draft-text")).to_contain_text("город")
            page.locator('#draft-text a[href="#source-S1"]').click()
            page.locator("#source-S1 summary").click()
            expect(page.locator("#source-S1 pre")).to_contain_text("способ получения")
            page.screenshot(path=str(ARTIFACTS / "ui-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(ARTIFACTS / "ui-mobile.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

            page.locator('[data-tab="knowledge"]').click()
            title = '<img src=x onerror="window.__xss=1"> Документ'
            page.locator("#document-title").fill(title)
            page.locator("#document-key").fill("browser-delivery")
            page.locator("#document-file").set_input_files(
                {
                    "name": "delivery.md",
                    "mimeType": "text/markdown",
                    "buffer": "Обновлённые сведения доставки: уточните город назначения и адрес получателя.".encode(),
                }
            )
            page.locator("#upload-button").click()
            expect(page.locator("#documents")).to_contain_text("Версия 2")
            assert page.locator("#documents img").count() == 0
            assert page.evaluate("window.__xss") is None
            page.on("dialog", lambda dialog: dialog.accept())
            page.locator(".delete-button").click()
            expect(page.locator("#document-count")).to_have_text("0")
            page.locator('[data-tab="assistant"]').click()
            page.locator("#generate").click()
            expect(page.locator("#result-label")).to_have_text("Недостаточно данных")
            assert not errors, errors
            browser.close()
            report = {
                "mode": "isolated lexical/extractive",
                "passed": True,
                "checks": [
                    "authentication",
                    "upload",
                    "answer with source",
                    "source preview",
                    "mobile overflow",
                    "version update",
                    "HTML escaped",
                    "delete",
                    "empty knowledge refusal",
                ],
                "javascript_errors": errors,
            }
            (ARTIFACTS / "browser-report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps(report, ensure_ascii=False))
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
