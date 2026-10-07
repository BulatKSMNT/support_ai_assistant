"""Live Ollama smoke checks against the fictional demo knowledge base."""

import argparse
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8010")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
key = os.environ.get("RAG_API_KEY") or dotenv_values(root / ".env").get("RAG_API_KEY")
if not key:
    raise SystemExit("Run scripts/setup.py first")
cases = [
    ("inquiry", json.loads((root / "examples/inquiry.json").read_text())),
    (
        "unknown_price",
        {"question": "Назови точную текущую цену Модуля А и сколько штук осталось на складе."},
    ),
    ("unrelated", {"question": "Кто написал роман Евгений Онегин?"}),
]
results = []
with httpx.Client(
    base_url=args.url, headers={"Authorization": "Bearer " + key}, timeout=400, trust_env=False
) as client:
    for name, payload in cases:
        print("Starting:", name, flush=True)
        started = time.perf_counter()
        response = client.post("/v1/drafts", json=payload)
        result = {
            "case": name,
            "http_status": response.status_code,
            "wall_seconds": round(time.perf_counter() - started, 2),
            "response": response.json(),
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "case": name,
                    "http_status": response.status_code,
                    "seconds": result["wall_seconds"],
                    "status": result["response"].get("status"),
                    "draft": result["response"].get("draft"),
                    "missing": result["response"].get("missing_information"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
output = root / "artifacts/live-generation.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
assert all(row["http_status"] == 200 for row in results), "Live request failed"
assert results[0]["response"]["status"] == "ready", "Expected a sourced draft"
draft = results[0]["response"]["draft"].lower()
assert not any(word in draft for word in ["симптом", "неисправност", "серийный номер"]), (
    "New order draft mixed with repair scenario"
)
assert results[1]["response"]["status"] == "insufficient_evidence", "Invented current values"
assert results[2]["response"]["status"] == "insufficient_evidence", (
    "Unrelated question was answered"
)
print("Live smoke checks passed. This is not a statistical evaluation of answer quality.")
