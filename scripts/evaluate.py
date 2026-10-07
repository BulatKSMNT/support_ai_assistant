"""Measure retrieval only; does not turn retrieval recall into answer accuracy."""

import argparse
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import dotenv_values

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8010")
parser.add_argument(
    "--modes",
    nargs="+",
    choices=["lexical", "vector", "hybrid"],
    default=["lexical", "vector", "hybrid"],
)
parser.add_argument("--output", default="artifacts/retrieval-report.json")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
questions = [
    json.loads(line) for line in (root / "evaluation/questions.jsonl").read_text().splitlines()
]
key = os.environ.get("RAG_API_KEY") or dotenv_values(root / ".env").get("RAG_API_KEY")
if not key:
    raise SystemExit("Missing .env; run scripts/setup.py")
report = {
    "measured_at": datetime.now(UTC).isoformat(),
    "scope": "synthetic retrieval smoke benchmark",
    "warning": "Not a held-out business evaluation; no answer quality or production claims.",
    "modes": {},
}
with httpx.Client(
    base_url=args.url, headers={"Authorization": "Bearer " + key}, timeout=300, trust_env=False
) as client:
    for mode in args.modes:
        rows, recalls, reciprocal_ranks, negatives, durations = [], [], [], [], []
        for question in questions:
            started = time.perf_counter()
            response = client.post(
                "/v1/search", json={"query": question["query"], "mode": mode, "limit": 5}
            )
            response.raise_for_status()
            latency = (time.perf_counter() - started) * 1000
            sources = response.json()
            returned = list(dict.fromkeys(source["source_key"] for source in sources))
            expected = set(question["expected_sources"])
            if expected:
                recalls.append(len(expected & set(returned)) / len(expected))
                reciprocal_ranks.append(
                    next(
                        (1 / rank for rank, source in enumerate(returned, 1) if source in expected),
                        0,
                    )
                )
            else:
                negatives.append(not returned)
            durations.append(latency)
            rows.append(
                {
                    "id": question["id"],
                    "expected": sorted(expected),
                    "returned": returned,
                    "latency_ms": round(latency, 1),
                }
            )
        ordered = sorted(durations)
        result = {
            "recall_at_5_chunks": round(statistics.mean(recalls), 4),
            "mrr_documents_within_5_chunks": round(statistics.mean(reciprocal_ranks), 4),
            "empty_result_rate_on_unrelated": round(statistics.mean(negatives), 4),
            "p50_ms": round(statistics.median(durations), 1),
            "p95_ms": round(ordered[max(0, int(len(ordered) * 0.95 + 0.999) - 1)], 1),
            "answerable_questions": len(recalls),
            "unrelated_questions": len(negatives),
            "cases": rows,
        }
        report["modes"][mode] = result
        print(mode, {k: v for k, v in result.items() if k != "cases"}, flush=True)
output = root / args.output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print("Saved:", output)
