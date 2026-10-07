"""Import explicitly fictional knowledge through the public HTTP API."""

import argparse
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8010")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
key = os.environ.get("RAG_API_KEY") or dotenv_values(root / ".env").get("RAG_API_KEY")
if not key:
    raise SystemExit("Сначала выполните python3 scripts/setup.py")
with httpx.Client(
    base_url=args.url, headers={"Authorization": "Bearer " + key}, timeout=300, trust_env=False
) as client:
    for path in sorted((root / "examples/knowledge").glob("*.md")):
        text = path.read_text()
        result = client.post(
            "/v1/documents",
            json={
                "source_key": "demo-" + path.stem,
                "title": text.splitlines()[0].lstrip("# "),
                "text": text,
            },
        )
        if result.is_error:
            raise SystemExit(f"{path.name}: HTTP {result.status_code}: {result.text}")
        data = result.json()
        print(
            path.name,
            "version=" + str(data["document"]["version"]),
            "updated" if data["changed"] else "unchanged",
            flush=True,
        )
