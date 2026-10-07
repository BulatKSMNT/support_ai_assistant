"""Create local configuration without printing or replacing the service secret."""

import argparse
import os
import secrets
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--demo", action="store_true", help="Use lexical search and quotations, no LLM")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
target = root / ".env"
if target.exists():
    print(".env уже существует и сохранён. Измените режим в нём вручную при необходимости.")
else:
    mode = "demo" if args.demo else "local"
    text = (
        f"RAG_API_KEY={secrets.token_urlsafe(32)}\n"
        f"RAG_DATA_DIR=./data/{mode}\n"
        f"RAG_SEARCH_MODE={'lexical' if args.demo else 'hybrid'}\n"
        f"RAG_GENERATION_MODE={'extractive' if args.demo else 'ollama'}\n"
        "RAG_OLLAMA_URL=http://127.0.0.1:11435\n"
        "RAG_CHAT_MODEL=qwen2.5:1.5b\nRAG_EMBEDDING_MODEL=embeddinggemma\n"
        "RAG_QDRANT_URL=\n"
    )
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(text)
    print(f"Создан .env ({mode}). Ключ доступа находится в RAG_API_KEY; не публикуйте .env.")
