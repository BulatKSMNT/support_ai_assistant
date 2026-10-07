import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.errors import ServiceError
from app.main import create_app
from app.storage import Repository
from tests.conftest import TEST_KEY


def embedding_response(request):
    payload = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "embeddings": [
                [1.0, 0.1, 0.0] if "возврат" not in text.lower() else [0.1, 1.0, 0.0]
                for text in payload["input"]
            ]
        },
    )


def test_real_embedded_qdrant_version_replacement_and_cleanup_failure(
    settings, document, monkeypatch
):
    settings.search_mode = "hybrid"
    with TestClient(create_app(settings, httpx.MockTransport(embedding_response))) as client:
        client.headers["Authorization"] = "Bearer " + TEST_KEY
        first = client.post("/v1/documents", json=document)
        assert first.status_code == 200, first.text
        first_hits = client.post("/v1/search", json={"query": "доставка", "mode": "vector"}).json()
        assert first_hits and first_hits[0]["version"] == 1

        async def broken_cleanup(ids):
            raise RuntimeError("Simulated vector cleanup failure")

        monkeypatch.setattr(client.app.state.assistant.vectors, "delete", broken_cleanup)
        document["text"] = "Для доставки теперь нужно уточнить город назначения и подъезд к складу."
        assert client.post("/v1/documents", json=document).status_code == 200
        hits = client.post("/v1/search", json={"query": "доставка", "mode": "vector"}).json()
        assert {hit["version"] for hit in hits} == {2}
        assert hits[0]["chunk_id"] != first_hits[0]["chunk_id"]
        assert client.delete("/v1/documents/" + first.json()["document"]["id"]).status_code == 204
        assert client.post("/v1/search", json={"query": "доставка", "mode": "vector"}).json() == []


def test_failed_embeddings_keep_previous_version(settings, document, monkeypatch):
    settings.search_mode = "hybrid"
    with TestClient(create_app(settings, httpx.MockTransport(embedding_response))) as client:
        client.headers["Authorization"] = "Bearer " + TEST_KEY
        assert client.post("/v1/documents", json=document).status_code == 200

        async def unavailable(*args, **kwargs):
            raise ServiceError("model_unavailable", "Временно недоступно.")

        monkeypatch.setattr(client.app.state.assistant.ollama, "embed", unavailable)
        document["text"] = "Новый порядок возврата товара с повреждениями упаковки."
        assert client.post("/v1/documents", json=document).status_code == 503
        records = client.get("/v1/documents").json()
        assert len(records) == 1 and records[0]["version"] == 1
        hits = client.post("/v1/search", json={"query": "доставка", "mode": "lexical"}).json()
        assert hits and "проверки наличия" in hits[0]["text"]


def test_incompatible_index_configuration_fails_loudly(tmp_path):
    Repository(tmp_path / "db.sqlite", "model-a")
    with pytest.raises(RuntimeError, match="configuration changed"):
        Repository(tmp_path / "db.sqlite", "model-b")
